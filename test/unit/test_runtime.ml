(* Runtime over cudajit. SKIPS without a device; never fails for lack
   of one. With a device it runs a hand-written kernel end to end, which is
   the single most important de-risking test in the project. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module Rt = Runtime

let src =
  {|
extern "C" __global__ void saxpy(float* x, float* y, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 1000; i += gridDim.x * blockDim.x)
    out[i] = 2.0f * x[i] + y[i];
}
|}

let () =
  if not (Rt.Device.available ()) then begin
    C.skip "no CUDA device";
    C.run ()
  end;
  C.test "init is idempotent" (fun () ->
      Rt.Device.init ();
      Rt.Device.init ());
  C.test "device has a name" (fun () -> C.bool ~expect:true (String.length (Rt.Device.name ()) > 0));
  C.test "alloc / free / byte_size" (fun () ->
      let b = Rt.Buffer.alloc ~bytes:4096 in
      C.int ~expect:4096 (Rt.Buffer.byte_size b);
      Rt.Buffer.free b);
  C.test "upload then download roundtrips" (fun () ->
      let v = Value.of_list Dtype.I32 (Shape.of_dims [ 4 ]) [ 1l; 2l; 3l; 4l ] in
      let b = Rt.Buffer.alloc ~bytes:16 in
      Rt.Buffer.upload (Value.P v) b;
      let w = Value.create Dtype.I32 (Shape.of_dims [ 4 ]) in
      Rt.Buffer.download b (Value.P w);
      C.bool ~expect:true (Value.to_list w = [ 1l; 2l; 3l; 4l ]);
      Rt.Buffer.free b);
  C.test "upload with mismatched size raises" (fun () ->
      let v = Value.create Dtype.F32 (Shape.of_dims [ 4 ]) in
      let b = Rt.Buffer.alloc ~bytes:8 in
      C.raises (fun () -> Rt.Buffer.upload (Value.P v) b);
      Rt.Buffer.free b);
  C.test "compile produces PTX and exposes the kernel by name" (fun () ->
      let m = Rt.Jit.compile ~name:"saxpy" ~source:src in
      C.contains ~sub:".entry saxpy" (Rt.Jit.ptx m);
      ignore (Rt.Jit.get_kernel m "saxpy");
      C.raises (fun () -> ignore (Rt.Jit.get_kernel m "nope"));
      Rt.Jit.unload m);
  C.test "compile error raises" (fun () ->
      C.raises (fun () -> ignore (Rt.Jit.compile ~name:"bad" ~source:"this is not cuda")));
  C.test "hand-written saxpy runs end to end" (fun () ->
      let n = 1000 in
      let x = Value.of_list Dtype.F32 (Shape.of_dims [ n ]) (List.init n float_of_int) in
      let y = Value.of_list Dtype.F32 (Shape.of_dims [ n ]) (List.init n (fun i -> float_of_int (n - i))) in
      let out = Value.create Dtype.F32 (Shape.of_dims [ n ]) in
      let bx = Rt.Buffer.alloc ~bytes:(4 * n) in
      let by = Rt.Buffer.alloc ~bytes:(4 * n) in
      let bo = Rt.Buffer.alloc ~bytes:(4 * n) in
      Rt.Buffer.upload (Value.P x) bx;
      Rt.Buffer.upload (Value.P y) by;
      let m = Rt.Jit.compile ~name:"saxpy" ~source:src in
      Rt.Launch.run (Rt.Jit.get_kernel m "saxpy") ~grid:4 ~block:256 ~shared_bytes:0 [ bx; by; bo ];
      Rt.Device.synchronize ();
      Rt.Buffer.download bo (Value.P out);
      C.floats
        ~expect:(List.init n (fun i -> (2.0 *. float_of_int i) +. float_of_int (n - i)))
        (Value.to_list out);
      List.iter Rt.Buffer.free [ bx; by; bo ];
      Rt.Jit.unload m);
  C.run ()
