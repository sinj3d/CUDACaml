(* T27: device selection and the data-parallel driver. GPU-dependent; the
   true multi-device assertions run only when two or more devices exist. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module Multi = Backend_cuda.Multi
module Bs = Ocaml_cuda_examples.Black_scholes

let vec n = Shape.of_dims [ n ]
let f32s n f = Value.P (Value.of_list Dtype.F32 (vec n) (List.init n f))
let live () = Runtime.Buffer.live_count ()

let floats (Value.P v) : float list =
  match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> C.fail "dtype"

let ints (Value.P v) : int32 list = match Value.dtype v with Dtype.I32 -> Value.to_list v | _ -> C.fail "dtype"
let scalar o name = List.hd (floats (List.assoc name o))
let saxpy ~n = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0

(* i32 program with negative division, as in S6 *)
let i32_prog ~n =
  let x = param "x" Dtype.I32 (vec n) in
  let r = map (fun e -> div (sub e (const Dtype.I32 4l)) (const Dtype.I32 2l)) x in
  Graph.create ~name:"i32" ~outputs:[ ("r", Tensor.P r) ]

let () =
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  let n = 8192 in
  let inputs = [ ("x", f32s n float_of_int); ("y", f32s n (fun i -> float_of_int (n - i))) ] in
  C.test "device enumeration and with_device" (fun () ->
      C.bool ~expect:true (Runtime.Device.count () >= 1);
      C.int ~expect:5 (Runtime.Device.with_device 0 (fun () -> 5));
      C.int ~expect:0 (Runtime.Device.current ());
      C.bool ~expect:true (String.length (Runtime.Device.info_of 0).name > 0));
  C.test "Multi on [0; 0]: concat and mean_scalar reproduce the single-device run" (fun () ->
      let c = Backend_cuda.compile (saxpy ~n) in
      let single = Backend_cuda.run c ~inputs in
      Backend_cuda.release c;
      let before = live () in
      let t = Multi.create ~devices:[ 0; 0 ] ~n saxpy in
      let per = Multi.run t ~inputs in
      C.int ~expect:2 (List.length per);
      C.floats ~tol:0.0 ~expect:(floats (List.assoc "r" single)) (floats (Multi.concat (List.map (fun o -> List.assoc "r" o) per)));
      let s = scalar single "s" in
      C.float ~tol:(1e-5 *. Float.abs s) ~expect:s (2.0 *. Multi.mean_scalar (List.map (fun o -> List.assoc "s" o) per));
      Multi.release t;
      C.int ~expect:before (live ()));
  C.test "per-device inputs: one seed per device" (fun () ->
      let t = Multi.create ~devices:[ 0; 0 ] ~n:(1 lsl 17) (fun ~n -> Bs.call_mc ~n_paths:n ~dtype:Dtype.F32 Bs.market) in
      let per = Multi.run_per_device t ~inputs:(fun d -> Bs.inputs ~dtype:Dtype.F32 Bs.market ~seed:(Int32.of_int (d + 1))) in
      let prices = List.map (fun o -> Bs.scalar_out o "price") per in
      let cf = Bs.analytic Bs.market in
      List.iter (fun p -> C.float ~tol:1.0 ~expect:cf.price p) prices;
      C.bool ~expect:true (List.nth prices 0 <> List.nth prices 1);
      Multi.release t);
  if Runtime.Device.count () >= 2 then begin
    C.test "compile_on device 1 runs saxpy" (fun () ->
        let c = Backend_cuda.compile_on ~device:1 (saxpy ~n) in
        C.int ~expect:1 (Backend_cuda.device_of c);
        let want = Backend_cuda.run (Backend_cuda.compile (saxpy ~n)) ~inputs in
        C.floats ~tol:1e-6 ~expect:(floats (List.assoc "r" want)) (floats (List.assoc "r" (Backend_cuda.run c ~inputs)));
        Backend_cuda.release c;
        C.bool ~expect:true (String.length (Runtime.Device.info_of 1).name > 0));
    C.test "Multi on [0; 1] equals [0; 0]" (fun () ->
        let xin = [ ("x", Value.P (Value.of_list Dtype.I32 (vec n) (List.init n (fun i -> Int32.of_int (i - 100))))) ] in
        let a = Multi.create ~devices:[ 0; 0 ] ~n i32_prog and b = Multi.create ~devices:[ 0; 1 ] ~n i32_prog in
        let ra = Multi.concat (List.map (fun o -> List.assoc "r" o) (Multi.run a ~inputs:xin)) in
        let rb = Multi.concat (List.map (fun o -> List.assoc "r" o) (Multi.run b ~inputs:xin)) in
        C.bool ~expect:true (List.for_all2 Int32.equal (ints ra) (ints rb));
        let fa = Multi.create ~devices:[ 0; 0 ] ~n saxpy and fb = Multi.create ~devices:[ 0; 1 ] ~n saxpy in
        let sa = Multi.concat (List.map (fun o -> List.assoc "r" o) (Multi.run fa ~inputs)) in
        let sb = Multi.concat (List.map (fun o -> List.assoc "r" o) (Multi.run fb ~inputs)) in
        C.floats ~tol:1e-6 ~expect:(floats sa) (floats sb);
        List.iter Multi.release [ a; b; fa; fb ])
  end
  else C.skip "fewer than 2 devices: multi-device assertions not run";
  C.run ()
