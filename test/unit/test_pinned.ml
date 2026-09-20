(* Pinned host memory and Value.of_raw. GPU-dependent. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check

let vec n = Shape.of_dims [ n ]

let floats (Value.P v) : float list =
  match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> C.fail "dtype"

let scalar o name = List.hd (floats (List.assoc name o))

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

let () =
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  C.test "alloc: writable, readable, pinned" (fun () ->
      let v = Runtime.Pinned.alloc Dtype.F32 (vec 1000) in
      C.int ~expect:1000 (Value.numel v);
      for i = 0 to 999 do Value.set v i (float_of_int i) done;
      C.float ~tol:0.0 ~expect:999.0 (Value.get v 999);
      C.bool ~expect:true (Runtime.Pinned.is_pinned v);
      C.bool ~expect:false (Runtime.Pinned.is_pinned (Value.create Dtype.F32 (vec 4))));
  C.test "of_raw: kind and length checks; shared storage" (fun () ->
      let ba64 = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 4 in
      C.raises (fun () -> ignore (Value.of_raw Dtype.F32 (vec 4) (Value.Raw ba64)));
      let ba32 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 4 in
      C.raises (fun () -> ignore (Value.of_raw Dtype.F32 (vec 5) (Value.Raw ba32)));
      let v = Value.of_raw Dtype.F32 (vec 4) (Value.Raw ba32) in
      Bigarray.Array1.set ba32 2 7.5;
      C.float ~tol:0.0 ~expect:7.5 (Value.get v 2);
      Value.set v 3 1.25;
      C.float ~tol:0.0 ~expect:1.25 (Bigarray.Array1.get ba32 3));
  C.test "pinned in, pinned out, through a device buffer" (fun () ->
      let n = 4096 in
      let src = Runtime.Pinned.alloc Dtype.F32 (vec n) and dst = Runtime.Pinned.alloc Dtype.F32 (vec n) in
      for i = 0 to n - 1 do Value.set src i (float_of_int (i * 3)) done;
      let b = Runtime.Buffer.alloc ~bytes:(Value.byte_size src) in
      Runtime.Buffer.upload (Value.P src) b;
      Runtime.Buffer.download b (Value.P dst);
      Runtime.Buffer.free b;
      C.floats ~tol:0.0 ~expect:(Value.to_list src) (Value.to_list dst));
  C.test "pinned inputs through run and run_async" (fun () ->
      let n = 4096 in
      let g = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0 in
      let x = Runtime.Pinned.alloc Dtype.F32 (vec n) and y = Runtime.Pinned.alloc Dtype.F32 (vec n) in
      for i = 0 to n - 1 do Value.set x i (float_of_int i); Value.set y i 1.0 done;
      let inputs = [ ("x", Value.P x); ("y", Value.P y) ] in
      let c = Backend_cuda.compile_with ~streams:2 g in
      let sync = scalar (Backend_cuda.run c ~inputs) "s" in
      let expect = List.fold_left ( +. ) 0.0 (List.init n (fun i -> (2.0 *. float_of_int i) +. 1.0)) in
      C.float ~tol:(expect *. 1e-4) ~expect sync;
      let jobs = List.init 4 (fun _ -> Backend_cuda.run_async c ~inputs) in
      List.iter (fun j -> C.float ~tol:0.0 ~expect:sync (scalar (Backend_cuda.wait j) "s")) jobs;
      Backend_cuda.release c);
  C.test "informational: H->D bandwidth pinned vs pageable (64 MiB)" (fun () ->
      let n = 16 * 1024 * 1024 in
      let pinned = Runtime.Pinned.alloc Dtype.F32 (vec n) and pageable = Value.create Dtype.F32 (vec n) in
      let b = Runtime.Buffer.alloc ~bytes:(n * 4) in
      Runtime.Buffer.upload (Value.P pinned) b;
      let _, tp = time (fun () -> for _ = 1 to 5 do Runtime.Buffer.upload (Value.P pinned) b done) in
      let _, tg = time (fun () -> for _ = 1 to 5 do Runtime.Buffer.upload (Value.P pageable) b done) in
      Runtime.Buffer.free b;
      let gbs t = 5.0 *. float_of_int (n * 4) /. t /. 1e9 in
      Printf.printf "    pinned %.1f GB/s   pageable %.1f GB/s\n%!" (gbs tp) (gbs tg));
  C.test "finalisers free pinned memory" (fun () ->
      for _ = 1 to 50 do ignore (Runtime.Pinned.alloc Dtype.F32 (vec (256 * 1024))) done;
      Gc.full_major ();
      ignore (Runtime.Pinned.alloc Dtype.F32 (vec (256 * 1024))));
  C.run ()
