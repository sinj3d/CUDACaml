(* T25: streams, events, async runs, resident values. GPU-dependent. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module Programs = Ocaml_cuda_examples.Programs
module Lsm = Ocaml_cuda_examples.Lsm
module Bs = Ocaml_cuda_examples.Black_scholes

let vec n = Shape.of_dims [ n ]
let f32s n f = Value.P (Value.of_list Dtype.F32 (vec n) (List.init n f))
let live () = Runtime.Buffer.live_count ()

let floats (Value.P v) : float list =
  match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> C.fail "dtype"

let scalar o name = List.hd (floats (List.assoc name o))
let interp g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

let () =
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  let n = 4096 in
  let saxpy = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0 in
  let inputs k = [ ("x", f32s n (fun i -> float_of_int (i + k))); ("y", f32s n (fun _ -> 1.0)) ] in
  C.test "streams and events" (fun () ->
      let s = Runtime.Stream.create () in
      let e0 = Runtime.Event.record Runtime.Stream.default in
      let e1 = Runtime.Event.record s in
      Runtime.Event.synchronize e1;
      Runtime.Stream.synchronize s;
      C.bool ~expect:true (Runtime.Event.elapsed_ms ~start:e0 ~stop:e1 >= 0.0));
  C.test "8 async jobs on 3 streams equal the synchronous results" (fun () ->
      let c = Backend_cuda.compile_with ~streams:3 saxpy in
      let jobs = List.init 8 (fun k -> (k, Backend_cuda.run_async c ~inputs:(inputs k))) in
      List.iter
        (fun (k, j) ->
          let got = Backend_cuda.wait j in
          let want = Backend_cuda.run c ~inputs:(inputs k) in
          C.floats ~tol:0.0 ~expect:(floats (List.assoc "r" want)) (floats (List.assoc "r" got));
          C.float ~tol:0.0 ~expect:(scalar want "s") (scalar got "s"))
        jobs;
      let j = Backend_cuda.run_async c ~inputs:(inputs 99) in
      let a = Backend_cuda.wait j and b = Backend_cuda.wait j in
      C.float ~tol:0.0 ~expect:(scalar a "s") (scalar b "s");
      Backend_cuda.release c);
  C.test "resident upload / download round trip" (fun () ->
      let v = f32s 1000 (fun i -> float_of_int i *. 0.25) in
      let r = Backend_cuda.upload v in
      C.bool ~expect:true (Shape.equal r.shape (vec 1000));
      C.floats ~tol:0.0 ~expect:(floats v) (floats (Backend_cuda.download r));
      Backend_cuda.free_resident r);
  C.test "run_resident saxpy equals the host run" (fun () ->
      let c = Backend_cuda.compile saxpy in
      let ins = inputs 3 in
      let rx = Backend_cuda.upload (List.assoc "x" ins) and ry = Backend_cuda.upload (List.assoc "y" ins) in
      let outs = Backend_cuda.run_resident c ~inputs:[ ("x", rx); ("y", ry) ] in
      let want = Backend_cuda.run c ~inputs:ins in
      let rr = List.assoc "r" outs and rs = List.assoc "s" outs in
      C.bool ~expect:true (Shape.equal rr.shape (vec n));
      C.bool ~expect:true (Shape.equal rs.shape Shape.scalar);
      C.bool ~expect:true (match rr.dtype with Dtype.P Dtype.F32 -> true | _ -> false);
      C.floats ~tol:0.0 ~expect:(floats (List.assoc "r" want)) (floats (Backend_cuda.download rr));
      C.float ~tol:0.0 ~expect:(scalar want "s") (List.hd (floats (Backend_cuda.download rs)));
      List.iter Backend_cuda.free_resident [ rx; ry; rr; rs ];
      Backend_cuda.release c);
  C.test "resident chaining: chain -> sum with no host round trip" (fun () ->
      let before = live () in
      let chain = Option.get (Programs.find "chain") in
      let gc = chain.graph () and gs = (Programs.sum n).graph () in
      let cc = Backend_cuda.compile gc and cs = Backend_cuda.compile gs in
      let x = List.assoc "x" (chain.inputs ()) in
      let rx = Backend_cuda.upload x in
      let r = List.assoc "r" (Backend_cuda.run_resident cc ~inputs:[ ("x", rx) ]) in
      let s = List.assoc "s" (Backend_cuda.run_resident cs ~inputs:[ ("x", r) ]) in
      let got = List.hd (floats (Backend_cuda.download s)) in
      let want = scalar (interp gs [ ("x", List.assoc "r" (interp gc (chain.inputs ()))) ]) "s" in
      C.float ~tol:(1e-4 *. (1.0 +. Float.abs want)) ~expect:want got;
      List.iter Backend_cuda.free_resident [ rx; r; s ];
      Backend_cuda.release cc;
      Backend_cuda.release cs;
      C.int ~expect:before (live ()));
  C.test "informational: 16 sync vs 16 async runs on 4 streams" (fun () ->
      let n = 1 lsl 22 in
      let g = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0 in
      let ins = [ ("x", f32s n float_of_int); ("y", f32s n (fun _ -> 1.0)) ] in
      let c1 = Backend_cuda.compile g and c4 = Backend_cuda.compile_with ~streams:4 g in
      ignore (Backend_cuda.run c1 ~inputs:ins);
      let _, t_sync = time (fun () -> for _ = 1 to 16 do ignore (Backend_cuda.run c1 ~inputs:ins) done) in
      let _, t_async =
        time (fun () ->
            let jobs = List.init 16 (fun _ -> Backend_cuda.run_async c4 ~inputs:ins) in
            List.iter (fun j -> ignore (Backend_cuda.wait j)) jobs)
      in
      Printf.printf "    sync %.3fs   async(4 streams) %.3fs\n%!" t_sync t_async;
      Backend_cuda.release c1;
      Backend_cuda.release c4);
  C.test "LSM resident driver agrees with the generic CUDA driver" (fun () ->
      let module G = Lsm.Make (Backend_cuda) in
      let a = G.price (G.create ~n_paths:4096 ~n_steps:20 Bs.market) ~seed:5l in
      let b = Lsm.Make_cuda_resident.price (Lsm.Make_cuda_resident.create ~n_paths:4096 ~n_steps:20 Bs.market) ~seed:5l in
      C.float ~tol:1e-9 ~expect:a b);
  C.run ()
