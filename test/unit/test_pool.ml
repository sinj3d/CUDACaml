(* T24: persistent executor. GPU-dependent; skips without one. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module Programs = Ocaml_cuda_examples.Programs

let vec n = Shape.of_dims [ n ]
let f32s n f = Value.P (Value.of_list Dtype.F32 (vec n) (List.init n f))
let live () = Runtime.Buffer.live_count ()

(* A locally abstract type is what lets the existential under [Value.P]
   be matched against [Dtype.F32]; the checks below are unchanged. *)
let get_f32 : type a. a Dtype.t -> a Value.t -> float =
 fun d v -> match d with Dtype.F32 -> Value.get v 0 | _ -> C.fail "dtype"

let set_f32 : type a. a Dtype.t -> a Value.t -> float -> unit =
 fun d v x -> match d with Dtype.F32 -> Value.set v 0 x | _ -> C.fail "dtype"

let scalar o name = match List.assoc name o with Value.P v -> get_f32 (Value.dtype v) v
let first_elem o name = match List.assoc name o with Value.P v -> get_f32 (Value.dtype v) v

let () =
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  let n = 4096 in
  let saxpy = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0 in
  let inputs k = [ ("x", f32s n (fun i -> float_of_int (i + k))); ("y", f32s n (fun _ -> 1.0)) ] in
  let expect k = List.fold_left ( +. ) 0.0 (List.init n (fun i -> (2.0 *. float_of_int (i + k)) +. 1.0)) in
  C.test "buffers are allocated at compile and stable across 50 runs" (fun () ->
      let before = live () in
      let c = Backend_cuda.compile saxpy in
      let held = Backend_cuda.Executor.buffer_count (Backend_cuda.executor c) in
      C.bool ~expect:true (held >= 4);
      C.int ~expect:(before + held) (live ());
      for k = 1 to 50 do
        let s = scalar (Backend_cuda.run c ~inputs:(inputs k)) "s" in
        let e = expect k in
        C.float ~tol:(e *. 1e-4) ~expect:e s;
        C.int ~expect:(before + held) (live ())
      done;
      Backend_cuda.release c;
      C.int ~expect:before (live ()));
  C.test "release is idempotent; run after release raises" (fun () ->
      let before = live () in
      let c = Backend_cuda.compile saxpy in
      Backend_cuda.release c;
      Backend_cuda.release c;
      C.int ~expect:before (live ());
      C.raises (fun () -> ignore (Backend_cuda.run c ~inputs:(inputs 1))));
  C.test "two compiled programs interleave" (fun () ->
      let chain = Option.get (Programs.find "chain") in
      let c1 = Backend_cuda.compile saxpy and c2 = Backend_cuda.compile (chain.graph ()) in
      let expect_chain = Backend_interp.run (Backend_interp.compile (chain.graph ())) ~inputs:(chain.inputs ()) in
      for k = 1 to 5 do
        let e = expect k in
        C.float ~tol:(e *. 1e-4) ~expect:e (scalar (Backend_cuda.run c1 ~inputs:(inputs k)) "s");
        C.float ~tol:1e-4 ~expect:(first_elem expect_chain "r") (first_elem (Backend_cuda.run c2 ~inputs:(chain.inputs ())) "r")
      done;
      Backend_cuda.release c1;
      Backend_cuda.release c2);
  C.test "outputs of successive runs are distinct values" (fun () ->
      let c = Backend_cuda.compile saxpy in
      let o1 = Backend_cuda.run c ~inputs:(inputs 1) in
      let o2 = Backend_cuda.run c ~inputs:(inputs 1) in
      (match List.assoc "r" o1 with Value.P v -> set_f32 (Value.dtype v) v (-999.0));
      C.float ~tol:0.0 ~expect:3.0 (first_elem o2 "r");
      Backend_cuda.release c);
  C.test "missing input raises and leaks nothing" (fun () ->
      let c = Backend_cuda.compile saxpy in
      let l = live () in
      C.raises (fun () -> ignore (Backend_cuda.run c ~inputs:[ ("x", f32s n float_of_int) ]));
      C.int ~expect:l (live ());
      Backend_cuda.release c);
  C.run ()
