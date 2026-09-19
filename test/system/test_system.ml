(* SYSTEM TESTS. Gated twice:
     1. `make system` only runs after `make unit` is green.
     2. This binary exits 0 with a SKIP message unless OCAML_CUDA_SYSTEM=1
        is set AND a CUDA device is present.
   Everything here is end to end: Dsl -> Graph -> Lower -> Emit -> NVRTC ->
   launch -> download, compared against the interpreter. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module Programs = Ocaml_cuda_examples.Programs

let gated = Sys.getenv_opt "OCAML_CUDA_SYSTEM" = Some "1"

let diff ?tolerance g inputs =
  match Differential.check ?tolerance ~reference:(module Backend_interp) ~candidate:(module Backend_cuda) g ~inputs with
  | Ok () -> ()
  | Error m -> C.fail "%s" m

let vec n = Shape.of_dims [ n ]
let f32s n f = Value.P (Value.of_list Dtype.F32 (vec n) (List.init n f))

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

let () =
  if not gated then (C.skip "OCAML_CUDA_SYSTEM is not 1"; C.run ());
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  Printf.printf "device: %s\n%!" (Runtime.Device.name ());

  (* S1: every registered example, interp vs cuda. *)
  List.iter
    (fun (e : Programs.t) -> C.test ("S1 example " ^ e.name) (fun () -> diff (e.graph ()) (e.inputs ())))
    Programs.all;

  (* S2: sizes that stress launch geometry: 0, 1, block-1, block, block+1,
     a prime, and something larger than grid*block so the stride loop
     actually strides. *)
  List.iter
    (fun n ->
      C.test (Printf.sprintf "S2 saxpy n=%d" n) (fun () ->
          diff (Ocaml_cuda_examples.Saxpy.program ~n ~a:3.0)
            [ ("x", f32s n float_of_int); ("y", f32s n (fun i -> float_of_int (n - i))) ]))
    [ 0; 1; 255; 256; 257; 1009; 300_000 ];

  (* S3: reduction over a large input; float sums reorder, so a relative
     tolerance is legitimate here and must NOT be widened beyond 1e-3. *)
  C.test "S3 sum of 4M elements" (fun () ->
      let n = 1 lsl 22 in
      diff ~tolerance:1e-3 ((Programs.sum n).graph ()) [ ("x", f32s n (fun i -> float_of_int (i mod 7) -. 3.0)) ]);

  (* S4: fusion is real: chain of five maps emits ONE kernel and one launch. *)
  C.test "S4 chain is one kernel end to end" (fun () ->
      let e = Option.get (Programs.find "chain") in
      let prog = Lower.program (e.graph ()) in
      C.int ~expect:1 (List.length prog.kernels);
      diff (e.graph ()) (e.inputs ()));

  (* S5: a compiled program is reusable and has no per-run state leak. *)
  C.test "S5 run the same compiled saxpy 50 times" (fun () ->
      let n = 4096 in
      let c = Backend_cuda.compile (Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0) in
      for k = 1 to 50 do
        let inputs = [ ("x", f32s n (fun i -> float_of_int (i + k))); ("y", f32s n (fun _ -> 1.0)) ] in
        let o = Backend_cuda.run c ~inputs in
        match List.assoc "s" o with
        | Value.P v -> (
            match Value.dtype v with
            | Dtype.F32 ->
                let expect = List.fold_left ( +. ) 0.0 (List.init n (fun i -> (2.0 *. float_of_int (i + k)) +. 1.0)) in
                C.float ~tol:(expect *. 1e-4) ~expect (Value.get v 0)
            | _ -> C.fail "dtype")
      done);

  (* S6: i32 programs, including negative division and the reverse gather. *)
  C.test "S6 i32 arithmetic" (fun () ->
      let x = param "x" Dtype.I32 (vec 9) in
      let r = map (fun e -> div (sub e (const Dtype.I32 4l)) (const Dtype.I32 2l)) x in
      let g = Graph.create ~name:"i32" ~outputs:[ ("r", Tensor.P r) ] in
      diff g [ ("x", Value.P (Value.of_list Dtype.I32 (vec 9) (List.init 9 Int32.of_int))) ]);

  (* S7: the demo number. Not an assertion on speed (CI boxes vary); it
     asserts only that the GPU path completes and prints the ratio. *)
  C.test "S7 timing saxpy n=2^24 (informational)" (fun () ->
      let n = 1 lsl 24 in
      let g = Ocaml_cuda_examples.Saxpy.program ~n ~a:2.0 in
      let inputs = [ ("x", f32s n float_of_int); ("y", f32s n (fun _ -> 1.0)) ] in
      let ci = Backend_interp.compile g and cc = Backend_cuda.compile g in
      let _, t_interp = time (fun () -> Backend_interp.run ci ~inputs) in
      let _, t_cuda_first = time (fun () -> Backend_cuda.run cc ~inputs) in
      let _, t_cuda = time (fun () -> Backend_cuda.run cc ~inputs) in
      Printf.printf "    interp %.3fs   cuda first %.3fs   cuda warm %.3fs   ratio %.1fx\n%!" t_interp t_cuda_first t_cuda (t_interp /. t_cuda));

  C.run ()
