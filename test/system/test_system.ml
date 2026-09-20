(* SYSTEM TESTS, v2. Gated twice:
     1. `make system` only runs after `make unit` is green.
     2. This binary exits 0 with a SKIP message unless CUDACAML_SYSTEM=1
        is set AND a CUDA device is present.
   S1–S7 are the v1 suite (S5 now also watches the buffer pool). S8–S20
   cover RNG, AD, rows, matmul, scatter, streams, residents, LSM, pinned
   memory, multi-GPU and the timing table. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check
module Programs = Cudacaml_examples.Programs
module Bs = Cudacaml_examples.Black_scholes
module Lsm = Cudacaml_examples.Lsm
module Multi = Backend_cuda.Multi

let gated = Sys.getenv_opt "CUDACAML_SYSTEM" = Some "1"

let diff ?tolerance g inputs =
  match Differential.check ?tolerance ~reference:(module Backend_interp) ~candidate:(module Backend_cuda) g ~inputs with
  | Ok () -> ()
  | Error m -> C.fail "%s" m

let vec n = Shape.of_dims [ n ]
let f32s n f = Value.P (Value.of_list Dtype.F32 (vec n) (List.init n f))
let f32d dims f = let n = Shape.numel (Shape.of_dims dims) in Value.P (Value.of_list Dtype.F32 (Shape.of_dims dims) (List.init n f))
let f64d dims f = let n = Shape.numel (Shape.of_dims dims) in Value.P (Value.of_list Dtype.F64 (Shape.of_dims dims) (List.init n f))
let i32s n f = Value.P (Value.of_list Dtype.I32 (vec n) (List.init n f))
let seed v = ("seed", Value.P (Value.of_list Dtype.I32 Shape.scalar [ v ]))
let zero = const Dtype.F32 0.0
let one name t = [ (name, Tensor.P t) ]

let floats (Value.P v) : float list =
  match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> C.fail "dtype"

let scalar o name = List.hd (floats (List.assoc name o))
let live () = Runtime.Buffer.live_count ()

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

(* mean and variance without building a list of 2^20 floats *)
let moments (Value.P v) =
  let n = Value.numel v in
  let get i : float = match Value.dtype v with Dtype.F32 -> Value.get v i | Dtype.F64 -> Value.get v i | _ -> C.fail "dtype" in
  let s = ref 0.0 and s2 = ref 0.0 in
  for i = 0 to n - 1 do
    let x = get i in
    s := !s +. x;
    s2 := !s2 +. (x *. x)
  done;
  let m = !s /. float_of_int n in
  (m, (!s2 /. float_of_int n) -. (m *. m))

let () =
  if not gated then (C.skip "CUDACAML_SYSTEM is not 1"; C.run ());
  if not (Runtime.Device.available ()) then (C.skip "no CUDA device"; C.run ());
  print_string (Runtime.Device.info_to_string (Runtime.Device.info ()));
  print_newline ();

  (* ---------------------------------------------------------------- v1 *)
  List.iter
    (fun (e : Programs.t) -> C.test ("S1 example " ^ e.name) (fun () -> diff (e.graph ()) (e.inputs ())))
    Programs.all;
  List.iter
    (fun n ->
      C.test (Printf.sprintf "S2 saxpy n=%d" n) (fun () ->
          diff (Cudacaml_examples.Saxpy.program ~n ~a:3.0)
            [ ("x", f32s n float_of_int); ("y", f32s n (fun i -> float_of_int (n - i))) ]))
    [ 0; 1; 255; 256; 257; 1009; 300_000 ];
  C.test "S3 sum of 4M elements (two-kernel reduce)" (fun () ->
      let n = 1 lsl 22 in
      C.int ~expect:2 (List.length (Lower.program ((Programs.sum n).graph ())).kernels);
      diff ~tolerance:1e-3 ((Programs.sum n).graph ()) [ ("x", f32s n (fun i -> float_of_int (i mod 7) -. 3.0)) ]);
  C.test "S4 chain is one kernel end to end" (fun () ->
      let e = Option.get (Programs.find "chain") in
      C.int ~expect:1 (List.length (Lower.program (e.graph ())).kernels);
      diff (e.graph ()) (e.inputs ()));
  C.test "S5 run the same compiled saxpy 50 times; pool is stable" (fun () ->
      let n = 4096 in
      let c = Backend_cuda.compile (Cudacaml_examples.Saxpy.program ~n ~a:2.0) in
      let l = live () in
      for k = 1 to 50 do
        let inputs = [ ("x", f32s n (fun i -> float_of_int (i + k))); ("y", f32s n (fun _ -> 1.0)) ] in
        let expect = List.fold_left ( +. ) 0.0 (List.init n (fun i -> (2.0 *. float_of_int (i + k)) +. 1.0)) in
        C.float ~tol:(expect *. 1e-4) ~expect (scalar (Backend_cuda.run c ~inputs) "s");
        C.int ~expect:l (live ())
      done;
      Backend_cuda.release c);
  C.test "S6 i32 arithmetic" (fun () ->
      let x = param "x" Dtype.I32 (vec 9) in
      let r = map (fun e -> div (sub e (const Dtype.I32 4l)) (const Dtype.I32 2l)) x in
      diff (Graph.create ~name:"i32" ~outputs:(one "r" r)) [ ("x", i32s 9 Int32.of_int) ]);
  C.test "S7 timing saxpy n=2^24 (informational)" (fun () ->
      let n = 1 lsl 24 in
      let g = Cudacaml_examples.Saxpy.program ~n ~a:2.0 in
      let inputs = [ ("x", f32s n float_of_int); ("y", f32s n (fun _ -> 1.0)) ] in
      let ci = Backend_interp.compile g and cc = Backend_cuda.compile g in
      let _, t_interp = time (fun () -> Backend_interp.run ci ~inputs) in
      let _, t_first = time (fun () -> Backend_cuda.run cc ~inputs) in
      let _, t_warm = time (fun () -> Backend_cuda.run cc ~inputs) in
      Printf.printf "    interp %.3fs   cuda first %.3fs   cuda warm %.3fs   ratio %.1fx\n%!" t_interp t_first t_warm
        (t_interp /. t_warm);
      Backend_cuda.release cc);

  (* ---------------------------------------------------------------- v2 *)
  C.test "S8 Rng.u32 over 2^20 is bit-exact" (fun () ->
      let g = Graph.create ~name:"u32" ~outputs:(one "r" (Rng.u32 ~key:(Dsl.scalar "seed" Dtype.I32) (vec (1 lsl 20)))) in
      diff g [ seed 12345l ]);
  C.test "S9 Rng.normal F32 on device: moments" (fun () ->
      let g = Graph.create ~name:"n" ~outputs:(one "z" (Rng.normal Dtype.F32 ~key:(Dsl.scalar "seed" Dtype.I32) (vec (1 lsl 20)))) in
      let c = Backend_cuda.compile g in
      let m, v = moments (List.assoc "z" (Backend_cuda.run c ~inputs:[ seed 99l ])) in
      Printf.printf "    normal mean %.4f var %.4f\n%!" m v;
      C.float ~tol:0.01 ~expect:0.0 m;
      C.float ~tol:0.02 ~expect:1.0 v;
      Backend_cuda.release c);
  C.test "S10 reduce_rows and scan_rows over [1024; 4096]" (fun () ->
      let dims = [ 1024; 4096 ] in
      let x = param "x" Dtype.F32 (Shape.of_dims dims) in
      let g =
        Graph.create ~name:"rows"
          ~outputs:[ ("r", Tensor.P (reduce_rows add ~init:zero x)); ("p", Tensor.P (scan_rows add ~init:zero x)) ]
      in
      diff ~tolerance:1e-3 g [ ("x", f32d dims (fun i -> float_of_int (i mod 11) -. 5.0)) ]);
  C.test "S11 matmul 256x192 . 192x128, F32 and F64" (fun () ->
      let mm dtype =
        let a = param "a" dtype (Shape.of_dims [ 256; 192 ]) and b = param "b" dtype (Shape.of_dims [ 192; 128 ]) in
        Graph.create ~name:"mm" ~outputs:(one "c" (matmul a b))
      in
      let fa i = float_of_int (i mod 13) /. 7.0 -. 0.9 and fb i = float_of_int (i mod 17) /. 5.0 -. 1.6 in
      diff ~tolerance:1e-4 (mm Dtype.F32) [ ("a", f32d [ 256; 192 ] fa); ("b", f32d [ 192; 128 ] fb) ];
      diff ~tolerance:1e-9 (mm Dtype.F64) [ ("a", f64d [ 256; 192 ] fa); ("b", f64d [ 192; 128 ] fb) ]);
  C.test "S12 scatter_add histogram with collisions" (fun () ->
      let n = 100_000 in
      let idx = param "idx" Dtype.I32 (vec n) in
      let gi = Graph.create ~name:"hi" ~outputs:(one "h" (scatter_add idx (param "s" Dtype.I32 (vec n)) (vec 7))) in
      diff gi [ ("idx", i32s n (fun i -> Int32.of_int (i mod 7))); ("s", i32s n (fun _ -> 1l)) ];
      let idx = param "idx" Dtype.I32 (vec n) in
      let gf = Graph.create ~name:"hf" ~outputs:(one "h" (scatter_add idx (param "s" Dtype.F32 (vec n)) (vec 7))) in
      diff ~tolerance:1e-4 gf [ ("idx", i32s n (fun i -> Int32.of_int (i mod 7))); ("s", f32s n (fun i -> float_of_int (i mod 13) /. 4.0)) ]);
  C.test "S13 bs_greeks F64 n=65536, interp vs cuda" (fun () ->
      diff ~tolerance:1e-6 (Bs.call_greeks ~n_paths:65536 ~dtype:Dtype.F64 Bs.market) (Bs.inputs ~dtype:Dtype.F64 Bs.market ~seed:7l));
  C.test "S14 bs_greeks F32 n=2^20 on device vs closed form" (fun () ->
      let cf = Bs.analytic Bs.market in
      let c = Backend_cuda.compile (Bs.call_greeks ~n_paths:(1 lsl 20) ~dtype:Dtype.F32 Bs.market) in
      let o = Backend_cuda.run c ~inputs:(Bs.inputs ~dtype:Dtype.F32 Bs.market ~seed:3l) in
      let g w = Bs.scalar_out o (Grad.grad_name ~output:"price" ~wrt:w) in
      Printf.printf "    price %.4f (%.4f)  delta %.4f (%.4f)  vega %.3f (%.3f)  rho %.3f (%.3f)\n%!"
        (Bs.scalar_out o "price") cf.price (g "s0") cf.delta (g "vol") cf.vega (g "rate") cf.rho;
      C.float ~tol:0.1 ~expect:cf.price (Bs.scalar_out o "price");
      C.float ~tol:0.01 ~expect:cf.delta (g "s0");
      C.float ~tol:0.6 ~expect:cf.vega (g "vol");
      C.float ~tol:0.6 ~expect:cf.rho (g "rate");
      Backend_cuda.release c);
  C.test "S15 16 async jobs on 4 streams equal the sync results" (fun () ->
      let n = 1 lsl 16 in
      let g = Cudacaml_examples.Saxpy.program ~n ~a:2.0 in
      let c = Backend_cuda.compile_with ~streams:4 g in
      let inputs k = [ ("x", f32s n (fun i -> float_of_int ((i + k) mod 1000))); ("y", f32s n (fun _ -> 1.0)) ] in
      let jobs = List.init 16 (fun k -> (k, Backend_cuda.run_async c ~inputs:(inputs k))) in
      List.iter
        (fun (k, j) ->
          let got = Backend_cuda.wait j and want = Backend_cuda.run c ~inputs:(inputs k) in
          C.float ~tol:0.0 ~expect:(scalar want "s") (scalar got "s"))
        jobs;
      Backend_cuda.release c);
  C.test "S16 resident chain -> sum equals the interpreter" (fun () ->
      let n = 4096 in
      let chain = Option.get (Programs.find "chain") in
      let gc = chain.graph () and gs = (Programs.sum n).graph () in
      let cc = Backend_cuda.compile gc and cs = Backend_cuda.compile gs in
      let rx = Backend_cuda.upload (List.assoc "x" (chain.inputs ())) in
      let r = List.assoc "r" (Backend_cuda.run_resident cc ~inputs:[ ("x", rx) ]) in
      let s = List.assoc "s" (Backend_cuda.run_resident cs ~inputs:[ ("x", r) ]) in
      let got = List.hd (floats (Backend_cuda.download s)) in
      let interp g i = Backend_interp.run (Backend_interp.compile g) ~inputs:i in
      let want = scalar (interp gs [ ("x", List.assoc "r" (interp gc (chain.inputs ()))) ]) "s" in
      C.float ~tol:(1e-5 *. (1.0 +. Float.abs want)) ~expect:want got;
      List.iter Backend_cuda.free_resident [ rx; r; s ];
      Backend_cuda.release cc;
      Backend_cuda.release cs);
  C.test "S17 LSM American put on device vs binomial" (fun () ->
      let binom = Lsm.binomial_american_put ~steps:500 Bs.market in
      let t = Lsm.Make_cuda_resident.create ~n_paths:65536 ~n_steps:50 Bs.market in
      let px, secs = time (fun () -> Lsm.Make_cuda_resident.price t ~seed:11l) in
      Printf.printf "    lsm %.4f  binomial %.4f  (%.3fs)\n%!" px binom secs;
      C.float ~tol:0.15 ~expect:binom px);
  C.test "S18 pinned round trip and bandwidth" (fun () ->
      let n = 16 * 1024 * 1024 in
      let src = Runtime.Pinned.alloc Dtype.F32 (vec n) and dst = Runtime.Pinned.alloc Dtype.F32 (vec n) in
      for i = 0 to n - 1 do Value.set src i (float_of_int (i land 1023)) done;
      let b = Runtime.Buffer.alloc ~bytes:(n * 4) in
      let _, tp = time (fun () -> Runtime.Buffer.upload (Value.P src) b) in
      Runtime.Buffer.download b (Value.P dst);
      let pageable = Value.create Dtype.F32 (vec n) in
      let _, tg = time (fun () -> Runtime.Buffer.upload (Value.P pageable) b) in
      Runtime.Buffer.free b;
      Printf.printf "    H->D pinned %.1f GB/s   pageable %.1f GB/s\n%!" (float_of_int (n * 4) /. tp /. 1e9) (float_of_int (n * 4) /. tg /. 1e9);
      for i = 0 to n - 1 do
        if Value.get src i <> Value.get dst i then C.fail "mismatch at %d" i
      done);
  C.test "S19 Multi [0;0] equals single device; [0;1] when available" (fun () ->
      let n = 1 lsl 16 in
      let saxpy ~n = Cudacaml_examples.Saxpy.program ~n ~a:2.0 in
      let inputs = [ ("x", f32s n float_of_int); ("y", f32s n (fun _ -> 1.0)) ] in
      let c = Backend_cuda.compile (saxpy ~n) in
      let single = Backend_cuda.run c ~inputs in
      Backend_cuda.release c;
      let check devices =
        let t = Multi.create ~devices ~n saxpy in
        let per = Multi.run t ~inputs in
        C.floats ~tol:0.0 ~expect:(floats (List.assoc "r" single)) (floats (Multi.concat (List.map (fun o -> List.assoc "r" o) per)));
        Multi.release t
      in
      check [ 0; 0 ];
      if Runtime.Device.count () >= 2 then check [ 0; 1 ] else Printf.printf "    one device: [0;1] not run\n%!");
  C.test "S20 timing table (informational)" (fun () ->
      let n = 1 lsl 22 in
      let warm c inputs =
        ignore (Backend_cuda.run c ~inputs);
        let _, t = time (fun () -> for _ = 1 to 5 do ignore (Backend_cuda.run c ~inputs) done) in
        t /. 5.0
      in
      let row dtype label =
        let cp = Backend_cuda.compile (Bs.call_mc ~n_paths:n ~dtype Bs.market) in
        let cg = Backend_cuda.compile (Bs.call_greeks ~n_paths:n ~dtype Bs.market) in
        let inputs = Bs.inputs ~dtype Bs.market ~seed:1l in
        let tp = warm cp inputs and tg = warm cg inputs in
        Printf.printf "    %s  price %.1f ms   greeks %.1f ms   ratio %.2fx\n%!" label (tp *. 1e3) (tg *. 1e3) (tg /. tp);
        Backend_cuda.release cp;
        Backend_cuda.release cg;
        tp
      in
      let t32 = row Dtype.F32 "f32" in
      let t64 = row Dtype.F64 "f64" in
      Printf.printf "    f64/f32 price time ratio %.2fx\n%!" (t64 /. t32));

  C.run ()
