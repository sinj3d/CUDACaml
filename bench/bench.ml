(* Speedup benchmark: the same two workloads written twice.

     vanilla   plain OCaml over float arrays, no DSL, no FFI
     cuda      Dsl -> Graph -> Lower -> Emit -> NVRTC -> launch

   Two workloads, because they answer different questions:

     saxpy  r = a*x + y, s = sum r        3 flops per element
     poly   r = horner(x, k coeffs)       2k-1 flops per element

   saxpy is memory bound: the GPU time is almost entirely the host<->device
   copy of x, y and r, so it measures the bus, not the card. poly keeps the
   same traffic and scales the arithmetic, which is where a GPU actually wins.

   Reported GPU time is a whole [Backend_cuda.run]: upload, launch, download.
   Compilation is timed separately and excluded, as it is amortised over runs.

   usage: ocaml-cuda-bench [n] [reps] [degree]  (defaults 1<<24, 5, 64) *)

open Ocaml_cuda

let time f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

(* Best of [reps], not the mean: the minimum is the least noisy estimate of
   the cost when nothing else is competing for the machine. *)
let best reps f =
  let rec go i acc_r acc_t =
    if i = 0 then (acc_r, acc_t)
    else
      let r, t = time f in
      go (i - 1) (Some r) (Float.min acc_t t)
  in
  go reps None infinity

(* ---------------------------------------------------------------- vanilla *)

let vanilla_saxpy ~a ~x ~y ~r =
  let n = Array.length x in
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    let v = (a *. Array.unsafe_get x i) +. Array.unsafe_get y i in
    Array.unsafe_set r i v;
    s := !s +. v
  done;
  !s

let vanilla_poly ~c ~x ~r =
  let n = Array.length x and k = Array.length c in
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    let xi = Array.unsafe_get x i in
    let acc = ref (Array.unsafe_get c 0) in
    for j = 1 to k - 1 do
      acc := (!acc *. xi) +. Array.unsafe_get c j
    done;
    Array.unsafe_set r i !acc;
    s := !s +. !acc
  done;
  !s

(* ------------------------------------------------------------------ graphs *)

let vec n = Shape.of_dims [ n ]

(* Same recurrence as [vanilla_poly], built as k-1 chained maps. Fusion
   collapses the whole chain into one kernel however long it is, so the
   printed kernel count stays at 2 (the fused map, then the reduce) while
   the arithmetic per element grows with the degree. *)
let poly_graph ~n ~c =
  let open Dsl in
  let x = param "x" Dtype.F32 (vec n) in
  let step acc ci = map2 (fun a xi -> add (mul a xi) (const Dtype.F32 ci)) acc x in
  let acc =
    match Array.to_list c with
    | [] -> invalid_arg "poly_graph: no coefficients"
    | c0 :: rest -> List.fold_left step (map (fun _ -> const Dtype.F32 c0) x) rest
  in
  Graph.create ~name:"poly"
    ~outputs:[ ("r", Tensor.P acc); ("s", Tensor.P (reduce add ~init:(const Dtype.F32 0.0) acc)) ]

let saxpy_graph ~n ~a = Ocaml_cuda_examples.Saxpy.program ~n ~a

(* --------------------------------------------------------------- plumbing *)

(* float array -> Value, without going through a list: at n = 2^24 the
   intermediate list alone is 400 MB and dominates the measurement. *)
let value_of_array (a : float array) =
  let v = Value.create Dtype.F32 (vec (Array.length a)) in
  Array.iteri (fun i x -> Value.set v i x) a;
  Value.P v

let scalar_out name outs =
  let v : float =
    match List.assoc name outs with
    | Value.P v -> ( match Value.dtype v with Dtype.F32 -> Value.get v 0 | _ -> nan)
  in
  v

let agree ~label ~vanilla ~cuda =
  let tol = Float.max 1e-3 (Float.abs vanilla *. 1e-3) in
  if Float.abs (vanilla -. cuda) > tol then (
    (* f32 sums reorder on the GPU, so only a relative check is meaningful;
       a real disagreement is orders of magnitude bigger than 1e-3. *)
    Printf.eprintf "%s: MISMATCH vanilla %.6g vs cuda %.6g\n%!" label vanilla cuda;
    exit 1)

let kernel_count g = List.length (Lower.program g).kernels

let report label n flops_per_elem t_van t_cuda t_compile kernels =
  let gflops t = float_of_int n *. float_of_int flops_per_elem /. t /. 1e9 in
  Printf.printf "%-6s n=%-9d %2d kernel(s)  jit %6.3fs\n" label n kernels t_compile;
  Printf.printf "         vanilla %8.3f s   %7.2f GFLOP/s\n" t_van (gflops t_van);
  Printf.printf "         cuda    %8.3f s   %7.2f GFLOP/s   speedup %.1fx\n\n%!" t_cuda
    (gflops t_cuda) (t_van /. t_cuda)

(* ------------------------------------------------------------------- main *)

let () =
  let arg i d = try int_of_string Sys.argv.(i) with _ -> d in
  let n = arg 1 (1 lsl 24) and reps = arg 2 5 and degree = arg 3 64 in

  if not (Runtime.Device.available ()) then (
    prerr_endline "no CUDA device: nothing to compare against";
    exit 77);
  Printf.printf "device: %s\nn = %d, reps = %d (best of), poly degree = %d\n\n%!"
    (Runtime.Device.name ()) n reps degree;

  let x = Array.init n (fun i -> float_of_int (i mod 17) -. 8.0) in
  let y = Array.init n (fun i -> float_of_int (n - i) /. 4.0) in
  let r = Array.make n 0.0 in
  let vx = value_of_array x and vy = value_of_array y in

  (* --- saxpy --- *)
  let a = 2.0 in
  let g = saxpy_graph ~n ~a in
  let c, t_compile = time (fun () -> Backend_cuda.compile g) in
  let inputs = [ ("x", vx); ("y", vy) ] in
  ignore (Backend_cuda.run c ~inputs) (* warm up: context, module, allocator *);
  let s_van, t_van = best reps (fun () -> vanilla_saxpy ~a ~x ~y ~r) in
  let outs, t_cuda = best reps (fun () -> Backend_cuda.run c ~inputs) in
  agree ~label:"saxpy" ~vanilla:(Option.get s_van) ~cuda:(scalar_out "s" (Option.get outs));
  report "saxpy" n 3 t_van t_cuda t_compile (kernel_count g);

  (* --- poly --- *)
  let c_coef = Array.init degree (fun j -> 1.0 /. float_of_int (j + 1)) in
  let g = poly_graph ~n ~c:c_coef in
  let cc, t_compile = time (fun () -> Backend_cuda.compile g) in
  let inputs = [ ("x", vx) ] in
  ignore (Backend_cuda.run cc ~inputs);
  let s_van, t_van = best reps (fun () -> vanilla_poly ~c:c_coef ~x ~r) in
  let outs, t_cuda = best reps (fun () -> Backend_cuda.run cc ~inputs) in
  agree ~label:"poly" ~vanilla:(Option.get s_van) ~cuda:(scalar_out "s" (Option.get outs));
  report "poly" n ((2 * degree) - 1) t_van t_cuda t_compile (kernel_count g)
