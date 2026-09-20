(* Speedup benchmark: the same two workloads written twice.

     vanilla   plain OCaml over float arrays, no DSL, no FFI
     cuda      Dsl -> Graph -> Lower -> Emit -> NVRTC -> launch

   Two workloads, because they answer different questions:

     saxpy  r = a*x + y, s = sum r        3 flops per element
     poly   r = horner(x, k coeffs)       2k-1 flops per element
     greeks Black-Scholes MC price, then the same graph differentiated

   The third workload has no vanilla counterpart: the question it answers
   is not "how much faster than OCaml" but "what does reverse mode cost",
   so it reports the ratio of the gradient graph's time to the forward
   graph's time rather than a speedup.

   saxpy is memory bound: the GPU time is almost entirely the host<->device
   copy of x, y and r, so it measures the bus, not the card. poly keeps the
   same traffic and scales the arithmetic, which is where a GPU actually wins.

   Reported GPU time is a whole [Backend_cuda.run]: upload, launch, download.
   Compilation is timed separately and excluded, as it is amortised over runs.

   The dtype argument selects the precision of the *device* graphs. The
   vanilla loops are unchanged by it: an OCaml float is a double whatever
   the card is asked to do, which is exactly the point of the f64 column --
   vanilla poly costs the same in both rows, the GPU does not.

   --pinned puts every host value that crosses the bus -- the inputs of
   all three workloads and the destination of the download -- in
   page-locked memory. A copy out of pageable memory is staged through a
   page-locked buffer the driver owns; a copy out of page-locked memory is
   a DMA. Which class is in use is printed, and reported as hostmem= on
   the RESULT line.

   usage: cudacaml-bench [n] [reps] [degree] [f32|f64] [--pinned]
          (defaults 1<<24, 5, 64, f32, pageable) *)

open Cudacaml
module Bs = Cudacaml_examples.Black_scholes

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

(* ------------------------------------------------------------------ dtype *)

(* [F32] and [F64] are both [float Dtype.t], so one graph builder covers
   both precisions with no duplication and no existential wrapper. *)
let dtype_of_string : string -> float Dtype.t option = function
  | "f32" -> Some Dtype.F32
  | "f64" -> Some Dtype.F64
  | _ -> None

(* ------------------------------------------------------------------ graphs *)

let vec n = Shape.of_dims [ n ]

(* Same recurrence as [vanilla_poly], built as k-1 chained maps. Fusion
   collapses the whole chain into one kernel however long it is, so the
   printed kernel count stays at 2 (the fused map, then the reduce) while
   the arithmetic per element grows with the degree. *)
let poly_graph ~dt ~n ~c =
  let open Dsl in
  let x = param "x" dt (vec n) in
  let step acc ci = map2 (fun a xi -> add (mul a xi) (const dt ci)) acc x in
  let acc =
    match Array.to_list c with
    | [] -> invalid_arg "poly_graph: no coefficients"
    | c0 :: rest -> List.fold_left step (map (fun _ -> const dt c0) x) rest
  in
  Graph.create ~name:"poly"
    ~outputs:[ ("r", Tensor.P acc); ("s", Tensor.P (reduce add ~init:(const dt 0.0) acc)) ]

(* The shape of [Cudacaml_examples.Saxpy.program], but at the requested
   dtype: the example is fixed at F32 and the point here is to vary it. *)
let saxpy_graph ~dt ~n ~a =
  let open Dsl in
  let x = param "x" dt (vec n) in
  let y = param "y" dt (vec n) in
  let ax = map (fun xi -> mul (const dt a) xi) x in
  let r = map2 add ax y in
  let s = reduce add ~init:(const dt 0.0) r in
  Graph.create ~name:"saxpy" ~outputs:[ ("r", Tensor.P r); ("s", Tensor.P s) ]

(* --------------------------------------------------------------- plumbing *)

(* Host storage for everything that crosses the bus. [--pinned] makes it
   page-locked, which is the whole difference the flag makes: the driver
   can DMA straight out of (or into) it instead of staging the copy
   through a page-locked buffer of its own. *)
let host_value ~pinned dt shape =
  if pinned then Runtime.Pinned.alloc dt shape else Value.create dt shape

(* float array -> Value, without going through a list: at n = 2^24 the
   intermediate list alone is 400 MB and dominates the measurement. *)
let value_of_array ~pinned dt (a : float array) =
  let v = host_value ~pinned dt (vec (Array.length a)) in
  Array.iteri (fun i x -> Value.set v i x) a;
  Value.P v

(* Page-locked copies of an input list this benchmark did not build itself. *)
let pin_inputs ~pinned inputs =
  if not pinned then inputs
  else
    List.map
      (fun (name, v) -> match v with Value.P v -> (name, Value.P (Runtime.Pinned.of_value v)))
      inputs

let scalar_out name outs =
  let v : float =
    match List.assoc name outs with
    | Value.P v -> (
        match Value.dtype v with
        | Dtype.F32 -> Value.get v 0
        | Dtype.F64 -> Value.get v 0
        | _ -> nan)
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

let gflops ~n ~flops_per_elem t = float_of_int n *. float_of_int flops_per_elem /. t /. 1e9

let report label n dtype flops_per_elem t_van t_cuda t_compile kernels =
  let g t = gflops ~n ~flops_per_elem t in
  Printf.printf "%-6s n=%-9d %s  %2d kernel(s)  jit %6.3fs\n" label n dtype kernels t_compile;
  Printf.printf "         vanilla %8.3f s   %7.2f GFLOP/s\n" t_van (g t_van);
  Printf.printf "         cuda    %8.3f s   %7.2f GFLOP/s   speedup %.1fx\n\n%!" t_cuda (g t_cuda)
    (t_van /. t_cuda)

(* ------------------------------------------------------------------- main *)

let usage () =
  Printf.eprintf "usage: cudacaml-bench [n] [reps] [degree] [f32|f64] [--pinned]\n%!"

let () =
  (* Flags may sit anywhere; the positional arguments keep their places
     once the flags have been taken out. *)
  let flags, pos =
    List.partition (fun s -> String.length s > 1 && s.[0] = '-') (Array.to_list Sys.argv)
  in
  List.iter
    (fun f ->
      if f <> "--pinned" then (
        Printf.eprintf "unknown flag %s\n" f;
        usage ();
        exit 2))
    flags;
  let pinned = List.mem "--pinned" flags in
  let mem_label = if pinned then "pinned" else "pageable" in
  let pos = Array.of_list pos in
  let arg i d = try int_of_string pos.(i) with _ -> d in
  let n = arg 1 (1 lsl 24) and reps = arg 2 5 and degree = arg 3 64 in
  let dtype_s = try pos.(4) with _ -> "f32" in
  let dt =
    match dtype_of_string dtype_s with
    | Some dt -> dt
    | None ->
        Printf.eprintf "unknown dtype %S: expected f32 or f64\n" dtype_s;
        usage ();
        exit 2
  in

  if not (Runtime.Device.available ()) then (
    prerr_endline "no CUDA device: nothing to compare against";
    exit 77);
  let card = Runtime.Device.name () in
  Printf.printf
    "device: %s\nn = %d, reps = %d (best of), poly degree = %d, dtype = %s, host memory = %s\n\n%!"
    card n reps degree dtype_s mem_label;

  let x = Array.init n (fun i -> float_of_int (i mod 17) -. 8.0) in
  let y = Array.init n (fun i -> float_of_int (n - i) /. 4.0) in
  let r = Array.make n 0.0 in
  let vx = value_of_array ~pinned dt x and vy = value_of_array ~pinned dt y in

  (* --- host <-> device traffic ---

     What [--pinned] is for, measured on its own before any kernel muddies
     it. The download destination is a host value of the same class, so
     this covers the output side as well as the input side. *)
  let bytes = n * Dtype.size_in_bytes dt in
  let sink = host_value ~pinned dt (vec n) in
  let buf = Runtime.Buffer.alloc ~bytes in
  Runtime.Buffer.upload vx buf (* warm up: context, staging buffer *);
  let _, t_h2d = best reps (fun () -> Runtime.Buffer.upload vx buf) in
  let _, t_d2h = best reps (fun () -> Runtime.Buffer.download buf (Value.P sink)) in
  Runtime.Buffer.free buf;
  let bus t = float_of_int bytes /. t /. 1e9 in
  Printf.printf "hostmem %-8s %4d MiB   H->D %6.2f GB/s   D->H %6.2f GB/s\n\n%!" mem_label
    (bytes / 1048576) (bus t_h2d) (bus t_d2h);

  (* --- saxpy --- *)
  let a = 2.0 in
  let g = saxpy_graph ~dt ~n ~a in
  let c, t_compile = time (fun () -> Backend_cuda.compile g) in
  let inputs = [ ("x", vx); ("y", vy) ] in
  ignore (Backend_cuda.run c ~inputs) (* warm up: context, module, allocator *);
  let s_van, t_van = best reps (fun () -> vanilla_saxpy ~a ~x ~y ~r) in
  let outs, t_saxpy = best reps (fun () -> Backend_cuda.run c ~inputs) in
  agree ~label:"saxpy" ~vanilla:(Option.get s_van) ~cuda:(scalar_out "s" (Option.get outs));
  report "saxpy" n dtype_s 3 t_van t_saxpy t_compile (kernel_count g);
  let saxpy_speedup = t_van /. t_saxpy in

  (* --- poly --- *)
  let c_coef = Array.init degree (fun j -> 1.0 /. float_of_int (j + 1)) in
  let poly_flops = (2 * degree) - 1 in
  let g = poly_graph ~dt ~n ~c:c_coef in
  let cc, t_compile = time (fun () -> Backend_cuda.compile g) in
  let inputs = [ ("x", vx) ] in
  ignore (Backend_cuda.run cc ~inputs);
  let s_van, t_van = best reps (fun () -> vanilla_poly ~c:c_coef ~x ~r) in
  let outs, t_poly = best reps (fun () -> Backend_cuda.run cc ~inputs) in
  agree ~label:"poly" ~vanilla:(Option.get s_van) ~cuda:(scalar_out "s" (Option.get outs));
  report "poly" n dtype_s poly_flops t_van t_poly t_compile (kernel_count g);
  let poly_speedup = t_van /. t_poly in

  (* --- greeks ---

     Always f32 and always [n] paths, whatever the dtype argument says: the
     question is what the adjoint chain costs relative to the forward pass
     on the same inputs, and both graphs are built at the same precision,
     so the ratio stands on its own. *)
  let bs_dt = Dtype.F32 in
  let g_price = Bs.call_mc ~n_paths:n ~dtype:bs_dt Bs.market in
  let g_greeks = Bs.call_greeks ~n_paths:n ~dtype:bs_dt Bs.market in
  let bs_inputs = pin_inputs ~pinned (Bs.inputs ~dtype:bs_dt Bs.market ~seed:7l) in
  let cprice = Backend_cuda.compile g_price in
  let cgreeks, t_compile = time (fun () -> Backend_cuda.compile g_greeks) in
  ignore (Backend_cuda.run cprice ~inputs:bs_inputs);
  ignore (Backend_cuda.run cgreeks ~inputs:bs_inputs);
  let _, t_fwd = best reps (fun () -> Backend_cuda.run cprice ~inputs:bs_inputs) in
  let g_outs, t_greeks = best reps (fun () -> Backend_cuda.run cgreeks ~inputs:bs_inputs) in
  let g_outs = Option.get g_outs in
  let greek w = scalar_out (Grad.grad_name ~output:"price" ~wrt:w) g_outs in
  let greeks_ratio = t_greeks /. t_fwd in
  Printf.printf "greeks n=%-9d f32  %2d kernel(s)  jit %6.3fs\n" n (kernel_count g_greeks)
    t_compile;
  Printf.printf "         price   %8.3f s\n" t_fwd;
  Printf.printf "         greeks  %8.3f s   greeks/price ratio = %.2fx\n" t_greeks greeks_ratio;
  Printf.printf "         price %.6g  delta %.6g  vega %.6g  rho %.6g\n\n%!"
    (scalar_out "price" g_outs) (greek "s0") (greek "vol") (greek "rate");

  (* One machine-readable line, the only thing scripts/bench-record.sh reads.
     Keep the key set and the spelling stable: it is a data format. *)
  Printf.printf
    "RESULT card=%S dtype=%s n=%d saxpy_cuda_s=%.4g poly_cuda_s=%.4g poly_gflops=%.4g \
     saxpy_speedup=%.4g poly_speedup=%.4g greeks_ratio=%.4g hostmem=%s h2d_gbs=%.4g \
     d2h_gbs=%.4g\n\
     %!"
    card dtype_s n t_saxpy t_poly
    (gflops ~n ~flops_per_elem:poly_flops t_poly)
    saxpy_speedup poly_speedup greeks_ratio mem_label (bus t_h2d) (bus t_d2h)
