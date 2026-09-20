(** Longstaff–Schwartz: an American put by least-squares Monte Carlo.

    This is the exotic that needs linear algebra. Paths are simulated once
    ({!Rng.normal} plus a [scan_rows] of the log-increments, exactly as in
    {!Black_scholes.call_mc_paths}); then the algorithm walks backwards
    through the exercise dates. At each date the discounted continuation
    value is regressed on the polynomial basis [[1; S/K; (S/K)^2]] over the
    in-the-money paths only, and a path exercises where its intrinsic value
    beats the fitted continuation.

    The regression is solved through its normal equations, so the device
    does two {!Dsl.matmul}s per step — [XᵀX : [3;3]] and [Xᵀy : [3;1]] —
    and the host does the 3x3 solve. The basis is scaled by the strike so
    that [XᵀX] stays well conditioned at any spot level.

    Three graphs are compiled once by {!Make.create} and then only *run*:
    the time index [t] and the fitted coefficients [beta] are [Param]s
    precisely so that the loop never rebuilds or re-JITs anything.

    The driver is a functor over {!Backend.S}, so the interpreter and CUDA
    run the identical program. Everything is F64: the regression is a
    difference of large numbers and f32 reassociation would show. *)

open Ocaml_cuda

(* ------------------------------------------------------------------ *)
(* Host-side references                                                *)
(* ------------------------------------------------------------------ *)

(** Cox–Ross–Rubinstein binomial American put. [u = e^{v sqrt dt}],
    [d = 1/u], risk-neutral [p = (e^{r dt} - d) / (u - d)]; at every node
    the value is the better of exercising and the discounted continuation.
    With [steps = 500] this is accurate to about a cent and is what the
    Monte Carlo price is measured against. *)
let binomial_american_put ~steps (m : Black_scholes.market) =
  let n = steps in
  let dt = m.maturity /. float_of_int n in
  let u = Stdlib.exp (m.vol *. Stdlib.sqrt dt) in
  let d = 1.0 /. u in
  let disc = Stdlib.exp (-.m.rate *. dt) in
  let p = ((1.0 /. disc) -. d) /. (u -. d) in
  (* [spot i j] is the node after [i] steps with [j] of them up. *)
  let spot i j = m.s0 *. (u ** float_of_int j) *. (d ** float_of_int (i - j)) in
  let v = Array.init (n + 1) (fun j -> Stdlib.max (m.strike -. spot n j) 0.0) in
  for i = n - 1 downto 0 do
    for j = 0 to i do
      let cont = disc *. ((p *. v.(j + 1)) +. ((1.0 -. p) *. v.(j))) in
      v.(j) <- Stdlib.max cont (m.strike -. spot i j)
    done
  done;
  v.(0)

(** European put in closed form, by put–call parity on
    {!Black_scholes.analytic}: [P = C - S + K e^{-rT}]. It is a lower bound
    on the American put, so it is the cheap sanity check that the backward
    induction has not thrown away the early-exercise premium. *)
let european_put (m : Black_scholes.market) =
  let call = (Black_scholes.analytic m).price in
  call -. m.s0 +. (m.strike *. Stdlib.exp (-.m.rate *. m.maturity))

(* ------------------------------------------------------------------ *)
(* The 3x3 solve                                                       *)
(* ------------------------------------------------------------------ *)

(** Gaussian elimination with partial pivoting on a 3x3 system. A singular
    (or non-finite) system yields the zero vector, which is the right answer
    here: it means "no usable regression", so nothing exercises early at
    this date and the cash flow simply rolls back discounted. *)
let solve3 (a0 : float array array) (b0 : float array) : float array =
  let a = Array.map Array.copy a0 and b = Array.copy b0 in
  let x = Array.make 3 0.0 in
  (try
     for k = 0 to 2 do
       let piv = ref k in
       for i = k + 1 to 2 do
         if Float.abs a.(i).(k) > Float.abs a.(!piv).(k) then piv := i
       done;
       if not (Float.abs a.(!piv).(k) > 1e-300) then raise Exit;
       if !piv <> k then begin
         let ra = a.(k) in
         a.(k) <- a.(!piv);
         a.(!piv) <- ra;
         let rb = b.(k) in
         b.(k) <- b.(!piv);
         b.(!piv) <- rb
       end;
       for i = k + 1 to 2 do
         let f = a.(i).(k) /. a.(k).(k) in
         for j = k to 2 do
           a.(i).(j) <- a.(i).(j) -. (f *. a.(k).(j))
         done;
         b.(i) <- b.(i) -. (f *. b.(k))
       done
     done;
     for i = 2 downto 0 do
       let s = ref b.(i) in
       for j = i + 1 to 2 do
         s := !s -. (a.(i).(j) *. x.(j))
       done;
       x.(i) <- !s /. a.(i).(i)
     done;
     if not (Array.for_all Float.is_finite x) then raise Exit
   with Exit -> Array.fill x 0 3 0.0);
  x

(* ------------------------------------------------------------------ *)
(* Host <-> Value helpers                                              *)
(* ------------------------------------------------------------------ *)

let float_at (p : Value.packed) i : float =
  match p with
  | Value.P v -> (
      match Value.dtype v with
      | Dtype.F32 -> Value.get v i
      | Dtype.F64 -> Value.get v i
      | Dtype.I32 | Dtype.I64 | Dtype.Bool -> invalid_arg "Lsm: expected a float value")

let to_floats (p : Value.packed) : float array =
  match p with Value.P v -> Array.init (Value.numel v) (float_at p)

(* ------------------------------------------------------------------ *)
(* The graphs                                                          *)
(* ------------------------------------------------------------------ *)

(* Pure graph construction: no backend is named here, so every driver
   below prices with exactly the same three graphs. *)
module Graphs = struct
  let dtype = Dtype.F64
  let vec n = Shape.of_dims [ n ]
  let i32 v = Value.P (Value.of_list Dtype.I32 Shape.scalar [ v ])

  (* --- graph 1: the paths ----------------------------------------- *)

  (* [S : [n_paths; n_steps]], column [t] holding time [(t+1) dt], so the
     exercise dates are [dt, 2dt, ..., T] and the last column is maturity.
     GBM is exact under this discretisation: the log-increments are iid
     normals, [scan_rows] accumulates them along each row. The market is
     fixed at [create] time, so [s0], [vol] and [rate] are folded into
     constants and [seed] is the only Param. *)
  let paths_graph ~n_paths ~n_steps (m : Black_scholes.market) =
    let open Dsl in
    let grid = Shape.of_dims [ n_paths; n_steps ] in
    let kf v = const dtype v in
    let seed = scalar "seed" Dtype.I32 in
    let dt = m.maturity /. float_of_int n_steps in
    let drift = (m.rate -. (0.5 *. m.vol *. m.vol)) *. dt in
    let vol_dt = m.vol *. Stdlib.sqrt dt in
    let z = Rng.normal dtype ~key:seed grid in
    let incr = map (fun zi -> add (kf drift) (mul (kf vol_dt) zi)) z in
    let cum = scan_rows add ~init:(kf 0.0) incr in
    let s = map (fun c -> mul (kf m.s0) (exp c)) cum in
    Graph.create ~name:"lsm_paths" ~outputs:[ ("S", Tensor.P s) ]

  (* --- the shared front half of graphs 2 and 3 --------------------- *)

  type parts = {
    cf : float Tensor.t;  (** the running cash flow, one per path *)
    st : float Tensor.t;  (** column [t] of [S] *)
    itm : float Tensor.t;  (** 1.0 where in the money, else 0.0 *)
    xm : float Tensor.t;  (** the masked design matrix, [[n_paths; 3]] *)
  }

  (* Column [t] of [S] is the gather of [p * n_steps + t] over the paths;
     [t] is a scalar Param broadcast up, which is what keeps it out of the
     compiled constants. The design matrix is built from [iota [n_paths;3]]:
     row [k / 3], basis index [k mod 3]. Multiplying through by the
     in-the-money indicator is what restricts the regression to the
     in-the-money paths — an out-of-the-money row is all zeros and
     contributes nothing to either normal equation. *)
  let basis ~n_paths ~n_steps ~strike =
    let open Dsl in
    let grid = Shape.of_dims [ n_paths; n_steps ] in
    let g3 = Shape.of_dims [ n_paths; 3 ] in
    let kf v = const dtype v in
    let ki v = const Dtype.I32 (Int32.of_int v) in
    let s = param "S" dtype grid in
    let cf = param "cf" dtype (vec n_paths) in
    let tb = broadcast (vec n_paths) (scalar "t" Dtype.I32) in
    let idx = map2 (fun p tv -> add (mul p (ki n_steps)) tv) (iota (vec n_paths)) tb in
    let st = gather idx s in
    let itm = map (fun x -> select (lt x (kf strike)) (kf 1.0) (kf 0.0)) st in
    let row = map (fun k -> div k (ki 3)) (iota g3) in
    let st3 = gather row st and itm3 = gather row itm in
    let phi =
      map2
        (fun k sv ->
          let j = sub k (mul (div k (ki 3)) (ki 3)) in
          let x = div sv (kf strike) in
          select (eq j (ki 0)) (kf 1.0) (select (eq j (ki 1)) x (mul x x)))
        (iota g3) st3
    in
    { cf; st; itm; xm = map2 mul phi itm3 }

  (* --- graph 2: the normal equations ------------------------------- *)

  (* [y] is the cash flow discounted one step back to [t], masked the same
     way as [X]. [XᵀX] is [[3;3]] and [Xᵀy] is [[3;1]], both tiny, so the
     host solve that follows costs nothing. [XᵀX]'s (0,0) entry is the
     number of in-the-money paths, since the first basis column *is* the
     indicator. *)
  let regress_graph ~n_paths ~n_steps ~strike ~disc =
    let open Dsl in
    let p = basis ~n_paths ~n_steps ~strike in
    let kf v = const dtype v in
    let y = map2 (fun c mk -> mul (mul (kf disc) c) mk) p.cf p.itm in
    let xt = transpose p.xm in
    Graph.create ~name:"lsm_regress"
      ~outputs:
        [
          ("xtx", Tensor.P (matmul xt p.xm));
          ("xty", Tensor.P (matmul xt (reshape (Shape.of_dims [ n_paths; 1 ]) y)));
        ]

  (* --- graph 3: the exercise decision ------------------------------ *)

  (* Longstaff–Schwartz, not Tsitsiklis–Van Roy: the fitted continuation
     decides *whether* to exercise, but where a path does not exercise it
     keeps its realised cash flow, discounted, not the fitted value. *)
  let update_graph ~n_paths ~n_steps ~strike ~disc =
    let open Dsl in
    let p = basis ~n_paths ~n_steps ~strike in
    let kf v = const dtype v in
    let beta = param "beta" dtype (Shape.of_dims [ 3; 1 ]) in
    let cont = reshape (vec n_paths) (matmul p.xm beta) in
    let exercises s c =
      let e = max (sub (kf strike) s) (kf 0.0) in
      and_ (gt e (kf 0.0)) (gt e c)
    in
    let exv =
      map2 (fun s c -> select (exercises s c) (sub (kf strike) s) (kf 0.0)) p.st cont
    in
    let keep = map2 (fun s c -> select (exercises s c) (kf 0.0) (kf 1.0)) p.st cont in
    let held = map2 (fun k c -> mul k (mul (kf disc) c)) keep p.cf in
    Graph.create ~name:"lsm_update" ~outputs:[ ("cf", Tensor.P (map2 add exv held)) ]
end

(* ------------------------------------------------------------------ *)
(* The pricer                                                          *)
(* ------------------------------------------------------------------ *)

module type S = sig
  type t

  (** Compiles the three graphs once for the given sizes. *)
  val create : n_paths:int -> n_steps:int -> Black_scholes.market -> t

  (** Runs the whole algorithm for one seed and returns the price. *)
  val price : t -> seed:int32 -> float
end

module Make (B : Backend.S) : S = struct
  open Graphs

  type t = {
    n_paths : int;
    n_steps : int;
    strike : float;
    disc : float;
    paths : B.compiled;
    regress : B.compiled;
    update : B.compiled;
  }

  (* --- the driver --------------------------------------------------- *)

  let create ~n_paths ~n_steps (m : Black_scholes.market) =
    if n_paths < 1 || n_steps < 1 then invalid_arg "Lsm.create: n_paths and n_steps must be >= 1";
    let dt = m.maturity /. float_of_int n_steps in
    let disc = Stdlib.exp (-.m.rate *. dt) in
    let strike = m.strike in
    {
      n_paths;
      n_steps;
      strike;
      disc;
      paths = B.compile (paths_graph ~n_paths ~n_steps m);
      regress = B.compile (regress_graph ~n_paths ~n_steps ~strike ~disc);
      update = B.compile (update_graph ~n_paths ~n_steps ~strike ~disc);
    }

  let price t ~seed =
    let n = t.n_paths and ns = t.n_steps in
    (* [S] is downloaded once and handed straight back as an input to the
       two per-step graphs; until T25's device-resident values there is no
       way to keep it there, but at least it crosses the bus as a Value
       that is never converted on the host. *)
    let s = List.assoc "S" (B.run t.paths ~inputs:[ ("seed", i32 seed) ]) in
    let cf0 =
      List.init n (fun p -> Stdlib.max (t.strike -. float_at s ((p * ns) + ns - 1)) 0.0)
    in
    let cf = ref (Value.P (Value.of_list Dtype.F64 (vec n) cf0)) in
    for step = ns - 2 downto 0 do
      let tv = i32 (Int32.of_int step) in
      let outs = B.run t.regress ~inputs:[ ("S", s); ("cf", !cf); ("t", tv) ] in
      let xtx = to_floats (List.assoc "xtx" outs) in
      let xty = to_floats (List.assoc "xty" outs) in
      (* [xtx.(0)] counts the in-the-money paths; below three of them the
         3x3 fit is not determined and the regression is skipped. *)
      let beta =
        if xtx.(0) < 3.0 then [| 0.0; 0.0; 0.0 |]
        else
          let a =
            Array.init 3 (fun i ->
                Array.init 3 (fun j ->
                    xtx.((i * 3) + j) +. if i = j then 1e-10 else 0.0))
          in
          solve3 a xty
      in
      let bv = Value.P (Value.of_list Dtype.F64 (Shape.of_dims [ 3; 1 ]) (Array.to_list beta)) in
      let outs =
        B.run t.update ~inputs:[ ("S", s); ("cf", !cf); ("t", tv); ("beta", bv) ]
      in
      cf := List.assoc "cf" outs
    done;
    let final = to_floats !cf in
    t.disc *. (Array.fold_left ( +. ) 0.0 final /. float_of_int n)
end

(* ------------------------------------------------------------------ *)
(* The resident CUDA driver                                            *)
(* ------------------------------------------------------------------ *)

(** The same algorithm as {!Make}, specialised to the CUDA backend and
    written against T25's device-resident values.

    {!Make} hands the whole path matrix [S] back to the host after the
    simulation and then re-uploads it as an input to both per-step
    graphs: [2 (n_steps - 1)] crossings of the bus carrying
    [n_paths * n_steps] doubles each. Here [S] is uploaded never and
    downloaded once, and the running cash flow [cf] — the other big
    intermediate — never leaves the device at all: {!Backend_cuda.run_resident}
    substitutes the caller's buffers for the plan's own, so the kernels
    read and write them in place. What crosses the bus per step is the
    two normal equations ([xtx] 9 doubles, [xty] 3), the fitted [beta]
    (3), and the step index (1 int).

    The arithmetic is identical to {!Make}'s, kernel for kernel and input
    for input, so the two agree bit for bit. *)
module Make_cuda_resident : S = struct
  open Graphs
  module B = Backend_cuda

  type t = {
    n_paths : int;
    n_steps : int;
    strike : float;
    disc : float;
    paths : B.compiled;
    regress : B.compiled;
    update : B.compiled;
  }

  let create ~n_paths ~n_steps (m : Black_scholes.market) =
    if n_paths < 1 || n_steps < 1 then
      invalid_arg "Lsm.Make_cuda_resident.create: n_paths and n_steps must be >= 1";
    let dt = m.maturity /. float_of_int n_steps in
    let disc = Stdlib.exp (-.m.rate *. dt) in
    let strike = m.strike in
    {
      n_paths;
      n_steps;
      strike;
      disc;
      paths = B.compile (paths_graph ~n_paths ~n_steps m);
      regress = B.compile (regress_graph ~n_paths ~n_steps ~strike ~disc);
      update = B.compile (update_graph ~n_paths ~n_steps ~strike ~disc);
    }

  (* One device round trip in, one out: everything in between is
     residents handed from one compiled graph to the next. *)
  let price t ~seed =
    let n = t.n_paths and ns = t.n_steps in
    let seed_r = B.upload (i32 seed) in
    let s = List.assoc "S" (B.run_resident t.paths ~inputs:[ ("seed", seed_r) ]) in
    B.free_resident seed_r;
    (* The only look the host gets at [S], and it is once, not per step:
       the initial cash flow is the intrinsic value at maturity, which is
       the last column. [S] itself stays where it was produced. *)
    let s_host = B.download s in
    let cf0 =
      List.init n (fun p -> Stdlib.max (t.strike -. float_at s_host ((p * ns) + ns - 1)) 0.0)
    in
    let cf = ref (B.upload (Value.P (Value.of_list Dtype.F64 (vec n) cf0))) in
    for step = ns - 2 downto 0 do
      let tv = B.upload (i32 (Int32.of_int step)) in
      let outs = B.run_resident t.regress ~inputs:[ ("S", s); ("cf", !cf); ("t", tv) ] in
      let rxtx = List.assoc "xtx" outs and rxty = List.assoc "xty" outs in
      let xtx = to_floats (B.download rxtx) and xty = to_floats (B.download rxty) in
      B.free_resident rxtx;
      B.free_resident rxty;
      (* [xtx.(0)] counts the in-the-money paths; below three of them the
         3x3 fit is not determined and the regression is skipped. *)
      let beta =
        if xtx.(0) < 3.0 then [| 0.0; 0.0; 0.0 |]
        else
          let a =
            Array.init 3 (fun i ->
                Array.init 3 (fun j -> xtx.((i * 3) + j) +. if i = j then 1e-10 else 0.0))
          in
          solve3 a xty
      in
      let bv =
        B.upload (Value.P (Value.of_list Dtype.F64 (Shape.of_dims [ 3; 1 ]) (Array.to_list beta)))
      in
      let outs =
        B.run_resident t.update ~inputs:[ ("S", s); ("cf", !cf); ("t", tv); ("beta", bv) ]
      in
      let next = List.assoc "cf" outs in
      (* [run_resident] has synchronised, so the previous cash flow and
         this step's scalars are dead and their buffers can go back. *)
      B.free_resident !cf;
      B.free_resident tv;
      B.free_resident bv;
      cf := next
    done;
    let final = to_floats (B.download !cf) in
    B.free_resident !cf;
    B.free_resident s;
    t.disc *. (Array.fold_left ( +. ) 0.0 final /. float_of_int n)
end
