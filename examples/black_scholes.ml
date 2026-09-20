(** Black–Scholes: a European call priced by Monte Carlo, with pathwise
    Greeks produced by [Grad.grad] rather than by bumping.

    This is the first program a quant will look at, so it is deliberately
    written the way a user would write it: [s0], [vol] and [rate] are
    [Param]s of shape [Shape.scalar], broadcast up to the path shape, and
    the Greeks come out of the same graph as the price. Nothing here is
    special-cased anywhere in the compiler — the gradient graph is built
    from [Dsl] calls only, so both backends run it unchanged.

    Three graphs:

    - {!call_mc}       one-step GBM, [n_paths] independent normals;
    - {!call_mc_paths} an [n_steps]-step path built with [scan_rows], which
                       is exact for GBM and therefore prices the same
                       option — it exists to exercise the row primitives;
    - {!call_greeks}   [call_mc] plus ∂price/∂s0, ∂price/∂vol, ∂price/∂rate.

    Every price is checked against {!analytic}, the closed form. *)

open Cudacaml

type market = {
  s0 : float;
  strike : float;
  vol : float;
  rate : float;
  maturity : float;
}

let market = { s0 = 100.0; strike = 100.0; vol = 0.2; rate = 0.05; maturity = 1.0 }

type closed_form = { price : float; delta : float; vega : float; rho : float }

(* Stdlib names, spelled out: [Dsl] shadows [sqrt], [exp], [log], [min] and
   [max] with the expression builders, and this half of the file is host
   arithmetic. *)
let norm_cdf x = 0.5 *. (1.0 +. Float.erf (x /. Stdlib.sqrt 2.0))
let norm_pdf x = Stdlib.exp (-0.5 *. x *. x) /. Stdlib.sqrt (2.0 *. Float.pi)

(** The textbook call: [C = S N(d1) - K e^{-rT} N(d2)], with
    [delta = N(d1)], [vega = S phi(d1) sqrt T] and
    [rho = K T e^{-rT} N(d2)]. *)
let analytic m =
  let sqrt_t = Stdlib.sqrt m.maturity in
  let d1 =
    (Stdlib.log (m.s0 /. m.strike)
    +. ((m.rate +. (0.5 *. m.vol *. m.vol)) *. m.maturity))
    /. (m.vol *. sqrt_t)
  in
  let d2 = d1 -. (m.vol *. sqrt_t) in
  let disc = Stdlib.exp (-.m.rate *. m.maturity) in
  {
    price = (m.s0 *. norm_cdf d1) -. (m.strike *. disc *. norm_cdf d2);
    delta = norm_cdf d1;
    vega = m.s0 *. norm_pdf d1 *. sqrt_t;
    rho = m.strike *. m.maturity *. disc *. norm_cdf d2;
  }

(* ------------------------------------------------------------------ *)
(* Graphs                                                              *)
(* ------------------------------------------------------------------ *)

let vec n = Shape.of_dims [ n ]

(* The three market params plus the RNG key, in one place so the two
   pricers cannot drift apart. *)
let params dtype =
  let open Dsl in
  ( scalar "s0" dtype,
    scalar "vol" dtype,
    scalar "rate" dtype,
    scalar "seed" Dtype.I32 )

(* [exp (-rT) * mean (max (S_T - K) 0)].

   [n_paths] is a compile-time constant, so the division by it is a
   constant folded into one [map]; making it a device scalar would only add
   an input and a dependency. The discount is the one place [rate] appears
   outside the drift, and forgetting it makes [rho] wrong by [T * price]. *)
let discounted_mean ~dtype ~n_paths ~(m : market) st rate =
  let open Dsl in
  let kf v = const dtype v in
  let payoff = map (fun s -> max (sub s (kf m.strike)) (kf 0.0)) st in
  let total = reduce add ~init:(kf 0.0) payoff in
  let mean = map (fun s -> div s (kf (float_of_int n_paths))) total in
  map2 (fun mu r -> mul mu (exp (neg (mul r (kf m.maturity))))) mean rate

(** One-step GBM: [S_T = s0 exp ((r - v^2/2) T + v sqrt T Z)] with [Z] a
    standard normal from [Rng.normal]. Strike and maturity are constants
    baked into the graph; [s0], [vol], [rate] and [seed] are params. *)
let call_mc ~n_paths ~dtype (m : market) =
  let open Dsl in
  let shape = vec n_paths in
  let s0, vol, rate, seed = params dtype in
  let kf v = const dtype v in
  let z = Rng.normal dtype ~key:seed shape in
  let s0b = broadcast shape s0 in
  let volb = broadcast shape vol in
  let rateb = broadcast shape rate in
  let drift =
    map2 (fun r v -> mul (sub r (mul (kf 0.5) (mul v v))) (kf m.maturity)) rateb volb
  in
  let diffusion =
    map2 (fun v zi -> mul (mul v (kf (Stdlib.sqrt m.maturity))) zi) volb z
  in
  let st = map2 (fun s e -> mul s (exp e)) s0b (map2 add drift diffusion) in
  Graph.create ~name:"bs_mc"
    ~outputs:[ ("price", Tensor.P (discounted_mean ~dtype ~n_paths ~m st rate)) ]

(** The same price from an [n_steps]-step path over [[n_paths; n_steps]]:
    log-increments [(r - v^2/2) dt + v sqrt dt Z], [scan_rows add] to get
    [log S_t] along each row, [exp], then the last column by [gather]. GBM
    is exact under this discretisation, so it must agree with {!call_mc} up
    to Monte Carlo noise. *)
let call_mc_paths ~n_paths ~n_steps ~dtype (m : market) =
  let open Dsl in
  let grid = Shape.of_dims [ n_paths; n_steps ] in
  let s0, vol, rate, seed = params dtype in
  let kf v = const dtype v in
  let ki v = const Dtype.I32 (Int32.of_int v) in
  let dt = m.maturity /. float_of_int n_steps in
  let z = Rng.normal dtype ~key:seed grid in
  let s0b = broadcast grid s0 in
  let volb = broadcast grid vol in
  let rateb = broadcast grid rate in
  let drift = map2 (fun r v -> mul (sub r (mul (kf 0.5) (mul v v))) (kf dt)) rateb volb in
  let diffusion = map2 (fun v zi -> mul (mul v (kf (Stdlib.sqrt dt))) zi) volb z in
  let cum = scan_rows add ~init:(kf 0.0) (map2 add drift diffusion) in
  let paths = map exp (map2 (fun s c -> add (log s) c) s0b cum) in
  (* Row [p]'s terminal value lives at flat offset [p * n_steps + n_steps - 1]. *)
  let idx = map (fun p -> add (mul p (ki n_steps)) (ki (n_steps - 1))) (iota (vec n_paths)) in
  let st = gather idx paths in
  Graph.create ~name:"bs_paths"
    ~outputs:[ ("price", Tensor.P (discounted_mean ~dtype ~n_paths ~m st rate)) ]

(** [call_mc] plus delta, vega and rho, all pathwise. The forward outputs
    of the original graph are preserved exactly, so the price this returns
    is bit-for-bit the price {!call_mc} returns. *)
let call_greeks ~n_paths ~dtype (m : market) =
  Grad.grad (call_mc ~n_paths ~dtype m) ~output:"price" ~wrt:[ "s0"; "vol"; "rate" ]

(* ------------------------------------------------------------------ *)
(* Inputs and outputs                                                  *)
(* ------------------------------------------------------------------ *)

(** Inputs for any of the three graphs: the same four scalars. *)
let inputs ~dtype (m : market) ~seed =
  let f name v = (name, Value.P (Value.of_list dtype Shape.scalar [ v ])) in
  [
    f "s0" m.s0;
    f "vol" m.vol;
    f "rate" m.rate;
    ("seed", Value.P (Value.of_list Dtype.I32 Shape.scalar [ seed ]));
  ]

(** Read a scalar float output by name. *)
let scalar_out outs name : float =
  match List.assoc name outs with
  | Value.P v -> (
      match Value.dtype v with
      | Dtype.F32 -> Value.get v 0
      | Dtype.F64 -> Value.get v 0
      | Dtype.I32 | Dtype.I64 | Dtype.Bool ->
          invalid_arg
            (Printf.sprintf "Black_scholes.scalar_out: output %S is not a float" name))
