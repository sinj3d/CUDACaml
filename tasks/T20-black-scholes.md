# T20 — Black–Scholes Monte Carlo with pathwise Greeks

## Goal

The first program a quant will look at. A European call priced by Monte
Carlo, its delta/vega/rho produced by `Grad.grad`, all checked against the
closed form. Three examples in the registry, one benchmark workload that
measures the gradient-to-forward cost ratio instead of asserting it.

Depends on: T16, T18. Phase 2+3 integration.

## Files you own

- `examples/black_scholes.ml` (new), `examples/programs.ml`, `examples/dune`
- `bench/bench.ml`
- `test/staged/unit/test_black_scholes.ml` → promote

## Interfaces (`examples/black_scholes.ml`)

```ocaml
open Ocaml_cuda

type market = { s0 : float; strike : float; vol : float; rate : float; maturity : float }
val market : market   (* s0 100, strike 100, vol 0.2, rate 0.05, maturity 1.0 *)

type closed_form = { price : float; delta : float; vega : float; rho : float }
(** Black–Scholes call via [Float.erf]. *)
val analytic : market -> closed_form

(** One-step GBM: S_T = s0 exp((r - v²/2) T + v sqrt T Z).
    Params (all [Shape.scalar] except the seed, which is I32 scalar):
      "s0" "vol" "rate" : dtype;  "seed" : I32.
    Strike and maturity are constants baked into the graph.
    Output "price" = exp(-rT) * mean(max(S_T - K, 0)). *)
val call_mc : n_paths:int -> dtype:float Dtype.t -> market -> Graph.t

(** Same price from an [n_steps]-step path built with [Rng.normal] over
    [[n_paths; n_steps]], [scan_rows add] of the log-increments, [exp], and
    the last column via [gather]. Exact for GBM, so it must agree with
    [call_mc] up to Monte Carlo noise; it exists to exercise T14. *)
val call_mc_paths : n_paths:int -> n_steps:int -> dtype:float Dtype.t -> market -> Graph.t

(** [Grad.grad (call_mc ...) ~output:"price" ~wrt:["s0"; "vol"; "rate"]]. *)
val call_greeks : n_paths:int -> dtype:float Dtype.t -> market -> Graph.t

(** Inputs for any of the three, with the given seed. *)
val inputs : dtype:float Dtype.t -> market -> seed:int32 -> (string * Value.packed) list

(** Convenience: read a scalar output as a float. *)
val scalar_out : (string * Value.packed) list -> string -> float
```

Register in `Programs.all`: `bs_mc` (`call_mc ~n_paths:4096 ~dtype:F64`),
`bs_paths` (`call_mc_paths ~n_paths:512 ~n_steps:16 ~dtype:F64`),
`bs_greeks` (`call_greeks ~n_paths:4096 ~dtype:F64`), each with seed 7.
F64 so that the S1 system check at tolerance 1e-5 is not at the mercy of
f32 reassociation; the F32 path is covered by the timing test.

## Implementation notes

- Broadcasting: `s0`, `vol`, `rate` are `Dsl.scalar` params; use
  `broadcast shape` before `map2`. Make `mean` = `reduce add` then a `map`
  dividing by the constant `n_paths`.
- The payoff `max (S_T - K) 0` is differentiable almost everywhere; at
  `S_T = K` the subgradient convention of T17 applies. Fine for a call.
- `call_mc_paths`: increments `(r - v²/2) dt + v sqrt dt Z` with
  `dt = T / n_steps`; `log S_T = log s0 + Σ increments`; `S_T = exp (...)`;
  the last column is `gather (map (fun p -> p * n_steps + n_steps - 1) (iota [n_paths]))`.
- `bench.ml`: add a third workload, `greeks`: time `call_mc` (F32,
  n = the benchmark's `n`) and `call_greeks` on the same inputs, both warm,
  and print `greeks/price ratio = %.2fx` plus the three gradient values.
  Add `greeks_ratio=` to the `RESULT` line (T12).

## Failure modes to avoid

- Baking `s0` as a constant "for now": the point is that it is a Param and
  differentiable.
- Using `n_paths` as an `I32` scalar param and dividing on device: keep it
  a constant; it is static.
- Forgetting `exp(-rT)` in the discount, which also makes `rho` wrong by
  `T · price`.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_black_scholes.exe
dune exec ocaml-cuda -- check bs_greeks     # GPU machine
```

## Tests (already written: `test/staged/unit/test_black_scholes.ml`)

All on the interpreter, F64, seed 7, deterministic.

- `analytic market` : price 10.4506, delta 0.6368, vega 37.524, rho 53.232 (tol 1e-3)
- `call_mc ~n_paths:65536` price within 0.25 of analytic
- `call_mc_paths ~n_paths:16384 ~n_steps:8` price within 0.5 of analytic
- `call_greeks ~n_paths:65536`: forward price output equals `call_mc`'s exactly; delta within 0.02; vega within 1.5; rho within 1.5 of analytic
- the greeks graph has exactly 4 params (`s0`, `vol`, `rate`, `seed`) and 4 outputs
- `Lower.program` of `call_greeks` has fewer than 40 kernels (a regression guard against un-fused adjoint chains)
- all three registry entries are found by `Programs.find`
