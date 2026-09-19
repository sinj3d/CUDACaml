# T23 — Longstaff–Schwartz American put

## Goal

The exotic that needs linear algebra. An American put by least-squares
Monte Carlo: simulate paths once, then walk backwards through time; at
each step regress the discounted continuation value on a polynomial basis
of the spot over the in-the-money paths, and exercise where intrinsic
beats continuation. The regression normal equations are `matmul`s, the
3×3 solve is on the CPU, and the driver is a functor over `Backend.S` so
it runs identically on the interpreter and on CUDA.

Depends on: T16, T21, T22. Phase 4 integration.

## Files you own

- `examples/lsm.ml` (new), `examples/dune`
- `test/staged/unit/test_lsm.ml` → promote

## Interfaces (`examples/lsm.ml`)

```ocaml
open Ocaml_cuda

(** CRR binomial American put, the reference. *)
val binomial_american_put : steps:int -> Black_scholes.market -> float

(** Black–Scholes European put, closed form (put–call parity on [Black_scholes.analytic]). *)
val european_put : Black_scholes.market -> float

module Make (B : Backend.S) : sig
  type t
  (** Compiles the three graphs once for the given sizes. *)
  val create : n_paths:int -> n_steps:int -> Black_scholes.market -> t

  (** Runs the whole algorithm for one seed and returns the price. *)
  val price : t -> seed:int32 -> float
end
```

## Algorithm

`dt = T / n_steps`, `disc = exp (-r dt)`. Basis `φ(S) = [1; S/K; (S/K)²]`
(scaling by K keeps the normal equations well conditioned).

Three compiled graphs, all F64:

1. **paths** (inputs `seed`): `S : [n_paths; n_steps]` via `Rng.normal`,
   `scan_rows add` of log-increments, `exp`, times `s0`. Output `"S"`.
2. **regress** (inputs `S`, `cf : [n_paths]`, `t : I32 scalar`):
   `St = gather (p*n_steps + t) S` (column `t`, using `broadcast` of `t`);
   `itm = select (lt St K) 1 0`; `X = [n_paths; 3]` basis rows masked by
   `itm` (built with `map`s over `iota [n_paths; 3]` and a `gather` of `St`
   by `k / 3`); `y = disc * cf` masked. Outputs `"xtx" = matmul (transpose X) X : [3;3]`
   and `"xty" = matmul (transpose X) (reshape [n_paths; 1] y) : [3;1]`.
3. **update** (inputs `S`, `cf`, `t`, `beta : [3;1]`): `St` as above,
   `cont = matmul X beta` reshaped to `[n_paths]`, `ex = max (K - St) 0`;
   new `cf = select (and_ (gt ex 0) (gt ex cont)) ex (disc * cf)`. Output `"cf"`.

Driver: `S = paths seed`; `cf = max (K - S_last) 0`; for `t = n_steps-2`
down to `0`: `(xtx, xty) = regress`, `beta = solve (xtx + 1e-10 I) xty`
(Gaussian elimination with partial pivoting, 3×3, in OCaml; if fewer than
3 paths are in the money, `beta = 0`), `cf = update`. Price =
`disc * mean cf` (the last discount from `t = 0` to today).

Note the regression uses `cf` discounted one step, and `update` compares
against the fitted continuation but *keeps* the realised discounted cash
flow where it does not exercise; that is Longstaff–Schwartz, not a
Tsitsiklis–Van Roy fit-and-replace. The known small low bias is fine.

## Failure modes to avoid

- Regressing on all paths instead of the in-the-money ones (large
  downward bias; the test's tolerance will not absorb it).
- Rebuilding or recompiling any graph inside the time loop. `t` and `beta`
  are Params for exactly this reason.
- Reading `S` back to the host each step: `B.run` returns host values, so
  the round trip is unavoidable until T25's resident values; keep it to
  `S` once (it is an *input* to graphs 2 and 3, so upload it each call;
  the T25 follow-up replaces this) and say in the report how long the
  loop takes on each backend.
- Comparing to the European put and calling it done: the test compares to
  the binomial American price.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_lsm.exe
```

## Tests (already written: `test/staged/unit/test_lsm.ml`)

Interpreter backend, F64, seed 11, `market` from `Black_scholes`.

- `binomial_american_put ~steps:500 market` is 6.09 within 0.02; `european_put market` is 5.5735 within 1e-3
- `Make (Backend_interp)`: `n_paths 8192`, `n_steps 50`: price within 0.25 of the binomial value and ≥ `european_put − 0.05`
- deterministic: the same seed twice gives the same price; two seeds differ by less than 0.5
- `create` compiles exactly three graphs (the test wraps the backend in a counting functor)
