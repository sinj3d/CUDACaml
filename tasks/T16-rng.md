# T16 — `Rng`: counter-based random numbers as element functions

## Goal

A new library, `ocaml_cuda.rng`, that builds random tensors out of the DSL
and nothing else. The generator is **Philox-4x32-10** (Salmon, Moraes,
Dror, Shaw 2011; the default in cuRAND, JAX and NumPy): a pure function
from (counter, key) to four 32-bit words. Element `i` uses counter `i`, so
a random tensor is `map` over `iota`: stateless, reproducible, parallel by
construction, and it fuses into whatever consumes it. The seed is a
`Param` of shape `Shape.scalar`, broadcast, so a new seed is a new input
and not a re-JIT.

Depends on: T13 (`broadcast`, `scalar`), T15 (bit ops, `erfinv`). Phase 2.

## Files you own

- `lib/rng/dune`, `lib/rng/rng.ml`, `lib/rng/rng.mli` (new)
- `lib/ocaml_cuda.ml`, `lib/dune` (add `module Rng = Ocaml_cuda_rng.Rng` under a "Layer 1½: libraries over the DSL" comment, and the dependency)
- `test/staged/unit/test_rng.ml` → promote

```
(library
 (name ocaml_cuda_rng)
 (public_name ocaml_cuda.rng)
 (libraries ocaml_cuda_ir))
```

## Interfaces (`rng.mli`)

```ocaml
open Ocaml_cuda_ir

module Philox : sig
  (** Philox-4x32-10. [ctr] is the 128-bit counter as four I32 words
      (c0 least significant), [key] the 64-bit key as two I32 words. Returns
      the four output words. Pure expression; every word may be used. *)
  val round10 :
    ctr:int32 Expr.t * int32 Expr.t * int32 Expr.t * int32 Expr.t ->
    key:int32 Expr.t * int32 Expr.t ->
    int32 Expr.t * int32 Expr.t * int32 Expr.t * int32 Expr.t
end

(** [u32 ~key shape]: element [i] is output word 0 of
    [Philox.round10 ~ctr:(i, 0, 0, 0) ~key:(key, 0)]. [key] must have numel 1
    (normally [Dsl.scalar "seed" Dtype.I32]). Uniform over all 2^32 bit
    patterns, delivered as a signed [int32]. *)
val u32 : key:int32 Tensor.t -> Shape.t -> int32 Tensor.t

(** [to_unit_interval dtype w]: the 32 bits of [w] read as unsigned, mapped
    to the OPEN interval (0, 1) as [(u + 0.5) * 2^-32], computed in F64 and
    then cast to [dtype]. *)
val to_unit_interval : float Dtype.t -> int32 Expr.t -> float Expr.t

(** Standard normal from a uniform in (0,1): [sqrt 2 * erfinv (2u - 1)]. *)
val normal_inv_cdf : float Expr.t -> float Expr.t

(** [uniform dtype ~key shape] = [map (to_unit_interval dtype) (u32 ~key shape)]. *)
val uniform : float Dtype.t -> key:int32 Tensor.t -> Shape.t -> float Tensor.t

(** [normal dtype ~key shape] = [map normal_inv_cdf (uniform dtype ~key shape)]. *)
val normal : float Dtype.t -> key:int32 Tensor.t -> Shape.t -> float Tensor.t
```

`float Dtype.t` is inhabited by exactly `Dtype.F32` and `Dtype.F64`.

## Implementation

Philox constants: `M0 = 0xD2511F53`, `M1 = 0xCD9E8D7C`, `W0 = 0x9E3779B9`,
`W1 = 0xBB67AE85` (as `Int32.of_string "0x..."`, which wraps to negative).

One round on `(c0, c1, c2, c3)` with key `(k0, k1)`:

```
hi0, lo0 = mulhilo32 (M0, c0)
hi1, lo1 = mulhilo32 (M1, c2)
(c0, c1, c2, c3) <- (hi1 ^ c1 ^ k0,  lo1,  hi0 ^ c3 ^ k1,  lo0)
```

then bump the key: `k0 += W0`, `k1 += W1`. Ten rounds; the bump happens
after each of the first nine (the tenth round's output is the result).

`mulhilo32 a b` in `Expr`: `lo = mul a b` (I32 wraps); `hi` = take both
operands to I64 **zero-extended** (`bit_and (cast I64 a) (const I64 0xFFFFFFFFL)`),
multiply, `shr` by 32 (logical, so the sign of the I64 product does not
matter), `cast I32`. Both sides evaluate the same expression tree, so
bit-exactness follows from T15.

`u32`: `let i = iota shape in let k = broadcast shape key in map2 (fun i k -> word0 (round10 ~ctr:(i, z, z, z) ~key:(k, z))) i k`
with `z = const I32 0l`. Do not use `index ()` for the counter: `iota` fuses
identically and keeps the graph reshape-safe.

`to_unit_interval`: `u = select (lt w 0) (cast F64 w + 4294967296.0) (cast F64 w)`,
then `cast dtype ((u + 0.5) * 2.3283064365386963e-10)`.

## Invariants

- Same key, same shape, same values, on every backend, forever. The KATs
  pin this.
- `uniform` never returns 0.0 or 1.0 (the `+0.5` offset); `normal` is
  therefore always finite.
- `u32` builds no `Param` other than the caller's key.

## Failure modes to avoid

- Sign-extending in `mulhilo32` (`cast I64` alone): negative `a` corrupts
  the high word. The KAT with all-ones counter and key exists to catch it.
- Bumping the key after the tenth round: harmless to the output but
  confuses anyone comparing with Random123; and bumping *before* the first
  round is wrong.
- Computing `to_unit_interval` in F32 for F32 output before the offset:
  `(u + 0.5)` in f32 collapses low bits; do the arithmetic in F64 and cast last.
- Using `Float.of_int` tricks in the interpreter: nothing here is in the
  interpreter; `Rng` is a pure DSL library.

## Known-answer vectors (Random123 `kat_vectors`, philox4x32 10 rounds)

```
ctr 00000000 00000000 00000000 00000000  key 00000000 00000000
  -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8
ctr ffffffff ffffffff ffffffff ffffffff  key ffffffff ffffffff
  -> 408f276d 41c83b0e a20bc7c6 6d5451fd
ctr 243f6a88 85a308d3 13198a2e 03707344  key a4093822 299f31d0
  -> d16cfe09 94fdcceb 5001e420 24126ea1
```

If your implementation matches NumPy's `np.random.Philox(counter=..., key=...)`
raw output but not these, report a suspected transcription error with the
NumPy output; do not edit the test (tasks/README.md rule 2).

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_rng.exe
```

## Tests (already written: `test/staged/unit/test_rng.ml`)

- the three KATs, all four words each, via a 4-output interp graph over a `[1]` tensor
- `u32 ~key:0` element 0 is `0x6627e8d5`; elements 0..3 are pairwise distinct
- same key twice → identical tensors; key 0 vs key 1 → differ in ≥ 90% of 1024 positions
- `uniform F64` over 65536: every value in (0,1); mean within 0.01 of 0.5; variance within 0.005 of 1/12; 16-bucket chi-square statistic < 45
- `normal F64` over 65536: mean within 0.02 of 0; variance within 0.05 of 1; fraction with |z| > 1.96 within 0.006 of 0.05; all finite
- `normal F32`: all finite; mean within 0.02
- `uniform` graph has exactly one param, named `seed`, and no `Upload` other than it
- `u32` of shape `[4; 8]` reshaped equals `u32` of `[32]` element-wise (counter is the flat index)
