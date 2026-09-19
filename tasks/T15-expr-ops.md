# T15 — Bitwise, shift and transcendental operators in `Expr`

## Goal

Counter-based random number generation is integer arithmetic: multiply,
xor, shift. Normal sampling needs the inverse error function. Neither
exists in `Expr`. Add them, with **bit-exact** integer semantics shared by
the interpreter and the emitter, plus the DSL surface the IR already has
but never exposed (`ne`, `gt`, `ge`, `and_`, `or_`, `not_`).

Depends on: T03, T05, T06 (v1). Phase 1. Independent of T13/T14.

## Files you own

- `lib/ir/expr.ml`, `lib/ir/dsl.ml`, `lib/ir/dsl.mli` (you may edit)
- `lib/backend_interp/ocaml_cuda_backend_interp.ml`
- `lib/backend_cuda/emit.ml`
- `test/staged/unit/test_expr_ops.ml` → promote

`lib/lower/lower.ml` needs no change: `Binop`/`Unop` are carried through
by constructor. Check that it still builds; if a match there is exhaustive
over `binop` values, fix it and say so in the report.

## Interfaces

`expr.ml`:

```ocaml
type binop = Add | Sub | Mul | Div | Min | Max
           | Bit_and | Bit_or | Bit_xor | Shl | Shr
type unop  = Neg | Sqrt | Exp | Log | Abs | Sin | Cos | Erf | Erfinv
```

`dsl.mli` additions:

```ocaml
(** Integer only ([I32]/[I64]); the interpreter raises [Invalid_argument]
    and [Emit] fails on a float operand. *)
val bit_and : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val bit_or  : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val bit_xor : 'a Expr.t -> 'a Expr.t -> 'a Expr.t

(** Shift count is the second operand, of the same dtype, taken modulo the
    bit width (& 31 for I32, & 63 for I64) so no count is undefined.
    [shr] is LOGICAL: the sign bit is not replicated. *)
val shl : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val shr : 'a Expr.t -> 'a Expr.t -> 'a Expr.t

(** Float only. *)
val sin : 'a Expr.t -> 'a Expr.t
val cos : 'a Expr.t -> 'a Expr.t
val erf : 'a Expr.t -> 'a Expr.t
val erfinv : 'a Expr.t -> 'a Expr.t

val ne : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val gt : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val ge : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val and_ : bool Expr.t -> bool Expr.t -> bool Expr.t
val or_  : bool Expr.t -> bool Expr.t -> bool Expr.t
val not_ : bool Expr.t -> bool Expr.t
```

## Implementation

**Interp** (`binop`): `I32`: `Int32.logand/logor/logxor`;
`Shl`: `Int32.shift_left x (Int32.to_int y land 31)`;
`Shr`: `Int32.shift_right_logical x (Int32.to_int y land 31)`. `I64`
likewise with `land 63`. Floats: `invalid_arg`. `float_binop` must not
receive these; restructure so the float arm only sees the six arithmetic
ops (a nested match on the op with an `invalid_arg` fallthrough is fine,
but keep it exhaustive; no wildcard over the whole `binop`).

**Interp** (`unop`): `Sin`/`Cos`: `Stdlib.sin/cos`. `Erf`: `Float.erf`.
`Erfinv`: no stdlib function. Implement `erfinv x` for `F64` to full
double precision as: `x = ±1 → ±infinity`, `|x| > 1 → nan`, `x = 0 → 0`,
otherwise Giles' single-precision initial guess followed by **two Newton
steps** on `Float.erf`:
`y <- y - (erf y - x) / (2/sqrt(pi) * exp(-y*y))`. Round to f32 for `F32`
via the existing `round_f32`. Two steps from Giles' guess (relative error
≈ 1e-7) reach ≈ 1e-15; the test demands 1e-12 on `erfinv (erf x) = x`.
The Giles coefficients:

```
w = -log((1-x)*(1+x))
if w < 5:  w -= 2.5
  p = 2.81022636e-08; p = 3.43273939e-07 + p*w; p = -3.5233877e-06 + p*w;
  p = -4.39150654e-06 + p*w; p = 0.00021858087 + p*w; p = -0.00125372503 + p*w;
  p = -0.00417768164 + p*w; p = 0.246640727 + p*w; p = 1.50140941 + p*w
else:      w = sqrt(w) - 3
  p = -0.000200214257; p = 0.000100950558 + p*w; p = 0.00134934322 + p*w;
  p = -0.00367342844 + p*w; p = 0.00573950773 + p*w; p = -0.0076224613 + p*w;
  p = 0.00943887047 + p*w; p = 1.00167406 + p*w; p = 2.83297682 + p*w
erfinv ≈ p * x
```

**Emit** (`binop`): `Bit_and/Bit_or/Bit_xor` → `(a & b)`, `(a | b)`, `(a ^ b)`.
Shifts go through unsigned so no signed overflow is undefined:
`I32`: `((int)((unsigned int)(a) << ((b) & 31)))` and
`((int)((unsigned int)(a) >> ((b) & 31)))`; `I64`: `unsigned long long`,
`& 63`. Floats: `failwith` like `sqrt` on an int does today.
**Emit** (`unop`): `sinf/sin`, `cosf/cos`, `erff/erf`, `erfinvf/erfinv`
via the existing `math` helper.

**Dsl**: `ne/gt/ge` through `cmp`; `and_/or_` build `Expr.Logic`; `not_`
builds `Expr.Not`. Note `sin`, `cos`, `exp`, `log` shadow `Stdlib` under
`open Dsl`; that is already the case for `exp`/`log`/`sqrt`.

## Invariants

- Integer results are identical on the interpreter and the device, bit for
  bit. T16's system test relies on it.
- Shift by a count ≥ width equals shift by `count mod width` on both sides.
- `Erfinv` is odd: `erfinv (-x) = - erfinv x`.

## Failure modes to avoid

- `Int32.shift_right` (arithmetic) where `shift_right_logical` is required.
- Emitting `a >> b` on a signed `int`: implementation-defined for negatives.
- `Erf` spelled `erf(` for f32: that is the double version and silently
  doubles the cost; use `erff`.
- Adding the new unops to `Lower`'s `expr_of` by name: it forwards `op`
  and needs no change.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_expr_ops.exe
```

## Tests (already written: `test/staged/unit/test_expr_ops.ml`)

- I32 `bit_xor/and/or` on `0x0F0F0F0F`, `0x00FF00FF` and on negatives
- `shl 1 31 = min_int`; `shr (-1) 28 = 15` (logical); `shr x 32 = x`; `shl x 33 = shl x 1`
- I64: `shr (-1L) 60 = 15L`; `shl 1L 63 = min_int`
- floats: `bit_and` raises `Invalid_argument` on the interpreter
- `sin`, `cos` at 0 and π/2 (F64, tol 1e-12); `erf 0 = 0`, `erf 1 = 0.8427007929497149`
- `erfinv (erf x) = x` for x ∈ {−2, −0.5, 0, 0.3, 1.5, 2.5} within 1e-12 (F64); `erfinv 1 = inf`, `erfinv 0 = 0`, `erfinv (−1) = −inf`
- F32 `erfinv 0.5 = 0.4769362762` within 1e-6
- `ne/gt/ge/and_/or_/not_` on a small vector, checked through `select` to 1.0/0.0
- emit: the generated source for an I32 shift contains `unsigned int` and `& 31`; for I64 `unsigned long long`; an F32 `erfinv` contains `erfinvf(`; an F64 one contains `erfinv(` and not `erfinvf(`; `sin` on F32 prints `sinf(`
