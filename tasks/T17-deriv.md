# T17 — `Deriv`: symbolic derivatives of element functions

## Goal

The scalar half of reverse-mode AD. Given an `Expr.t` over `Arg`
placeholders, produce the `Expr.t` of its partial derivative with respect
to one `Arg`, over the same placeholders. Plus the two utilities T18 needs:
substituting an expression for an `Arg` (`apply1`/`apply2`), and an
algebraic simplifier so the adjoint kernels are not full of `x * 0`.

A new library `ocaml_cuda.ad` at layer 2 (depends on `ir` only; the
interpreter must stay independent of it).

Depends on: T15 (the new unops need rules). Phase 3.

## Files you own

- `lib/ad/dune`, `lib/ad/deriv.ml`, `lib/ad/deriv.mli` (new)
- `lib/ocaml_cuda.ml`, `lib/dune` (add `module Deriv = Ocaml_cuda_ad.Deriv` under "Layer 2")
- `test/staged/unit/test_deriv.ml` → promote

```
(library
 (name ocaml_cuda_ad)
 (public_name ocaml_cuda.ad)
 (libraries ocaml_cuda_ir))
```

## Interfaces (`deriv.mli`)

```ocaml
open Ocaml_cuda_ir

(** [d e ~wrt] = ∂e/∂(Arg wrt), as an expression over the same Args.
    [e] must have a float dtype ([Dtype.is_float]); [Invalid_argument]
    otherwise. Result dtype = dtype of [e]. The result is passed through
    [simplify]. *)
val d : 'a Expr.t -> wrt:int -> 'a Expr.t

(** Derivative of a unary element function with respect to its argument. *)
val fn1 : ('a, 'a) Expr.fn1 -> ('a, 'a) Expr.fn1

(** Partial derivative of a binary element function with respect to
    argument 0 or 1 ([Invalid_argument] otherwise). *)
val fn2 : ('a, 'a, 'a) Expr.fn2 -> wrt:int -> ('a, 'a, 'a) Expr.fn2

(** Substitution: [apply1 f x] is [f.body1] with every [Arg 0] replaced by
    [x]. [Invalid_argument] if the dtype of [x] differs from [f.arg1]'s.
    Fresh uids on every rebuilt node. *)
val apply1 : ('a, 'b) Expr.fn1 -> 'a Expr.t -> 'b Expr.t

val apply2 : ('a, 'b, 'c) Expr.fn2 -> 'a Expr.t -> 'b Expr.t -> 'c Expr.t

(** Semantics-preserving rewrites: constant folding of float arithmetic;
    [x*0 → 0], [0*x → 0], [x*1 → x], [x+0 → x], [x-0 → x], [0-x → -x],
    [Neg (Neg x) → x], [Select (c, a, a) → a], [Cast (x, d)] when [x] already
    has dtype [d]. Applied bottom-up, once. *)
val simplify : 'a Expr.t -> 'a Expr.t
```

## Rules

Let `da = d a ~wrt`, `db = d b ~wrt`, `one = Const 1`, `zero = Const 0`
at the dtype of `e`. Use `Dsl` to build nodes (`Dsl.mul`, `Dsl.select`, …).

| node | derivative |
|---|---|
| `Const _` | `zero` |
| `Arg j` | `one` if `j = wrt` else `zero` |
| `Index` | cannot occur (I32); unreachable once the float check passes at the root — but `Cast (Index, F32)` can: see `Cast` |
| `Add` | `da + db` |
| `Sub` | `da - db` |
| `Mul` | `da*b + a*db` |
| `Div` | `(da*b - a*db) / (b*b)` |
| `Min (a,b)` | `select (le a b) da db` |
| `Max (a,b)` | `select (ge a b) da db` |
| `Bit_*`, `Shl`, `Shr` | unreachable on floats; `invalid_arg` |
| `Neg` | `neg da` |
| `Sqrt` | `da / (2 * sqrt a)` |
| `Exp` | `exp a * da` |
| `Log` | `da / a` |
| `Abs` | `select (lt a 0) (neg da) da` |
| `Sin` | `cos a * da` |
| `Cos` | `neg (sin a) * da` |
| `Erf` | `(2/√π) * exp (neg (a*a)) * da` |
| `Erfinv` | `(√π/2) * exp (erfinv a * erfinv a) * da` |
| `Select (c, a, b)` | `select c da db` (the condition is not differentiated) |
| `Cast (x, dt)` | if `x`'s dtype is float: `cast dt (d x ~wrt)`; else `zero` |

`Cmp`, `Logic`, `Not` have dtype `Bool` and never reach `d` (the root check
rejects them; inside `Select` they are conditions, not differentiated).

Kink conventions (`Min`, `Max`, `Abs`, `Select`): the derivative is that of
the branch chosen at the point, with ties going to the first operand. This
is a subgradient and is what pathwise Greeks use.

## Implementation notes

- `d` is polymorphic-recursive like `Backend_interp.eval`: `Cast` recurses
  at a different type. Pattern: `let rec d : type a. a Expr.t -> wrt:int -> a Expr.t`.
- Checking "is float" for a `'a Expr.t`: `Dtype.is_float e.dtype`. Building
  `Const 1` at type `a` requires knowing `a = float`: match on `e.dtype`
  with `Dtype.F32 | Dtype.F64 -> Dsl.const e.dtype 1.0` and `invalid_arg`
  in the other arms.
- `apply1`: walk the tree; at `Arg 0` compare `Dtype.equal` between the
  Arg's dtype and `x`'s; on `Some Equal` return `x`, else `invalid_arg`.
  Every other node is rebuilt with `Expr.make` (fresh uid) so the result
  shares no uids with the input; `Arg i` for `i ≠ 0` is kept.
- `fn1 f = { arg1 = f.arg1; body1 = d f.body1 ~wrt:0 }`. `fn2` likewise.
- `simplify` constant-folds only when **both** operands are `Const` of a
  float dtype, using the interpreter's rounding rule for F32 (`Int32.float_of_bits (Int32.bits_of_float x)`); copy those two lines, do not depend on the interpreter.

## Failure modes to avoid

- Sharing uids between the input and output of `apply1`: `Graph` and
  `Lower` key on uids; a shared uid across two different trees is a silent
  miscompile in T18.
- Simplifying `x * 0 → 0` when `x` may be NaN or infinite: accepted, and
  documented; AD frameworks do the same. Do not "fix" by keeping the
  product.
- Folding `Div` by a zero constant: leave it alone (`x / 0` must stay `inf`/`nan`).
- Making `d` succeed on an I32 expression by returning 0: the contract is
  `Invalid_argument`, and T18 relies on it to detect integer paths.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_deriv.exe
```

## Tests (already written: `test/staged/unit/test_deriv.ml`)

Every derivative is checked numerically against a central difference on
the interpreter in F64 (h = 1e-6, tolerance 1e-6 relative + absolute) at
several points, by building `Map (fn1 f) x` with `Tensor.make` directly:

- polynomial `3x² − 2x + 1`; `x / (1 + x²)`; `sqrt`, `exp`, `log`, `sin`,
  `cos`, `erf`, `erfinv` (on (−0.9, 0.9)); `abs` away from 0; `min/max`
  against a constant away from the tie; `select (lt x 0) (x²) (exp x)`
- `fn2`: `∂(x·y + sin x)/∂x` and `/∂y`
- `Cast` of an int: `d (cast F64 (index ()) * x) ~wrt:0` equals `cast F64 index`
- `d` on an I32 expression raises `Invalid_argument`
- structural: `d (const 5.0)` is `Const 0.`; `d (Arg 0)` is `Const 1.`; `d (Arg 1) ~wrt:0` is `Const 0.`
- `apply1` substitutes and the result contains no `Arg 0`; dtype mismatch raises
- `simplify (x*0 + y*1)` is `y` (an `Arg 1` node); `simplify (Neg (Neg x))` is `x`; constants fold: `simplify (2+3)` is `Const 5.`; `Div` by constant 0 is not folded
- uids: `apply1 f x` shares no uid with `f.body1`
