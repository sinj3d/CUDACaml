(** Symbolic derivatives of element functions: the scalar half of
    reverse-mode AD.

    Everything here is a pure [Expr.t -> Expr.t] rewrite over [Arg]
    placeholders. Nothing in this module knows about tensors, graphs or
    devices, and the interpreter does not depend on it. *)

open Cudacaml_ir

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
