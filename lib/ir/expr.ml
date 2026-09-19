(** Scalar expressions: the element-level language that runs inside a kernel.

    An [Expr.t] computes ONE output element from the corresponding input
    elements. It is first-order and closed: its only free variables are
    [Arg] placeholders, which [Tensor] combinators bind. This is what keeps
    the IR inspectable: a user-written OCaml closure is applied to [Arg]
    nodes exactly once, at graph-construction time, and is never stored. *)

(* [Bit_and] .. [Shr] are integer only: both the interpreter and [Emit]
   reject them on a float dtype. The shift count is the second operand, of
   the same dtype, and is taken modulo the bit width on both sides, so no
   count is undefined. [Shr] is LOGICAL. *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Min
  | Max
  | Bit_and
  | Bit_or
  | Bit_xor
  | Shl
  | Shr

(* [Sqrt] .. [Erfinv] are float only. *)
type unop = Neg | Sqrt | Exp | Log | Abs | Sin | Cos | Erf | Erfinv
type cmp = Eq | Ne | Lt | Le | Gt | Ge
type logic = And | Or

type 'a t = { uid : Uid.t; dtype : 'a Dtype.t; node : 'a node }

and _ node =
  | Const : 'a -> 'a node
  | Arg : int -> 'a node  (** i-th argument of the enclosing element function *)
  | Index : int32 node  (** flat index of the element being computed *)
  | Binop : binop * 'a t * 'a t -> 'a node
  | Unop : unop * 'a t -> 'a node
  | Cmp : cmp * 'b t * 'b t -> bool node
  | Logic : logic * bool t * bool t -> bool node
  | Not : bool t -> bool node
  | Select : bool t * 'a t * 'a t -> 'a node  (** a divergence point on device *)
  | Cast : 'b t * 'a Dtype.t -> 'a node

type packed = P : _ t -> packed

(** Element functions in first-order form. The [arg] fields are the [Arg]
    placeholders the body refers to. *)
type ('a, 'b) fn1 = { arg1 : 'a t; body1 : 'b t }

type ('a, 'b, 'c) fn2 = { arg_a : 'a t; arg_b : 'b t; body2 : 'c t }

let make dtype node = { uid = Uid.fresh (); dtype; node }

(** The HOAS -> first-order bridge: run the closure once on fresh
    placeholders and keep only the resulting tree. *)
let fn1 dtype f =
  let arg1 = make dtype (Arg 0) in
  { arg1; body1 = f arg1 }

let fn2 da db f =
  let arg_a = make da (Arg 0) in
  let arg_b = make db (Arg 1) in
  { arg_a; arg_b; body2 = f arg_a arg_b }
