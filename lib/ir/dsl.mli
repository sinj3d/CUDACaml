(** The user-facing surface. Everything a program author writes goes
    through here; nothing here does work beyond building IR nodes.

    A future [%kernel ...] ppx would desugar to exactly these calls. *)

(** {1 Tensors} *)

val param : string -> 'a Dtype.t -> Shape.t -> 'a Tensor.t
val iota : Shape.t -> int32 Tensor.t
val map : ('a Expr.t -> 'b Expr.t) -> 'a Tensor.t -> 'b Tensor.t

val map2 :
  ('a Expr.t -> 'b Expr.t -> 'c Expr.t) -> 'a Tensor.t -> 'b Tensor.t -> 'c Tensor.t

val reduce :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

val scan :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

val gather : int32 Tensor.t -> 'a Tensor.t -> 'a Tensor.t
val reshape : Shape.t -> 'a Tensor.t -> 'a Tensor.t

(** {1 Scalars, inside element functions} *)

val const : 'a Dtype.t -> 'a -> 'a Expr.t
val index : unit -> int32 Expr.t
val add : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val sub : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val mul : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val div : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val min : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val max : 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val neg : 'a Expr.t -> 'a Expr.t
val sqrt : 'a Expr.t -> 'a Expr.t
val exp : 'a Expr.t -> 'a Expr.t
val log : 'a Expr.t -> 'a Expr.t
val lt : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val le : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val eq : 'a Expr.t -> 'a Expr.t -> bool Expr.t
val select : bool Expr.t -> 'a Expr.t -> 'a Expr.t -> 'a Expr.t
val cast : 'a Dtype.t -> _ Expr.t -> 'a Expr.t
