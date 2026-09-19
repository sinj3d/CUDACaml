(** The user-facing surface. Everything a program author writes goes
    through here; nothing here does work beyond building IR nodes.

    A future [%kernel ...] ppx would desugar to exactly these calls. *)

(** {1 Tensors} *)

val param : string -> 'a Dtype.t -> Shape.t -> 'a Tensor.t
val iota : Shape.t -> int32 Tensor.t
val map : ('a Expr.t -> 'b Expr.t) -> 'a Tensor.t -> 'b Tensor.t

val map2 :
  ('a Expr.t -> 'b Expr.t -> 'c Expr.t) -> 'a Tensor.t -> 'b Tensor.t -> 'c Tensor.t

(** Whole-tensor reduction to a scalar. [init] must be the identity of the
    operator. Rank >= 2 sources are flattened with [reshape] first; the
    result is [Shape.scalar] as before. *)
val reduce :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Whole-tensor inclusive scan in flat index order; shape preserved. Rank
    >= 2 sources are flattened and reshaped back. *)
val scan :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Reduce every row (the last axis), each row independently:
    [[d0;..;dk-1;n]] -> [[d0;..;dk-1]]. Rank >= 2 required, else
    [Invalid_argument]. Other axes are reached through [transpose]. *)
val reduce_rows :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Inclusive scan of every row (the last axis), each row restarting from
    [init]. Shape preserved. Rank >= 2 required, else [Invalid_argument]. *)
val scan_rows :
  ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

val gather : int32 Tensor.t -> 'a Tensor.t -> 'a Tensor.t
val reshape : Shape.t -> 'a Tensor.t -> 'a Tensor.t

(** Rank-2 only, else [Invalid_argument]. [transpose x] for [x : [m; n]] has
    shape [[n; m]] and is a [gather] over an [iota] index tensor
    ([out[i*m + j] = x[j*n + i]]), so it fuses into its consumer and never
    owns a buffer unless fan-out forces it. *)
val transpose : 'a Tensor.t -> 'a Tensor.t

(** [broadcast shape s]: [s] must have exactly one element, otherwise
    [Invalid_argument]. Result has [shape]. *)
val broadcast : Shape.t -> 'a Tensor.t -> 'a Tensor.t

(** [full dtype shape v]: every element is [v]. No Param, no buffer: it is
    [map (fun _ -> const dtype v) (iota shape)] and is always inlined. *)
val full : 'a Dtype.t -> Shape.t -> 'a -> 'a Tensor.t

(** [scalar name dtype] = [param name dtype Shape.scalar]. *)
val scalar : string -> 'a Dtype.t -> 'a Tensor.t

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
