(** Array-level IR: a DAG of whole-tensor operations.

    Parallelism is expressed ONLY through these combinators. The compiler
    never proves a loop parallel; each node has known parallel semantics by
    construction (the Futhark lesson). Data-dependent control flow lives
    inside element functions as [Expr.Select], never at this level.

    Hardcaml analogue: [Signal.Type.t], a small closed set of variants that
    every downstream pass matches exhaustively. *)

type 'a t = { uid : Uid.t; dtype : 'a Dtype.t; shape : Shape.t; node : 'a node }

and _ node =
  | Param : string -> 'a node  (** program input; a [Value.t] at run time *)
  | Iota : int32 node  (** [0, 1, ..., numel-1] *)
  | Map : ('a, 'b) Expr.fn1 * 'a t -> 'b node
  | Map2 : ('a, 'b, 'c) Expr.fn2 * 'a t * 'b t -> 'c node
  | Reduce : ('a, 'a, 'a) Expr.fn2 * 'a Expr.t * 'a t -> 'a node
      (** associative op, identity, source. Result has [Shape.scalar]. *)
  | Scan : ('a, 'a, 'a) Expr.fn2 * 'a Expr.t * 'a t -> 'a node
      (** inclusive prefix scan; same shape as source *)
  | Gather : int32 t * 'a t -> 'a node  (** out[i] = src[idx[i]] *)
  | Reshape : Shape.t * 'a t -> 'a node  (** metadata only; numel preserved *)

type packed = P : _ t -> packed

let make dtype shape node = { uid = Uid.fresh (); dtype; shape; node }
let uid (P t) = t.uid
let shape (P t) = t.shape

(** Direct data dependencies, in argument order. The one traversal
    primitive every pass is built on. *)
let deps (P t) : packed list =
  match t.node with
  | Param _ | Iota -> []
  | Map (_, a) -> [ P a ]
  | Map2 (_, a, b) -> [ P a; P b ]
  | Reduce (_, _, a) -> [ P a ]
  | Scan (_, _, a) -> [ P a ]
  | Gather (i, a) -> [ P i; P a ]
  | Reshape (_, a) -> [ P a ]
