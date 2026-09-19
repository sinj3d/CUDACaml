(** Tensor shapes: static rank, extents fixed at graph-construction time.

    Shape polymorphism is deliberately out of scope for v1: a [Graph.t] is
    built for concrete sizes and rebuilt if they change. Static extents are
    what let [Lower] pick launch geometry with no runtime feedback. *)

type t

val scalar : t
val of_dims : int list -> t
val dims : t -> int list
val rank : t -> int
val numel : t -> int
val equal : t -> t -> bool
val to_string : t -> string
