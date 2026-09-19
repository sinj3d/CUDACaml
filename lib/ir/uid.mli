(** Unique node identifiers. Every IR node carries one; hash-consing,
    topological sorting and name mangling all key on it. *)

type t

val fresh : unit -> t
val to_int : t -> int
val compare : t -> t -> int
val equal : t -> t -> bool
val to_string : t -> string
