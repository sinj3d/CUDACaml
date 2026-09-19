(** Identifier legalisation for the C++ target: reserved words, illegal
    characters, uniqueness within a translation unit.

    Hardcaml analogue: [Rtl_name] / [Mangler]. *)

type t

val create : unit -> t
val identifier : t -> string -> string  (** stable: same input, same output *)
