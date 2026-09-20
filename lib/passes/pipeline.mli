(** Ordered composition of passes. The only entry point backends call. *)

open Cudacaml_ir

val default : (module Pass.S) list
val run : ?passes:(module Pass.S) list -> Graph.t -> Graph.t
