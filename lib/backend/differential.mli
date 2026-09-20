(** Run the same graph on two backends and compare outputs elementwise.
    The correctness harness for every pass and every backend. *)

open Cudacaml_ir

val check :
  ?tolerance:float ->
  reference:(module Backend.S) ->
  candidate:(module Backend.S) ->
  Graph.t ->
  inputs:(string * Value.packed) list ->
  (unit, string) result
