(** A graph-to-graph rewrite.

    Passes are pure: same [Graph.t] in, new [Graph.t] out; nodes are never
    mutated. Every pass must preserve program semantics as observed by the
    reference interpreter; [Differential] is how that is checked. *)

open Ocaml_cuda_ir

module type S = sig
  val name : string
  val run : Graph.t -> Graph.t
end
