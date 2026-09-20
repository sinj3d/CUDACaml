(** Reverse-mode automatic differentiation over the tensor DAG.

    [grad] is a graph-to-graph transform: it walks [Graph.topological_order]
    backwards, accumulates one adjoint tensor per node, and returns a new
    graph that computes everything the original did plus the requested
    gradients. Every adjoint is built from [Dsl] calls only, so no new node
    kind appears and both backends run the result unchanged. *)

open Ocaml_cuda_ir

exception Not_differentiable of string
(** A node on a path from a [wrt] param to [output] has no adjoint rule.
    The message names the node kind and, for Reduce/Scan, the operator. *)

(** ["d" ^ output ^ "/d" ^ wrt]. *)
val grad_name : output:string -> wrt:string -> string

(** [grad g ~output ~wrt] : a graph named [Graph.name g ^ "_grad"] whose
    outputs are those of [g], in order, followed by
    [grad_name ~output ~wrt:p] for each [p] in [wrt] (in order), each with
    the dtype and shape of param [p], holding ∂(Σ output)/∂p.

    [output] must name an output of [g] with a float dtype; if it is not a
    scalar its elements are summed (seed adjoint = ones).
    A [wrt] param that [output] does not depend on gets an all-zero gradient.
    [Invalid_argument] for an unknown output or param name, or a non-float
    output. *)
val grad : Graph.t -> output:string -> wrt:string list -> Graph.t
