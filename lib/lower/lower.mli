(** [Graph.t] to [Kernel_ir.program].

    Expects a graph that has already been through [Pipeline]: one kernel is
    emitted per fused group, each with a bounds-guarded grid-stride loop.
    [Reduce] and [Scan] lower to their own kernels. Also produces the host
    plan: allocate, upload params, launch in topological order, download
    outputs, free. *)

open Ocaml_cuda_ir

val program : Graph.t -> Kernel_ir.program
