(** The executor signature.

    Every way of *running* a [Graph.t] implements this: the CUDA JIT, the
    pure-OCaml reference evaluator, and any future PTX or C backend. Tests,
    benchmarks and the CLI are written against [S] only.

    Hardcaml analogue: [Cyclesim.t] as produced by both the built-in
    simulator and out-of-tree [Hardcaml_c] / [Hardcaml_verilator]. One IR,
    many executors, and [Differential] to keep them honest. *)

open Ocaml_cuda_ir

module type S = sig
  val name : string

  type compiled

  (** All the expensive work: passes, lowering, JIT. Done once per graph. *)
  val compile : Graph.t -> compiled

  (** Inputs are keyed by [Tensor.Param] name; outputs by [Graph.outputs]
      name. Split from [compile] so benchmarks time only this. *)
  val run : compiled -> inputs:(string * Value.packed) list -> (string * Value.packed) list
end
