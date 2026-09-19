(** The CUDA backend: [Pipeline] -> [Lower] -> [Emit] -> [Jit], then
    [Executor] per run. *)

open Ocaml_cuda_ir

include Ocaml_cuda_backend.Backend.S

(** The generated CUDA C++ for a graph, without compiling it. What the CLI
    prints and what golden tests snapshot. *)
val source : Graph.t -> string

(** Internals, exposed for tests and tooling. *)
module Emit = Emit

module Mangle = Mangle
module Executor = Executor
