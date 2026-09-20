(** The CUDA backend: [Pipeline] -> [Lower] -> [Emit] -> [Jit], then
    [Executor] per run. *)

open Ocaml_cuda_ir

include Ocaml_cuda_backend.Backend.S

(** The generated CUDA C++ for a graph, without compiling it. What the CLI
    prints and what golden tests snapshot. *)
val source : Graph.t -> string

(** Frees the device buffers a [compile] allocated. Idempotent; [run]
    after it raises. A dropped [compiled] is released by a finaliser, but
    do not rely on that. *)
val release : compiled -> unit

(** The persistent executor behind a [compiled]. For tests and tooling. *)
val executor : compiled -> Executor.t

(** Internals, exposed for tests and tooling. *)
module Emit = Emit

module Mangle = Mangle
module Executor = Executor
