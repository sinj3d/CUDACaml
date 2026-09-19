(** Runs a [Kernel_ir.host_op] plan against a loaded [Jit.module_] using
    the runtime layer. Owns buffer lifetimes for the duration of one run. *)

open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime

val run :
  Kernel_ir.program ->
  Jit.module_ ->
  inputs:(string * Value.packed) list ->
  (string * Value.packed) list
