(** [Kernel_ir.program] to CUDA C++ source, one translation unit.

    A pure pretty-printer: no decisions, no optimisation. If a choice has to
    be made here, it belongs in [Lower] or [Schedule] instead. *)

open Ocaml_cuda_lower

val program : Kernel_ir.program -> string

(** Pieces, exposed for unit tests. *)
val expr : Kernel_ir.expr -> string

val stmt : indent:int -> Kernel_ir.stmt -> string
val kernel : Kernel_ir.kernel -> string
