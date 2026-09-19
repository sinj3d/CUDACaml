(** Runtime compilation: CUDA C++ source to a loaded module.

    Pipeline: nvrtcCompileProgram -> PTX -> cuModuleLoadData. The PTX is
    kept for [--dump-ptx] style debugging. *)

type module_
type kernel

val compile : name:string -> source:string -> module_
val ptx : module_ -> string
val get_kernel : module_ -> string -> kernel
val unload : module_ -> unit
