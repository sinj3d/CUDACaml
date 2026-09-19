(** Runtime compilation: CUDA C++ source to a loaded module.

    Pipeline: nvrtcCompileProgram -> PTX -> cuModuleLoadData. The PTX is
    kept for [--dump-ptx] style debugging. *)

type module_
type kernel

val compile : name:string -> source:string -> module_
val ptx : module_ -> string
val get_kernel : module_ -> string -> kernel
val unload : module_ -> unit

(* The raw driver handle, for Launch. Mirrors Buffer.unsafe_ptr: the runtime
   layer is the cudajit wrapper, so exposing the handle here is the intended
   seam rather than a leak. *)
val unsafe_func : kernel -> Cuda.Module.func
