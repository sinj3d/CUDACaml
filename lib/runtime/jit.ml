(* Runtime compilation: CUDA C++ -> PTX (NVRTC) -> loaded module (driver). *)

type module_ = { m : Cuda.Module.t; ptx : string }
type kernel = Cuda.Module.func

let log_of prog =
  match Nvrtc.compilation_log prog with
  | Some log -> log
  | None -> "no compilation log"

let compile ~name ~source =
  Device.init ();
  (* Never swallow the NVRTC log: the first real bug in generated CUDA is
     undiagnosable without it. cudajit already folds the program log into the
     Nvrtc_error message when compilation itself fails. *)
  let prog =
    try Nvrtc.compile_to_ptx ~cu_src:source ~name:(name ^ ".cu") ~options:[] ~with_debug:false
    with Nvrtc.Nvrtc_error { message; _ } -> failwith ("nvrtc: " ^ message)
  in
  let m =
    try Cuda.Module.load_data_ex prog []
    with Cuda.Cuda_error { message; _ } ->
      failwith ("nvrtc: " ^ message ^ ": " ^ log_of prog)
  in
  { m; ptx = Nvrtc.string_from_ptx prog }

let ptx m = m.ptx

(* Raises when the name is absent, which is the contract callers rely on. *)
let get_kernel m name = Cuda.Module.get_function m.m ~name

(* This cudajit version exposes no cuModuleUnload, so the module stays loaded
   for the lifetime of the context. *)
let unload _ = ()

let unsafe_func (k : kernel) = k
