type module_ = unit
type kernel = unit

let compile ~name:_ ~source:_ = failwith "Jit.compile: TODO (cudajit/NVRTC)"
let ptx _ = failwith "Jit.ptx: TODO"
let get_kernel _ _ = failwith "Jit.get_kernel: TODO"
let unload _ = failwith "Jit.unload: TODO"
