type t = { bytes : int }

let alloc ~bytes:_ = failwith "Buffer.alloc: TODO (cudajit)"
let free _ = failwith "Buffer.free: TODO (cudajit)"
let byte_size t = t.bytes
let upload _ _ = failwith "Buffer.upload: TODO (cudajit)"
let download _ _ = failwith "Buffer.download: TODO (cudajit)"
