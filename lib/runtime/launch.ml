(* Kernel launch. Asynchronous by contract: callers synchronize before
   reading results, so nothing here waits on the device. *)

let run k ~grid ~block ~shared_bytes bufs =
  Cuda.Stream.launch_kernel (Jit.unsafe_func k) ~grid_dim_x:grid ~block_dim_x:block
    ~shared_mem_bytes:shared_bytes Cuda.Stream.no_stream
    (List.map (fun b -> Cuda.Stream.Tensor (Buffer.unsafe_ptr b)) bufs)
