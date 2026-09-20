(* Kernel launch. Asynchronous by contract: callers synchronize before
   reading results, so nothing here waits on the device. *)

(* [grid_y] and [block_y] default to 1 -- CUDA's own default -- so every
   1-D caller launches exactly the geometry it did before. The z dimension
   is not exposed: no kernel here uses it.

   [stream] defaults to the NULL stream, which is what v1 launched on. *)
let run ?(stream = Stream.default) k ~grid ?(grid_y = 1) ~block ?(block_y = 1) ~shared_bytes bufs =
  Cuda.Stream.launch_kernel (Jit.unsafe_func k) ~grid_dim_x:grid ~grid_dim_y:grid_y
    ~block_dim_x:block ~block_dim_y:block_y ~shared_mem_bytes:shared_bytes
    (Stream.unsafe_stream stream)
    (List.map (fun b -> Cuda.Stream.Tensor (Buffer.unsafe_ptr b)) bufs)
