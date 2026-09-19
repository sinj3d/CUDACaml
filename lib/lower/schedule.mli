(** Launch geometry and per-kernel scheduling decisions.

    Kept apart from [Kernel_ir] to make the algorithm/schedule split
    (Halide) visible: a lowered kernel can be re-scheduled without being
    re-lowered. *)

type launch = { grid : int; block : int; shared_bytes : int }

(** Threads per block for every kernel in v1. Reduction kernels also size
    their [__shared__] scratch by this. *)
val block_size : int

(** Element-wise kernels: a 1-D grid-stride loop, [block = block_size],
    [grid = clamp 1 1024 (ceil (numel / block_size))]. Correct for any numel,
    including 0. *)
val grid_stride : numel:int -> launch

(** Reduction kernels: one block of [block_size] threads. *)
val single_block : launch

(** Scan kernels (v1 sequential): one thread. *)
val single_thread : launch
