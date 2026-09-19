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

(** Reduction kernels: one block of [block_size] threads PER ROW, so
    [grid = rows] and [blockIdx.x] IS the row index. [rows] is clamped up to
    1 so that a zero-row launch is still legal. *)
val rows_block : rows:int -> launch

(** Scan kernels (sequential per row until T21): one thread per row,
    [grid = rows], [block = 1]. *)
val rows_thread : rows:int -> launch

(** Reduction kernels over a single row: [rows_block ~rows:1]. *)
val single_block : launch

(** Scan kernels over a single row: [rows_thread ~rows:1]. *)
val single_thread : launch
