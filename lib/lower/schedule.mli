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

(** One thread per row, [grid = rows], [block = 1]. Used by the scan of
    the per-chunk totals, which is sequential along a row by construction. *)
val rows_thread : rows:int -> launch

(** Reduction kernels over a single row: [rows_block ~rows:1]. *)
val single_block : launch

(** Scan kernels over a single row: [rows_thread ~rows:1]. *)
val single_thread : launch

(** {1 Multi-kernel reduce and scan (T21)} *)

(** Row length at or below which a [Reduce] stays a single kernel: 4096. *)
val reduce_threshold : int

(** How many blocks co-operate on one row of a [Reduce]. [1] when
    [row_len <= reduce_threshold], so small reductions keep the v1
    single-kernel form; otherwise
    [min 128 (ceil (row_len / (block_size * 8)))], and the reduction
    becomes a partials kernel followed by a fold of the partials. *)
val reduce_blocks : row_len:int -> int

(** Elements per block-scan chunk; equals [block_size], because the
    Hillis-Steele scan is indexed by [threadIdx.x]. *)
val scan_chunk : int

(** Number of kernels a [Scan] over rows of [row_len] lowers to: [1] when
    [row_len <= scan_chunk], else [3]. *)
val scan_kernels : row_len:int -> int
