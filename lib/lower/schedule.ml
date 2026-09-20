type launch = { grid : int; block : int; shared_bytes : int }

let block_size = 256

(* One block per [block_size] elements, clamped to [1, 1024]: the loop in
   the kernel is grid-stride, so a grid that is too small is still correct
   (each thread just takes more elements) and a numel of 0 still needs a
   legal launch geometry. The 1024 cap keeps the grid small enough that the
   whole grid is resident, which is what makes the stride loop worthwhile. *)
let grid_stride ~numel =
  { grid = max 1 (min 1024 ((numel + 255) / 256)); block = 256; shared_bytes = 0 }

(* Row kernels. [Reduce] and [Scan] act along the LAST axis, so the row
   index is the block index and the grid is exactly the row count: a kernel
   never has to work out which row it owns from a global thread id. A rank-1
   source is the degenerate one-row case, which is why [single_block] and
   [single_thread] are these two at [rows = 1] rather than separate
   geometries. *)
let rows_block ~rows = { grid = max 1 rows; block = block_size; shared_bytes = 0 }
let rows_thread ~rows = { grid = max 1 rows; block = 1; shared_bytes = 0 }
let single_block = rows_block ~rows:1
let single_thread = rows_thread ~rows:1

(* ---------------------------------------------------------------- *)
(* T21: multi-kernel reduce and scan                                 *)
(* ---------------------------------------------------------------- *)

(* Below this row length a single block already has enough elements to
   keep its 256 threads busy, and a second kernel launch would cost more
   than the extra parallelism buys. Above it the row is split across
   several blocks whose partial results a second kernel folds. 4096 is
   16 elements per thread: the point at which the launch overhead of the
   second kernel is amortised. *)
let reduce_threshold = 4096

(* Each block walks at least [block_size * 8] elements, so the grid grows
   eight times more slowly than the row: fewer, fatter blocks mean a
   smaller partials buffer and a cheaper second pass. The 128 cap bounds
   that buffer at [rows * 128] and keeps the second kernel's fold inside
   one [block_size] pass ([128 <= block_size]), which is what lets it use
   the same shared-memory tree unchanged. *)
let reduce_blocks ~row_len =
  if row_len <= reduce_threshold then 1
  else
    let per_block = block_size * 8 in
    min 128 ((row_len + per_block - 1) / per_block)

(* One chunk per block, one element per thread: the Hillis-Steele scan in
   shared memory is indexed by [threadIdx.x], so the chunk size and the
   block size are necessarily the same number. *)
let scan_chunk = block_size

(* One kernel while the row fits in a single chunk; otherwise the classic
   three passes (scan each chunk, scan the chunk totals, add the running
   total back in). *)
let scan_kernels ~row_len = if row_len <= scan_chunk then 1 else 3
