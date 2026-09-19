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
