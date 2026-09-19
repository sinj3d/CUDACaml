type launch = { grid : int; block : int; shared_bytes : int }

let block_size = 256

(* One block per [block_size] elements, clamped to [1, 1024]: the loop in
   the kernel is grid-stride, so a grid that is too small is still correct
   (each thread just takes more elements) and a numel of 0 still needs a
   legal launch geometry. The 1024 cap keeps the grid small enough that the
   whole grid is resident, which is what makes the stride loop worthwhile. *)
let grid_stride ~numel =
  { grid = max 1 (min 1024 ((numel + 255) / 256)); block = 256; shared_bytes = 0 }

let single_block = { grid = 1; block = block_size; shared_bytes = 0 }
let single_thread = { grid = 1; block = 1; shared_bytes = 0 }
