type launch = { grid : int; block : int; shared_bytes : int }

let block_size = 256
let grid_stride ~numel:_ = failwith "Schedule.grid_stride: TODO"
let single_block = { grid = 1; block = block_size; shared_bytes = 0 }
let single_thread = { grid = 1; block = 1; shared_bytes = 0 }
