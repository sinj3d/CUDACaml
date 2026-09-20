(** Kernel launch. Arguments are device buffers only; scalars are baked
    into the source by [Emit] in v1.

    [grid_y] and [block_y] default to 1, so a 1-D launch is written exactly
    as before; only the tiled matmul passes them. *)
val run :
  Jit.kernel ->
  grid:int ->
  ?grid_y:int ->
  block:int ->
  ?block_y:int ->
  shared_bytes:int ->
  Buffer.t list ->
  unit
