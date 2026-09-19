(** Kernel launch. Arguments are device buffers only; scalars are baked
    into the source by [Emit] in v1. *)

val run :
  Jit.kernel -> grid:int -> block:int -> shared_bytes:int -> Buffer.t list -> unit
