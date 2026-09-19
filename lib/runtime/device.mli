(** Device discovery and context lifetime. *)

(** false on machines with no CUDA driver *)
val available : unit -> bool

(** cuInit + primary context; idempotent *)
val init : unit -> unit

val synchronize : unit -> unit
val name : unit -> string
