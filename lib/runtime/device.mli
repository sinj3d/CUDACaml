(** Device discovery and context lifetime. *)

val available : unit -> bool  (** false on machines with no CUDA driver *)
val init : unit -> unit  (** cuInit + primary context; idempotent *)
val synchronize : unit -> unit
val name : unit -> string
