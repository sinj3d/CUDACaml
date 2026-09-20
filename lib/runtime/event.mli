(** CUDA events: a marker in a stream that other streams can wait on and
    that the host can block on or time against. *)

type t

(** Creates a timing-enabled event and records it in [s]. The event
    completes when every operation issued on [s] before this call has
    completed. *)
val record : Stream.t -> t

(** [wait s e]: work issued on [s] after this call does not start until
    [e] has completed. Does not block the host. *)
val wait : Stream.t -> t -> unit

(** Blocks the calling thread until [t] has completed. *)
val synchronize : t -> unit

(** Device time between two completed events, in milliseconds. Both
    events must have completed; [synchronize] the later one first. *)
val elapsed_ms : start:t -> stop:t -> float
