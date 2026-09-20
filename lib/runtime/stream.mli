(** CUDA streams. A stream is an ordered queue of device work: everything
    issued on one stream runs in issue order, and work on two different
    streams may overlap.

    The whole runtime is stream-aware through this type, so a caller that
    never mentions a stream keeps the v1 behaviour of running on
    {!default}. *)

type t

(** The NULL stream. Legacy-synchronising: work on it does not overlap
    with work on a blocking stream. Every stream {!create} hands out is a
    non-blocking one, so the NULL stream does not serialise them. *)
val default : t

(** A fresh non-blocking stream. Streams are scarce-ish and long-lived:
    create them once, at compile time, and reuse them. *)
val create : unit -> t

(** Blocks the calling thread until every operation issued on [t] has
    completed. This is also the only thing that hands back the host-side
    resources cudajit retains per launch (the marshalled kernel argument
    arrays), so a loop that only ever waits on {!Event}s should reach for
    {!is_idle} now and then. *)
val synchronize : t -> unit

(** Non-blocking: true when every operation issued on [t] has completed.
    Calling it is also how a caller that never synchronises reclaims the
    retained launch arguments, which is why it is worth calling and
    discarding the answer. *)
val is_idle : t -> bool

(* The raw driver handle, for [Buffer] and [Launch]. Mirrors
   [Buffer.unsafe_ptr] and [Jit.unsafe_func]: the runtime layer *is* the
   cudajit wrapper, so this is the intended seam, not a leak. *)
val unsafe_stream : t -> Cuda.Stream.t
