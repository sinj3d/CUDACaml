(** Device memory. A [t] is an opaque device pointer plus a byte count;
    the only way data moves between host and device. *)

open Ocaml_cuda_ir

type t

val alloc : bytes:int -> t
val free : t -> unit
val byte_size : t -> int

(** Allocations minus frees since process start. For tests and leak checks. *)
val live_count : unit -> int

(** Synchronous copies. [download] writes into an existing host value
    whose [byte_size] must match. *)
val upload : Value.packed -> t -> unit

val download : t -> Value.packed -> unit

(** Stream-ordered copies. They return once the copy is {e issued}: the
    host [Value] must stay reachable until [stream] is synchronised,
    because nothing here keeps it alive. Callers hold it. *)
val upload_async : Value.packed -> t -> stream:Stream.t -> unit

val download_async : t -> Value.packed -> stream:Stream.t -> unit

(** Device-to-device, stream-ordered. Sizes must match. *)
val copy_device : dst:t -> src:t -> stream:Stream.t -> unit

val unsafe_ptr : t -> Cuda.Deviceptr.t
