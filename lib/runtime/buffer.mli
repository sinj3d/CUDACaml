(** Device memory. A [t] is an opaque device pointer plus a byte count;
    the only way data moves between host and device. *)

open Ocaml_cuda_ir

type t

val alloc : bytes:int -> t
val free : t -> unit
val byte_size : t -> int

(** Synchronous copies. [download] writes into an existing host value
    whose [byte_size] must match. *)
val upload : Value.packed -> t -> unit

val download : t -> Value.packed -> unit
