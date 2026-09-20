(** Runs a [Kernel_ir.host_op] plan against a loaded [Jit.module_] using
    the runtime layer. An executor owns its device buffers for its whole
    lifetime: allocate once at [create], free once at [release], and a
    [run] that only uploads, launches and downloads.

    It also owns a stream. Every copy and every launch of every run is
    issued on it, so the plan is ordered by the stream rather than by the
    host, and executors on different streams overlap. *)

open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime

type t

(** Runs every [Alloc] in the plan once. Raises if the plan allocates a
    buffer it never frees or frees one it never allocates (the plan is
    validated, not trusted). *)
val create : Kernel_ir.program -> Jit.module_ -> t

(** [create] on {!Stream.default}; [create_on] picks the stream. Two
    executors of one program on two streams may be driven concurrently:
    they have separate buffer pools. *)
val create_on : stream:Stream.t -> Kernel_ir.program -> Jit.module_ -> t

val stream : t -> Stream.t

(** [Upload], [Launch], [Download] only; [Alloc]/[Free] are skipped.
    Outputs are fresh host [Value]s. Raises [Invalid_argument] on a
    missing input and [Failure] after [release]. *)
val run : t -> inputs:(string * Value.packed) list -> (string * Value.packed) list

(** Like {!run}, but returns as soon as the work is {e issued}. The
    returned closure blocks on an event recorded after the downloads and
    then yields the outputs; calling it twice is harmless and yields the
    same values. The closure owns the input host [Value]s, so they
    outlive the copies that read them. *)
val run_async : t -> inputs:(string * Value.packed) list -> unit -> (string * Value.packed) list

(** Device in, device out: no host traffic at all. Inputs are buffers of
    exactly the byte size of the parameter they bind to (checked; raises
    [Invalid_argument] otherwise) and are substituted for the plan's own
    buffers, so nothing is copied. Outputs are freshly allocated buffers
    the caller owns and must [Buffer.free]. Synchronises before it
    returns, so the outputs are safe to use from any stream. *)
val run_resident : t -> inputs:(string * Buffer.t) list -> (string * Buffer.t) list

(** Frees every buffer. Idempotent. *)
val release : t -> unit

(** For tests: number of device buffers this executor holds. *)
val buffer_count : t -> int
