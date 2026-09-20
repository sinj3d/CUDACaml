(** Runs a [Kernel_ir.host_op] plan against a loaded [Jit.module_] using
    the runtime layer. An executor owns its device buffers for its whole
    lifetime: allocate once at [create], free once at [release], and a
    [run] that only uploads, launches and downloads. *)

open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime

type t

(** Runs every [Alloc] in the plan once. Raises if the plan allocates a
    buffer it never frees or frees one it never allocates (the plan is
    validated, not trusted). *)
val create : Kernel_ir.program -> Jit.module_ -> t

(** [Upload], [Launch], [Download] only; [Alloc]/[Free] are skipped.
    Outputs are fresh host [Value]s. Raises [Invalid_argument] on a
    missing input and [Failure] after [release]. *)
val run : t -> inputs:(string * Value.packed) list -> (string * Value.packed) list

(** Frees every buffer. Idempotent. *)
val release : t -> unit

(** For tests: number of device buffers this executor holds. *)
val buffer_count : t -> int
