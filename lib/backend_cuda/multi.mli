(** Data-parallel runs across several devices.

    Monte Carlo is embarrassingly parallel: split the paths, run the same
    program on every card, combine on the host. A [t] holds one JIT
    compilation and one {!Executor.t} per device, each built and driven
    under that device's own context.

    Everything that touches the driver here goes through
    {!Ocaml_cuda_runtime.Device.with_device}, because a buffer allocated
    under one device's context is not a valid pointer under another's. *)

open Ocaml_cuda_ir

type t

(** [create ~devices ~n build]: [build ~n:(n / d)] is compiled on each
    device in [devices] (d = List.length devices; [n] must be divisible by
    d). The leading dimension [n] is the split axis.

    [devices] may repeat an ordinal -- [[0; 0]] is two independent
    executors, each on its own stream, on one card -- which is how the
    splitting logic is tested on a single-GPU machine. *)
val create : devices:int list -> n:int -> (n:int -> Graph.t) -> t

(** Inputs whose leading dimension is [n] are split into d contiguous
    slices; inputs with any other shape are passed whole to every device.
    Returns one output list per device, in [devices] order. All devices
    are launched before any is waited on.

    The split rule is by shape alone, so a parameter table that happens to
    have [n] rows would be sliced too. Reshape it, or pass it through
    {!run_per_device}, to avoid that. *)
val run : t -> inputs:(string * Value.packed) list -> (string * Value.packed) list list

(** [run] with the inputs chosen per device: the function receives the
    index into [devices] (0 .. d-1), not the ordinal, so it distinguishes
    the two halves of a [[0; 0]] split.

    This is the seam for seeding. The same [seed] on every device gives
    every device the same paths, and averaging identical estimates buys
    nothing; {!Multi} deliberately does not invent a per-device seed for
    the caller, because only the caller knows which parameter is the RNG
    key. Pass [fun d -> ... ~seed:(d + 1) ...] here instead.

    As in {!run}, every device is launched before any is waited on. *)
val run_per_device :
  t -> inputs:(int -> (string * Value.packed) list) -> (string * Value.packed) list list

(** Concatenate per-device outputs along the leading axis (for outputs
    whose leading dimension is n / d). Every value must share a dtype and
    agree on every axis but the first; rank 0 has no leading axis and is
    rejected. *)
val concat : Value.packed list -> Value.packed

(** Mean of per-device scalar outputs (for Monte Carlo means). Each value
    must hold exactly one element. Note that this is the plain mean, which
    is the right combination only when the devices ran equal path counts
    -- which is what {!create} guarantees. *)
val mean_scalar : Value.packed list -> float

(** Frees every device's buffers, each under its own context. Idempotent;
    [run] after it raises. A dropped [t] is released by a finaliser, but
    do not rely on that. *)
val release : t -> unit
