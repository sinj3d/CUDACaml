(** Device discovery and context lifetime. *)

(** false on machines with no CUDA driver *)
val available : unit -> bool

(** cuInit + primary context; idempotent *)
val init : unit -> unit

val synchronize : unit -> unit
val name : unit -> string

(** {1 Device selection}

    Every device owns a primary context, and a context owns its
    allocations: a [Buffer] allocated under device 0 is not a valid
    pointer under device 1, and a kernel launched under the wrong context
    reads garbage or faults. Nothing below takes a device argument --
    [Buffer], [Jit], [Stream] and [Launch] all act on whichever context is
    current on the calling thread -- so a multi-device caller brackets
    every device-touching call with {!with_device}. *)

(** How many CUDA devices the driver reports. *)
val count : unit -> int

(** Make device [ordinal]'s primary context current on this thread for the
    call. Nested calls restore the previous device. [init ()] is device 0
    and remains the default. *)
val with_device : int -> (unit -> 'a) -> 'a

(** The ordinal whose context {!with_device} has made current, or 0 when
    no {!with_device} is in progress. *)
val current : unit -> int

type info = {
  name : string;
  compute_capability : int * int;  (** (major, minor); sm_120 is (12, 0) *)
  multiprocessors : int;
  total_memory_bytes : int;
}

(** Device 0. Raises when no device is present; call [available] first. *)
val info : unit -> info

(** Any device. Raises when [ordinal] is not a device the driver reports. *)
val info_of : int -> info

(** One line per field, in this order and with these exact keys, so a
    shell script can grep it:
    {v
      device: <name>
      compute: sm_<major><minor>
      multiprocessors: <n>
      memory_mib: <total / 1048576>
    v}
    The [sm_] spelling concatenates major and minor without a separator,
    which is how nvcc names architectures ([sm_80], [sm_120]). *)
val info_to_string : info -> string
