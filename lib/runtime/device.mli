(** Device discovery and context lifetime. *)

(** false on machines with no CUDA driver *)
val available : unit -> bool

(** cuInit + primary context; idempotent *)
val init : unit -> unit

val synchronize : unit -> unit
val name : unit -> string

type info = {
  name : string;
  compute_capability : int * int;  (** (major, minor); sm_120 is (12, 0) *)
  multiprocessors : int;
  total_memory_bytes : int;
}

(** Device 0. Raises when no device is present; call [available] first. *)
val info : unit -> info

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
