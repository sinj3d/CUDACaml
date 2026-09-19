(** A whole program: named parameters in, named tensors out. The unit of
    compilation.

    Hardcaml analogue: [Circuit]. Built from outputs only; inputs are
    discovered by back-tracing to [Tensor.Param] nodes. *)

type t

val create : name:string -> outputs:(string * Tensor.packed) list -> t
val name : t -> string
val params : t -> (string * Tensor.packed) list
val outputs : t -> (string * Tensor.packed) list

(** Dependencies before dependents. Stable: output order respects input
    order, so lowering is deterministic. *)
val topological_order : t -> Tensor.packed list

val iter : t -> f:(Tensor.packed -> unit) -> unit
val fold : t -> init:'acc -> f:('acc -> Tensor.packed -> 'acc) -> 'acc

(** Consumer count. Fusion may only pull a producer into its consumer when
    this is 1. *)
val fan_out : t -> Tensor.packed -> int

val to_dot : t -> string
