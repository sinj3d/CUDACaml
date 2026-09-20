(** Fusion, as an analysis rather than a rewrite.

    Element-wise nodes ([Map], [Map2], [Iota], [Gather], [Reshape]) are
    never given their own buffer unless they must be; [Lower] inlines their
    element expression into whichever kernel consumes them. This gives
    map-map, map-map2 and map-into-reduce fusion without adding an n-ary
    node to the IR. The only decision is which nodes get materialised. *)

open Cudacaml_ir

type plan

val plan : Graph.t -> plan

(** True for: every [Param]; every [Reduce], [Scan] and [Scatter_add]
    (fusion barriers); every program output; and any element-wise node with
    [Graph.fan_out] of 2 or more (so work is never duplicated across
    kernels). *)
val is_materialized : plan -> Tensor.packed -> bool

(** The materialised nodes in [Graph.topological_order], excluding
    [Param]s. At least one kernel is emitted per entry: [Scatter_add] needs
    two (a zero-fill followed by the atomics), everything else exactly
    one. *)
val kernel_roots : plan -> Tensor.packed list
