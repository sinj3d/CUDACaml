(** Memory-layout assignment.

    v1: every tensor is a dense, row-major, contiguous global buffer, and
    this pass only validates that. It is a separate pass so that tiled or
    struct-of-arrays layouts have a home without touching [Lower]. *)

include Pass.S
