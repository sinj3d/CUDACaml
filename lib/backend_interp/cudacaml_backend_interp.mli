(** Reference evaluator: walks [Graph.topological_order] and evaluates every
    [Tensor] node eagerly on host [Value]s, one element at a time.

    Deliberately naive and deliberately pass-free: it runs the graph as the
    user wrote it, so it is the oracle that [Pipeline] and [Backend_cuda]
    are both checked against. Speed is a non-goal. *)

include Cudacaml_backend.Backend.S
