(** Counter-based random numbers, built out of the DSL and nothing else.

    The generator is Philox-4x32-10 (Salmon, Moraes, Dror, Shaw 2011), the
    default in cuRAND, JAX and NumPy: a pure function from (counter, key) to
    four 32-bit words. Element [i] uses counter [i], so a random tensor is a
    [map] over [iota] -- stateless, reproducible, parallel by construction,
    and it fuses into whatever consumes it. The seed is a [Param] of shape
    [Shape.scalar], broadcast, so a new seed is a new input and not a
    re-JIT. *)

open Ocaml_cuda_ir

module Philox : sig
  (** Philox-4x32-10. [ctr] is the 128-bit counter as four I32 words
      (c0 least significant), [key] the 64-bit key as two I32 words. Returns
      the four output words. Pure expression; every word may be used. *)
  val round10 :
    ctr:int32 Expr.t * int32 Expr.t * int32 Expr.t * int32 Expr.t ->
    key:int32 Expr.t * int32 Expr.t ->
    int32 Expr.t * int32 Expr.t * int32 Expr.t * int32 Expr.t
end

(** [u32 ~key shape]: element [i] is output word 0 of
    [Philox.round10 ~ctr:(i, 0, 0, 0) ~key:(key, 0)]. [key] must have numel 1
    (normally [Dsl.scalar "seed" Dtype.I32]). Uniform over all 2^32 bit
    patterns, delivered as a signed [int32]. *)
val u32 : key:int32 Tensor.t -> Shape.t -> int32 Tensor.t

(** [to_unit_interval dtype w]: the 32 bits of [w] read as unsigned, mapped
    to the OPEN interval (0, 1) as [(u + 0.5) * 2^-32], computed in F64 and
    then cast to [dtype]. *)
val to_unit_interval : float Dtype.t -> int32 Expr.t -> float Expr.t

(** Standard normal from a uniform in (0,1): [sqrt 2 * erfinv (2u - 1)]. *)
val normal_inv_cdf : float Expr.t -> float Expr.t

(** [uniform dtype ~key shape] = [map (to_unit_interval dtype) (u32 ~key shape)]. *)
val uniform : float Dtype.t -> key:int32 Tensor.t -> Shape.t -> float Tensor.t

(** [normal dtype ~key shape] = [map normal_inv_cdf (uniform dtype ~key shape)]. *)
val normal : float Dtype.t -> key:int32 Tensor.t -> Shape.t -> float Tensor.t
