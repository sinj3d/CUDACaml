open Ocaml_cuda_ir
open Dsl

module Philox = struct
  (* Random123's PHILOX_M4x32_0/1 and PHILOX_W32_0/1. [Int32.of_string] on a
     hex literal wraps into the negative half of int32, which is exactly the
     bit pattern we want. *)
  let m0 = Int32.of_string "0xD2511F53"
  (* PHILOX_M4x32_1 is 0xCD9E8D57. Some transcriptions give
     0xCD9E8D7C, which reproduces none of the three known-answer vectors;
     the value below is what Random123's philox.h defines and what the KATs
     in test_rng.ml pin. *)
  let m1 = Int32.of_string "0xCD9E8D57"
  let w0 = Int32.of_string "0x9E3779B9"
  let w1 = Int32.of_string "0xBB67AE85"

  (* Zero-extend an I32 word into I64. A bare [cast I64] SIGN-extends, which
     corrupts the high half of the product for any operand with the top bit
     set; masking to the low 32 bits after the cast undoes that. *)
  let wide (x : int32 Expr.t) : int64 Expr.t =
    bit_and (cast Dtype.I64 x) (const Dtype.I64 0xFFFFFFFFL)

  (* The full 64-bit product, split. [lo] is the I32 product, which wraps.
     [hi] is the logical shift of the 64-bit product, so the sign the I64
     product happens to carry is irrelevant. *)
  let mulhilo32 (a : int32 Expr.t) (b : int32 Expr.t) : int32 Expr.t * int32 Expr.t =
    let lo = mul a b in
    let hi = cast Dtype.I32 (shr (mul (wide a) (wide b)) (const Dtype.I64 32L)) in
    (hi, lo)

  let round (c0, c1, c2, c3) (k0, k1) =
    let hi0, lo0 = mulhilo32 (const Dtype.I32 m0) c0 in
    let hi1, lo1 = mulhilo32 (const Dtype.I32 m1) c2 in
    (bit_xor (bit_xor hi1 c1) k0, lo1, bit_xor (bit_xor hi0 c3) k1, lo0)

  let bump (k0, k1) = (add k0 (const Dtype.I32 w0), add k1 (const Dtype.I32 w1))

  (* Ten rounds. The key is bumped after each of the first nine only: the
     tenth round's output IS the result, and a tenth bump -- while it cannot
     change that output -- would no longer match Random123. *)
  let round10 ~ctr ~key =
    let rec go n ctr key =
      let ctr = round ctr key in
      if n = 10 then ctr else go (n + 1) ctr (bump key)
    in
    go 1 ctr key
end

let u32 ~(key : int32 Tensor.t) shape =
  (* [iota] rather than [index ()]: it fuses to the same thing and keeps the
     graph reshape-safe, so the counter is always the flat index. *)
  let i = iota shape in
  let k = broadcast shape key in
  let z = const Dtype.I32 0l in
  map2
    (fun i k ->
      let w0, _, _, _ = Philox.round10 ~ctr:(i, z, z, z) ~key:(k, z) in
      w0)
    i k

let to_unit_interval (dtype : float Dtype.t) (w : int32 Expr.t) : float Expr.t =
  (* Read the word as unsigned by folding the negative half up by 2^32. The
     arithmetic is F64 throughout and the cast to [dtype] comes last: in F32
     the [+ 0.5] would swallow the low bits before the scaling. *)
  let f = cast Dtype.F64 w in
  let u = select (lt w (const Dtype.I32 0l)) (add f (const Dtype.F64 4294967296.0)) f in
  cast dtype (mul (add u (const Dtype.F64 0.5)) (const Dtype.F64 2.3283064365386963e-10))

let normal_inv_cdf (u : float Expr.t) : float Expr.t =
  let d = u.Expr.dtype in
  mul (const d (Stdlib.sqrt 2.0)) (erfinv (sub (mul (const d 2.0) u) (const d 1.0)))

let uniform dtype ~key shape = map (to_unit_interval dtype) (u32 ~key shape)
let normal dtype ~key shape = map normal_inv_cdf (uniform dtype ~key shape)
