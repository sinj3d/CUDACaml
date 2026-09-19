(* Tensors *)
let param name dtype shape = Tensor.make dtype shape (Tensor.Param name)
let iota shape = Tensor.make Dtype.I32 shape Tensor.Iota

let map f (src : _ Tensor.t) =
  let fn = Expr.fn1 src.dtype f in
  Tensor.make fn.body1.dtype src.shape (Tensor.Map (fn, src))

let map2 f (a : _ Tensor.t) (b : _ Tensor.t) =
  (* No broadcasting in v1: both operands must have the same shape. *)
  if not (Shape.equal a.shape b.shape) then
    invalid_arg
      (Printf.sprintf "Dsl.map2: shape mismatch: %s vs %s" (Shape.to_string a.shape)
         (Shape.to_string b.shape));
  let fn = Expr.fn2 a.dtype b.dtype f in
  Tensor.make fn.body2.dtype a.shape (Tensor.Map2 (fn, a, b))

let gather (idx : int32 Tensor.t) (src : _ Tensor.t) =
  Tensor.make src.dtype idx.shape (Tensor.Gather (idx, src))

let reshape shape (src : _ Tensor.t) =
  (* Metadata only: the element count must be preserved. *)
  if Shape.numel shape <> Shape.numel src.shape then
    invalid_arg
      (Printf.sprintf "Dsl.reshape: numel mismatch: %s (%d) vs %s (%d)"
         (Shape.to_string shape) (Shape.numel shape)
         (Shape.to_string src.shape)
         (Shape.numel src.shape));
  Tensor.make src.dtype shape (Tensor.Reshape (shape, src))

(* [Tensor.Reduce] and [Tensor.Scan] act along the last axis. The two
   surface forms differ only in what they hand the constructor: the
   whole-tensor forms flatten to rank 1 first, the row forms pass the source
   through unchanged. The output shape of a [Reduce] is decided HERE and
   nowhere else -- [Lower] and the interpreter read it off the node. *)

let all_but_last dims =
  match List.rev dims with [] -> [] | _ :: rest -> List.rev rest

let flatten (src : _ Tensor.t) =
  reshape (Shape.of_dims [ Shape.numel src.shape ]) src

let reduce f ~init (src : _ Tensor.t) =
  let fn = Expr.fn2 src.dtype src.dtype f in
  (* Rank 1 builds the node directly rather than going through a no-op
     [reshape], so a v1 program lowers to exactly the graph it always did. *)
  if Shape.rank src.shape = 1 then
    Tensor.make src.dtype Shape.scalar (Tensor.Reduce (fn, init, src))
  else
    let flat = flatten src in
    Tensor.make src.dtype Shape.scalar (Tensor.Reduce (fn, init, flat))

let scan f ~init (src : _ Tensor.t) =
  let fn = Expr.fn2 src.dtype src.dtype f in
  if Shape.rank src.shape = 1 then
    Tensor.make src.dtype src.shape (Tensor.Scan (fn, init, src))
  else
    let flat = flatten src in
    let scanned = Tensor.make src.dtype flat.shape (Tensor.Scan (fn, init, flat)) in
    reshape src.shape scanned

let reduce_rows f ~init (src : _ Tensor.t) =
  if Shape.rank src.shape < 2 then
    invalid_arg
      (Printf.sprintf "Dsl.reduce_rows: rank >= 2 required, got %s"
         (Shape.to_string src.shape));
  let fn = Expr.fn2 src.dtype src.dtype f in
  let out_shape = Shape.of_dims (all_but_last (Shape.dims src.shape)) in
  Tensor.make src.dtype out_shape (Tensor.Reduce (fn, init, src))

let scan_rows f ~init (src : _ Tensor.t) =
  if Shape.rank src.shape < 2 then
    invalid_arg
      (Printf.sprintf "Dsl.scan_rows: rank >= 2 required, got %s"
         (Shape.to_string src.shape));
  let fn = Expr.fn2 src.dtype src.dtype f in
  Tensor.make src.dtype src.shape (Tensor.Scan (fn, init, src))

let broadcast shape (src : _ Tensor.t) =
  (* The one-element precondition is checked HERE and nowhere else: [Lower]
     and the interpreter read [src.[0]] unconditionally and trust the
     graph. *)
  if Shape.numel src.shape <> 1 then
    invalid_arg
      (Printf.sprintf "Dsl.broadcast: source must have exactly one element: %s (%d)"
         (Shape.to_string src.shape)
         (Shape.numel src.shape));
  Tensor.make src.dtype shape (Tensor.Broadcast (shape, src))

(* Scalars *)
let const dtype v = Expr.make dtype (Expr.Const v)
let index () = Expr.make Dtype.I32 Expr.Index
let binop op (a : _ Expr.t) b = Expr.make a.dtype (Expr.Binop (op, a, b))
let unop op (a : _ Expr.t) = Expr.make a.dtype (Expr.Unop (op, a))
let cmp op a b = Expr.make Dtype.Bool (Expr.Cmp (op, a, b))
let add a b = binop Expr.Add a b
let sub a b = binop Expr.Sub a b
let mul a b = binop Expr.Mul a b
let div a b = binop Expr.Div a b
let min a b = binop Expr.Min a b
let max a b = binop Expr.Max a b
let bit_and a b = binop Expr.Bit_and a b
let bit_or a b = binop Expr.Bit_or a b
let bit_xor a b = binop Expr.Bit_xor a b
let shl a b = binop Expr.Shl a b
let shr a b = binop Expr.Shr a b
let neg a = unop Expr.Neg a
let sqrt a = unop Expr.Sqrt a
let exp a = unop Expr.Exp a
let log a = unop Expr.Log a

(* These four shadow [Stdlib] under [open Dsl], as [sqrt], [exp] and [log]
   already do. *)
let sin a = unop Expr.Sin a
let cos a = unop Expr.Cos a
let erf a = unop Expr.Erf a
let erfinv a = unop Expr.Erfinv a
let lt a b = cmp Expr.Lt a b
let le a b = cmp Expr.Le a b
let eq a b = cmp Expr.Eq a b
let ne a b = cmp Expr.Ne a b
let gt a b = cmp Expr.Gt a b
let ge a b = cmp Expr.Ge a b
let and_ a b = Expr.make Dtype.Bool (Expr.Logic (Expr.And, a, b))
let or_ a b = Expr.make Dtype.Bool (Expr.Logic (Expr.Or, a, b))
let not_ a = Expr.make Dtype.Bool (Expr.Not a)
let select c (a : _ Expr.t) b = Expr.make a.dtype (Expr.Select (c, a, b))
let cast dtype e = Expr.make dtype (Expr.Cast (e, dtype))

(* Tensors built out of the above. [full] is a [Map] over [Iota] whose
   element function ignores its argument, so it carries no [Param] and no
   host buffer: [Fusion] inlines the [Iota] and [Lower] emits a bare
   literal. *)
let full dtype shape v = map (fun _ -> const dtype v) (iota shape)
let scalar name dtype = param name dtype Shape.scalar

(* [transpose] is a permutation, and a permutation is a [gather] over a
   computed index tensor -- no new node, no new lowering rule, and it fuses
   into its consumer like any other gather. For [x : [m; n]] the output is
   [[n; m]] and [out[k] = x[(k mod m) * n + k / m]].

   [mod] is deliberately absent from [Expr.binop] (one integer division per
   operation is enough), so it is spelled [k - (k / m) * m]. Both the
   division and the multiplication are on I32 device-side expressions. *)
let transpose (x : _ Tensor.t) =
  match Shape.dims x.shape with
  | [ m; n ] ->
      let mi = const Dtype.I32 (Int32.of_int m) in
      let ni = const Dtype.I32 (Int32.of_int n) in
      let idx =
        map
          (fun k ->
            let q = div k mi in
            add (mul (sub k (mul q mi)) ni) q)
          (iota (Shape.of_dims [ n; m ]))
      in
      gather idx x
  | _ ->
      invalid_arg
        (Printf.sprintf "Dsl.transpose: rank 2 required, got %s" (Shape.to_string x.shape))
