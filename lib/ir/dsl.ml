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

let reduce f ~init (src : _ Tensor.t) =
  let fn = Expr.fn2 src.dtype src.dtype f in
  Tensor.make src.dtype Shape.scalar (Tensor.Reduce (fn, init, src))

let scan f ~init (src : _ Tensor.t) =
  let fn = Expr.fn2 src.dtype src.dtype f in
  Tensor.make src.dtype src.shape (Tensor.Scan (fn, init, src))

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
let neg a = unop Expr.Neg a
let sqrt a = unop Expr.Sqrt a
let exp a = unop Expr.Exp a
let log a = unop Expr.Log a
let lt a b = cmp Expr.Lt a b
let le a b = cmp Expr.Le a b
let eq a b = cmp Expr.Eq a b
let select c (a : _ Expr.t) b = Expr.make a.dtype (Expr.Select (c, a, b))
let cast dtype e = Expr.make dtype (Expr.Cast (e, dtype))
