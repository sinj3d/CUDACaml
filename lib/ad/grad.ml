(** Reverse-mode AD over the tensor DAG.

    The whole transform is one reverse walk of [Graph.topological_order]
    with a table of adjoints keyed by [Uid.to_int]. Nothing is evaluated,
    nothing in [g] is mutated, and every tensor built here comes out of
    [Dsl], so the result contains no node kind that [Fusion] and [Lower] did
    not already handle.

    Two GADT habits recur below:

    - a [Tensor.packed] is opened with an explicit [match], never a [let]
      pattern, so the existential stays inside its scope;
    - a node's operand may have a different element type from the node
      itself ([Cast] inside a [Map]), so an adjoint is only added to an
      operand after [Dtype.equal] has produced a witness -- or, between two
      float dtypes, after an explicit [Dsl.cast]. *)

open Ocaml_cuda_ir

exception Not_differentiable of string

let grad_name ~output ~wrt = "d" ^ output ^ "/d" ^ wrt

(* ------------------------------------------------------------------ *)
(* Literals at an element type                                          *)
(* ------------------------------------------------------------------ *)

(* Matching the dtype refines ['a] to [float] in the two float arms; the
   integer arms are unreachable, because an adjoint is only ever created for
   a float node. *)
let fconst : type a. a Dtype.t -> float -> a Expr.t =
 fun dt v ->
  match dt with
  | Dtype.F32 -> Dsl.const Dtype.F32 v
  | Dtype.F64 -> Dsl.const Dtype.F64 v
  | Dtype.I32 | Dtype.I64 | Dtype.Bool ->
      raise
        (Not_differentiable
           (Printf.sprintf "dtype %s carries no gradient" (Dtype.name dt)))

let fones : type a. a Dtype.t -> Shape.t -> a Tensor.t =
 fun dt shape ->
  match dt with
  | Dtype.F32 -> Dsl.full Dtype.F32 shape 1.0
  | Dtype.F64 -> Dsl.full Dtype.F64 shape 1.0
  | Dtype.I32 | Dtype.I64 | Dtype.Bool ->
      raise
        (Not_differentiable
           (Printf.sprintf "dtype %s carries no gradient" (Dtype.name dt)))

(* The zero gradient handed to a param the output does not depend on. Every
   dtype a [Value] can hold is covered, so an integer param still gets a
   well-formed (and constantly zero) gradient tensor. *)
let zeros : type a. a Dtype.t -> Shape.t -> a Tensor.t =
 fun dt shape ->
  match dt with
  | Dtype.F32 -> Dsl.full Dtype.F32 shape 0.0
  | Dtype.F64 -> Dsl.full Dtype.F64 shape 0.0
  | Dtype.I32 -> Dsl.full Dtype.I32 shape 0l
  | Dtype.I64 -> Dsl.full Dtype.I64 shape 0L
  | Dtype.Bool ->
      invalid_arg "Grad.grad: a bool tensor cannot hold a gradient"

(* ------------------------------------------------------------------ *)
(* The adjoint table                                                    *)
(* ------------------------------------------------------------------ *)

type tbl = (int, Tensor.packed) Hashtbl.t

let find_adj : type a. tbl -> a Tensor.t -> a Tensor.t option =
 fun tbl node ->
  match Hashtbl.find_opt tbl (Uid.to_int node.Tensor.uid) with
  | None -> None
  | Some (Tensor.P a) -> (
      match Dtype.equal node.Tensor.dtype a.Tensor.dtype with
      | Some Dtype.Equal -> Some a
      | None -> invalid_arg "Grad: internal adjoint dtype mismatch")

(* [accumulate] never touches the forward node: the sum lives only in the
   table, as a fresh [Map2 (add, ...)]. *)
let accumulate : type a. tbl -> a Tensor.t -> a Tensor.t -> unit =
 fun tbl node v ->
  let k = Uid.to_int node.Tensor.uid in
  match find_adj tbl node with
  | None -> Hashtbl.replace tbl k (Tensor.P v)
  | Some a -> Hashtbl.replace tbl k (Tensor.P (Dsl.map2 Dsl.add a v))

(* Add [v] to the adjoint of operand [tgt]. An integer operand carries no
   adjoint and is silently skipped (the [Select] condition and the [Gather]
   index are the usual cases). A float operand of a different float dtype --
   which only a [Cast] inside an element function can produce -- takes the
   adjoint through that cast. *)
let accum_into : type a p. tbl -> p Tensor.t -> a Tensor.t -> unit =
 fun tbl tgt v ->
  if Dtype.is_float tgt.Tensor.dtype then
    match Dtype.equal tgt.Tensor.dtype v.Tensor.dtype with
    | Some Dtype.Equal -> accumulate tbl tgt v
    | None ->
        accumulate tbl tgt
          (Dsl.map (fun e -> Dsl.cast tgt.Tensor.dtype e) v)

(* ------------------------------------------------------------------ *)
(* Derivatives of element functions                                     *)
(* ------------------------------------------------------------------ *)

(* [Deriv.fn1]/[Deriv.fn2] insist on a single element type throughout; these
   are the same definitions with the operand types left free, so a body that
   casts an integer argument up to float still differentiates (to zero in
   that argument, which is what [Deriv.d] already returns for a non-float
   subtree under a [Cast]).

   [Deriv.d] raises [Invalid_argument] on a body it cannot differentiate;
   that is a property of the node, so it is re-raised as
   [Not_differentiable] naming the node kind. *)
let dfn1 : type p r. what:string -> (p, r) Expr.fn1 -> (p, r) Expr.fn1 =
 fun ~what f ->
  match Deriv.d f.Expr.body1 ~wrt:0 with
  | body1 -> { Expr.arg1 = f.Expr.arg1; body1 }
  | exception Invalid_argument m -> raise (Not_differentiable (what ^ ": " ^ m))

let dfn2 :
    type p q r. what:string -> (p, q, r) Expr.fn2 -> wrt:int -> (p, q, r) Expr.fn2
    =
 fun ~what f ~wrt ->
  match Deriv.d f.Expr.body2 ~wrt with
  | body2 -> { Expr.arg_a = f.Expr.arg_a; arg_b = f.Expr.arg_b; body2 }
  | exception Invalid_argument m -> raise (Not_differentiable (what ^ ": " ^ m))

(* ------------------------------------------------------------------ *)
(* Recognising a reduction operator                                     *)
(* ------------------------------------------------------------------ *)

let binop_name = function
  | Expr.Add -> "Add"
  | Expr.Sub -> "Sub"
  | Expr.Mul -> "Mul"
  | Expr.Div -> "Div"
  | Expr.Min -> "Min"
  | Expr.Max -> "Max"
  | Expr.Bit_and -> "Bit_and"
  | Expr.Bit_or -> "Bit_or"
  | Expr.Bit_xor -> "Bit_xor"
  | Expr.Shl -> "Shl"
  | Expr.Shr -> "Shr"

let is_arg : type a. int -> a Expr.t -> bool =
 fun i e ->
  match e.Expr.node with
  | Expr.Arg j -> j = i
  | Expr.Const _ | Expr.Index | Expr.Binop _ | Expr.Unop _ | Expr.Cmp _
  | Expr.Logic _ | Expr.Not _ | Expr.Select _ | Expr.Cast _ ->
      false

(* Syntactic only, as the spec demands: the body must be exactly one binop
   over the two placeholders, in either order. No algebraic normalisation is
   attempted, so an operator written any other way is reported rather than
   silently mis-differentiated. *)
let combining_binop : type a. (a, a, a) Expr.fn2 -> Expr.binop option =
 fun f ->
  match f.Expr.body2.Expr.node with
  | Expr.Binop (op, a, b) ->
      if (is_arg 0 a && is_arg 1 b) || (is_arg 1 a && is_arg 0 b) then Some op
      else None
  | Expr.Const _ | Expr.Arg _ | Expr.Index | Expr.Unop _ | Expr.Cmp _
  | Expr.Logic _ | Expr.Not _ | Expr.Select _ | Expr.Cast _ ->
      None

let op_desc kind = function
  | Some op -> Printf.sprintf "%s %s" kind (binop_name op)
  | None -> Printf.sprintf "%s <unrecognised operator>" kind

(* ------------------------------------------------------------------ *)
(* Row helpers                                                          *)
(* ------------------------------------------------------------------ *)

let row_length shape =
  match List.rev (Shape.dims shape) with [] -> 1 | n :: _ -> n

(* Replicate a row-shaped tensor (one element per row of [src_shape]) back
   over the source: [out[k] = t[k / n]]. A rank-1 source has one row, so
   [k / n = 0] and this is the scalar broadcast; rank >= 2 gets the real row
   index. One form covers both, which is why it is a [gather] rather than a
   [Broadcast]. *)
let rows_of : type a. src_shape:Shape.t -> a Tensor.t -> a Tensor.t =
 fun ~src_shape t ->
  let n = row_length src_shape in
  let ni = Dsl.const Dtype.I32 (Int32.of_int n) in
  let idx = Dsl.map (fun k -> Dsl.div k ni) (Dsl.iota src_shape) in
  Dsl.gather idx t

(* Reduce / scan along the last axis, whatever the rank: the whole-tensor
   forms already are the last-axis forms at rank 1, and the row forms reject
   rank 1. *)
let reduce_last : type a. a Tensor.t -> a Tensor.t =
 fun t ->
  let init = fconst t.Tensor.dtype 0.0 in
  if Shape.rank t.Tensor.shape <= 1 then Dsl.reduce Dsl.add ~init t
  else Dsl.reduce_rows Dsl.add ~init t

let scan_last : type a. a Tensor.t -> a Tensor.t =
 fun t ->
  let init = fconst t.Tensor.dtype 0.0 in
  if Shape.rank t.Tensor.shape <= 1 then Dsl.scan Dsl.add ~init t
  else Dsl.scan_rows Dsl.add ~init t

(* ------------------------------------------------------------------ *)
(* One node's adjoint rule                                              *)
(* ------------------------------------------------------------------ *)

(* [abar] is the completed adjoint of [t]; push it to [t]'s operands. *)
let back_node : type a. tbl -> a Tensor.t -> a Tensor.t -> unit =
 fun tbl t abar ->
  match t.Tensor.node with
  | Tensor.Param _ -> ()
  | Tensor.Iota -> ()
  | Tensor.Map (f, src) ->
      if Dtype.is_float src.Tensor.dtype then begin
        let df = dfn1 ~what:"Map" f in
        accum_into tbl src
          (Dsl.map2 (fun adj x -> Dsl.mul adj (Deriv.apply1 df x)) abar src)
      end
  | Tensor.Map2 (f, xa, xb) ->
      let contribute wrt (tgt : _ Tensor.t) =
        if Dtype.is_float tgt.Tensor.dtype then begin
          let df = dfn2 ~what:"Map2" f ~wrt in
          let dk = Dsl.map2 (fun x y -> Deriv.apply2 df x y) xa xb in
          accum_into tbl tgt (Dsl.map2 Dsl.mul abar dk)
        end
      in
      contribute 0 xa;
      contribute 1 xb
  | Tensor.Reduce (f, _init, src) -> (
      let src_shape = src.Tensor.shape in
      let spread u = rows_of ~src_shape u in
      match combining_binop f with
      | Some Expr.Add -> accum_into tbl src (spread abar)
      | Some (Expr.Max | Expr.Min) ->
          (* Every element equal to the extremum receives the whole adjoint.
             At a tie that over-counts by the multiplicity; it is the
             standard convention and matches a one-sided bump only when the
             extremum is unique. *)
          let m = spread t in
          let one = fconst t.Tensor.dtype 1.0
          and zero = fconst t.Tensor.dtype 0.0 in
          let mask =
            Dsl.map2 (fun x mv -> Dsl.select (Dsl.eq x mv) one zero) src m
          in
          accum_into tbl src (Dsl.map2 Dsl.mul (spread abar) mask)
      | Some Expr.Mul ->
          (* d(prod)/dx_i = prod / x_i, which needs every element non-zero. *)
          let q = Dsl.map2 Dsl.div (spread t) src in
          accum_into tbl src (Dsl.map2 Dsl.mul (spread abar) q)
      | Some
          (( Expr.Sub | Expr.Div | Expr.Bit_and | Expr.Bit_or | Expr.Bit_xor
           | Expr.Shl | Expr.Shr ) as op) ->
          raise (Not_differentiable (op_desc "Reduce" (Some op)))
      | None -> raise (Not_differentiable (op_desc "Reduce" None)))
  | Tensor.Scan (f, _init, src) -> (
      match combining_binop f with
      | Some Expr.Add ->
          (* The adjoint of an inclusive prefix sum is a reverse-inclusive
             prefix sum within each row:
               adj_src[i] = row_total - prefix(abar)[i] + abar[i]. *)
          let total = rows_of ~src_shape:src.Tensor.shape (reduce_last abar) in
          let prefix = scan_last abar in
          accum_into tbl src
            (Dsl.map2 Dsl.add (Dsl.map2 Dsl.sub total prefix) abar)
      | Some
          (( Expr.Sub | Expr.Mul | Expr.Div | Expr.Min | Expr.Max
           | Expr.Bit_and | Expr.Bit_or | Expr.Bit_xor | Expr.Shl | Expr.Shr )
          as op) ->
          raise (Not_differentiable (op_desc "Scan" (Some op)))
      | None -> raise (Not_differentiable (op_desc "Scan" None)))
  | Tensor.Gather (idx, src) ->
      (* out[i] = src[idx[i]], so adj_src[idx[i]] += abar[i]: exactly a
         scatter-add back into the source's shape. The accumulation is what
         makes a repeated index correct -- two output elements reading the
         same source element must both contribute. [idx] is an integer
         tensor and carries no adjoint of its own. *)
      if Dtype.is_float src.Tensor.dtype then
        accumulate tbl src (Dsl.scatter_add idx abar src.Tensor.shape)
  | Tensor.Scatter_add (idx, src, _) ->
      (* The exact transpose of [Gather]: out[idx[i]] += src[i], so
         adj_src[i] = abar[idx[i]] -- a gather by the very same index. No
         accumulation is needed on this side, because every source element
         contributes to exactly one output element. *)
      accum_into tbl src (Dsl.gather idx abar)
  | Tensor.Reshape (_, src) ->
      accum_into tbl src (Dsl.reshape src.Tensor.shape abar)
  | Tensor.Broadcast (_, src) ->
      let total = Dsl.reduce Dsl.add ~init:(fconst t.Tensor.dtype 0.0) abar in
      accum_into tbl src (Dsl.reshape src.Tensor.shape total)

let backprop tbl (Tensor.P t) =
  match find_adj tbl t with None -> () | Some abar -> back_node tbl t abar

(* ------------------------------------------------------------------ *)
(* The transform                                                        *)
(* ------------------------------------------------------------------ *)

let lookup what assoc name =
  match List.assoc_opt name assoc with
  | Some p -> p
  | None ->
      invalid_arg (Printf.sprintf "Grad.grad: no %s named %S" what name)

let seed tbl (Tensor.P out) =
  if not (Dtype.is_float out.Tensor.dtype) then
    invalid_arg
      (Printf.sprintf "Grad.grad: output has dtype %s, a float dtype is required"
         (Dtype.name out.Tensor.dtype));
  (* [Dsl.full] rather than [map (fun _ -> 1) out]: seeding through the
     output node would add a use of it and change its fan-out. *)
  accumulate tbl out (fones out.Tensor.dtype out.Tensor.shape)

let gradient_of tbl (Tensor.P p) =
  match find_adj tbl p with
  | Some a -> Tensor.P a
  | None -> Tensor.P (zeros p.Tensor.dtype p.Tensor.shape)

let grad g ~output ~wrt =
  let outs = Graph.outputs g in
  let params = Graph.params g in
  let out = lookup "output" outs output in
  (* Resolve every [wrt] name before any work, so an unknown one raises
     [Invalid_argument] rather than [Not_differentiable]. *)
  let wrt_nodes = List.map (fun p -> (p, lookup "param" params p)) wrt in
  let tbl : tbl = Hashtbl.create 64 in
  seed tbl out;
  List.iter (backprop tbl) (List.rev (Graph.topological_order g));
  let grads =
    List.map (fun (p, node) -> (grad_name ~output ~wrt:p, gradient_of tbl node))
      wrt_nodes
  in
  Graph.create ~name:(Graph.name g ^ "_grad") ~outputs:(outs @ grads)
