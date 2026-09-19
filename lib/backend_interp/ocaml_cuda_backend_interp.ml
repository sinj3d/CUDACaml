open Ocaml_cuda_ir

let name = "interp"

type compiled = Graph.t

let compile g = g

(* ------------------------------------------------------------------ *)
(* Environment for element functions                                    *)
(* ------------------------------------------------------------------ *)

(* [Arg i] placeholders have different element types inside one body, so the
   environment stores packed typed values and lookup recovers the type with
   the [Dtype] equality witness. *)
type binding = B : 'a Dtype.t * 'a -> binding

let lookup : type a. binding list -> int -> a Dtype.t -> a =
 fun env i want ->
  match List.nth_opt env i with
  | None -> invalid_arg (Printf.sprintf "Backend_interp: Arg %d is unbound" i)
  | Some (B (have, v)) -> (
      match Dtype.equal want have with
      | Some Dtype.Equal -> v
      | None ->
          invalid_arg
            (Printf.sprintf "Backend_interp: Arg %d has dtype %s, expected %s" i
               (Dtype.name have) (Dtype.name want)))

(* ------------------------------------------------------------------ *)
(* Scalar operations                                                    *)
(* ------------------------------------------------------------------ *)

(* Every F32 result goes through this, so the interpreter rounds exactly
   where single-precision device arithmetic does. *)
let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The float family sees only the six arithmetic operators: the bitwise
   ones and the shifts are integer only. The arm is spelled out rather than
   wildcarded so that a future [binop] constructor is a compile error here
   too. *)
let float_binop op x y =
  match op with
  | Expr.Add -> x +. y
  | Expr.Sub -> x -. y
  | Expr.Mul -> x *. y
  | Expr.Div -> x /. y
  | Expr.Min -> Float.min x y
  | Expr.Max -> Float.max x y
  | Expr.Bit_and -> invalid_arg "Backend_interp: bit_and on a float dtype"
  | Expr.Bit_or -> invalid_arg "Backend_interp: bit_or on a float dtype"
  | Expr.Bit_xor -> invalid_arg "Backend_interp: bit_xor on a float dtype"
  | Expr.Shl -> invalid_arg "Backend_interp: shl on a float dtype"
  | Expr.Shr -> invalid_arg "Backend_interp: shr on a float dtype"

(* Integer division truncates toward zero, like C: [Int32.div (-7l) 2l = -3l].
   Division by zero raises [Division_by_zero] and is deliberately let through. *)
let binop : type a. a Dtype.t -> Expr.binop -> a -> a -> a =
 fun d op x y ->
  match d with
  | Dtype.F32 -> round_f32 (float_binop op x y)
  | Dtype.F64 -> float_binop op x y
  | Dtype.I32 -> (
      (* The shift count is masked to the bit width, exactly as the emitted
         [(b) & 31] does, and [Shr] is logical so the sign bit is not
         replicated: this is what makes the two backends bit-identical. *)
      match op with
      | Expr.Add -> Int32.add x y
      | Expr.Sub -> Int32.sub x y
      | Expr.Mul -> Int32.mul x y
      | Expr.Div -> Int32.div x y
      | Expr.Min -> if Int32.compare x y < 0 then x else y
      | Expr.Max -> if Int32.compare x y > 0 then x else y
      | Expr.Bit_and -> Int32.logand x y
      | Expr.Bit_or -> Int32.logor x y
      | Expr.Bit_xor -> Int32.logxor x y
      | Expr.Shl -> Int32.shift_left x (Int32.to_int y land 31)
      | Expr.Shr -> Int32.shift_right_logical x (Int32.to_int y land 31))
  | Dtype.I64 -> (
      match op with
      | Expr.Add -> Int64.add x y
      | Expr.Sub -> Int64.sub x y
      | Expr.Mul -> Int64.mul x y
      | Expr.Div -> Int64.div x y
      | Expr.Min -> if Int64.compare x y < 0 then x else y
      | Expr.Max -> if Int64.compare x y > 0 then x else y
      | Expr.Bit_and -> Int64.logand x y
      | Expr.Bit_or -> Int64.logor x y
      | Expr.Bit_xor -> Int64.logxor x y
      | Expr.Shl -> Int64.shift_left x (Int64.to_int y land 63)
      | Expr.Shr -> Int64.shift_right_logical x (Int64.to_int y land 63))
  | Dtype.Bool -> invalid_arg "Backend_interp: binop on bool"

(* The inverse error function, which no stdlib provides. Giles' polynomial
   (single precision, relative error about 1e-7) supplies the starting
   point; two Newton steps on [Float.erf] take that to full double
   precision, since the iteration squares the error each time. *)
let two_over_sqrt_pi = 2.0 /. Stdlib.sqrt Float.pi

let erfinv_guess x =
  let w = -.Stdlib.log ((1.0 -. x) *. (1.0 +. x)) in
  let p =
    if w < 5.0 then (
      let w = w -. 2.5 in
      let p = 2.81022636e-08 in
      let p = 3.43273939e-07 +. (p *. w) in
      let p = -3.5233877e-06 +. (p *. w) in
      let p = -4.39150654e-06 +. (p *. w) in
      let p = 0.00021858087 +. (p *. w) in
      let p = -0.00125372503 +. (p *. w) in
      let p = -0.00417768164 +. (p *. w) in
      let p = 0.246640727 +. (p *. w) in
      1.50140941 +. (p *. w))
    else
      let w = Stdlib.sqrt w -. 3.0 in
      let p = -0.000200214257 in
      let p = 0.000100950558 +. (p *. w) in
      let p = 0.00134934322 +. (p *. w) in
      let p = -0.00367342844 +. (p *. w) in
      let p = 0.00573950773 +. (p *. w) in
      let p = -0.0076224613 +. (p *. w) in
      let p = 0.00943887047 +. (p *. w) in
      let p = 1.00167406 +. (p *. w) in
      2.83297682 +. (p *. w)
  in
  p *. x

let erfinv x =
  if Float.is_nan x then x
  else if x = 1.0 then Float.infinity
  else if x = -1.0 then Float.neg_infinity
  else if Float.abs x > 1.0 then Float.nan
  else if x = 0.0 then x
  else begin
    let y = ref (erfinv_guess x) in
    for _ = 1 to 2 do
      let d = two_over_sqrt_pi *. Stdlib.exp (-.(!y *. !y)) in
      (* [d] underflows to zero only where [erf] has already saturated and
         the guess is the best answer there is. *)
      if d <> 0.0 then y := !y -. ((Float.erf !y -. x) /. d)
    done;
    !y
  end

let float_unop op x =
  match op with
  | Expr.Neg -> -.x
  | Expr.Sqrt -> Stdlib.sqrt x
  | Expr.Exp -> Stdlib.exp x
  | Expr.Log -> Stdlib.log x
  | Expr.Abs -> Float.abs x
  | Expr.Sin -> Stdlib.sin x
  | Expr.Cos -> Stdlib.cos x
  | Expr.Erf -> Float.erf x
  | Expr.Erfinv -> erfinv x

let unop : type a. a Dtype.t -> Expr.unop -> a -> a =
 fun d op x ->
  match d with
  | Dtype.F32 -> round_f32 (float_unop op x)
  | Dtype.F64 -> float_unop op x
  | Dtype.I32 -> (
      match op with
      | Expr.Neg -> Int32.neg x
      | Expr.Abs -> Int32.abs x
      | Expr.Sqrt -> invalid_arg "Backend_interp: sqrt on i32"
      | Expr.Exp -> invalid_arg "Backend_interp: exp on i32"
      | Expr.Log -> invalid_arg "Backend_interp: log on i32"
      | Expr.Sin -> invalid_arg "Backend_interp: sin on i32"
      | Expr.Cos -> invalid_arg "Backend_interp: cos on i32"
      | Expr.Erf -> invalid_arg "Backend_interp: erf on i32"
      | Expr.Erfinv -> invalid_arg "Backend_interp: erfinv on i32")
  | Dtype.I64 -> (
      match op with
      | Expr.Neg -> Int64.neg x
      | Expr.Abs -> Int64.abs x
      | Expr.Sqrt -> invalid_arg "Backend_interp: sqrt on i64"
      | Expr.Exp -> invalid_arg "Backend_interp: exp on i64"
      | Expr.Log -> invalid_arg "Backend_interp: log on i64"
      | Expr.Sin -> invalid_arg "Backend_interp: sin on i64"
      | Expr.Cos -> invalid_arg "Backend_interp: cos on i64"
      | Expr.Erf -> invalid_arg "Backend_interp: erf on i64"
      | Expr.Erfinv -> invalid_arg "Backend_interp: erfinv on i64")
  | Dtype.Bool -> invalid_arg "Backend_interp: unop on bool"

(* Polymorphic comparison gives the C/IEEE NaN behaviour the oracle needs:
   [nan < x] is false, [nan <> nan] is true. *)
let cmp : type a. Expr.cmp -> a -> a -> bool =
 fun op x y ->
  match op with
  | Expr.Eq -> x = y
  | Expr.Ne -> x <> y
  | Expr.Lt -> x < y
  | Expr.Le -> x <= y
  | Expr.Gt -> x > y
  | Expr.Ge -> x >= y

let of_float : type b. b Dtype.t -> float -> b =
 fun d x ->
  match d with
  | Dtype.F32 -> round_f32 x
  | Dtype.F64 -> x
  | Dtype.I32 -> Int64.to_int32 (Int64.of_float x) (* truncates toward zero *)
  | Dtype.I64 -> Int64.of_float x
  | Dtype.Bool -> x <> 0.0

let of_int64 : type b. b Dtype.t -> int64 -> b =
 fun d x ->
  match d with
  | Dtype.F32 -> round_f32 (Int64.to_float x)
  | Dtype.F64 -> Int64.to_float x
  | Dtype.I32 -> Int64.to_int32 x
  | Dtype.I64 -> x
  | Dtype.Bool -> Int64.compare x 0L <> 0

let of_bool : type b. b Dtype.t -> bool -> b =
 fun d x ->
  match d with
  | Dtype.F32 -> if x then 1.0 else 0.0
  | Dtype.F64 -> if x then 1.0 else 0.0
  | Dtype.I32 -> if x then 1l else 0l
  | Dtype.I64 -> if x then 1L else 0L
  | Dtype.Bool -> x

(* Floats go through [float], integers through [Int64]: one intermediate per
   family keeps the number of cases linear instead of quadratic. *)
let cast : type a b. from:a Dtype.t -> to_:b Dtype.t -> a -> b =
 fun ~from ~to_ x ->
  match Dtype.equal from to_ with
  | Some Dtype.Equal -> x
  | None -> (
      match from with
      | Dtype.F32 -> of_float to_ x
      | Dtype.F64 -> of_float to_ x
      | Dtype.I32 -> of_int64 to_ (Int64.of_int32 x)
      | Dtype.I64 -> of_int64 to_ x
      | Dtype.Bool -> of_bool to_ x)

(* ------------------------------------------------------------------ *)
(* Expression evaluation                                                *)
(* ------------------------------------------------------------------ *)

let rec eval : type a. binding list -> index:int -> a Expr.t -> a =
 fun env ~index e ->
  match e.node with
  | Expr.Const v -> v
  | Expr.Arg i -> lookup env i e.dtype
  | Expr.Index -> Int32.of_int index
  | Expr.Binop (op, x, y) -> binop e.dtype op (eval env ~index x) (eval env ~index y)
  | Expr.Unop (op, x) -> unop e.dtype op (eval env ~index x)
  | Expr.Cmp (op, x, y) -> cmp op (eval env ~index x) (eval env ~index y)
  | Expr.Logic (Expr.And, x, y) -> eval env ~index x && eval env ~index y
  | Expr.Logic (Expr.Or, x, y) -> eval env ~index x || eval env ~index y
  | Expr.Not x -> not (eval env ~index x)
  | Expr.Select (c, x, y) ->
      if eval env ~index c then eval env ~index x else eval env ~index y
  | Expr.Cast (x, d) -> cast ~from:x.dtype ~to_:d (eval env ~index x)

(* ------------------------------------------------------------------ *)
(* Node evaluation                                                      *)
(* ------------------------------------------------------------------ *)

type memo = (int, Value.packed) Hashtbl.t
type inputs = (string * Value.packed) list

(* The extent of the last axis: [Reduce] and [Scan] walk one row of this
   length at a time. A rank-0 shape has a single row of one element, which
   is the only reading that keeps [numel = rows * row_length]. *)
let row_length shape =
  match List.rev (Shape.dims shape) with [] -> 1 | n :: _ -> n

let input_of : type a. inputs -> string -> a Tensor.t -> a Value.t =
 fun inputs pname t ->
  match List.assoc_opt pname inputs with
  | None -> invalid_arg (Printf.sprintf "Backend_interp: missing input %S" pname)
  | Some (Value.P v) -> (
      match Dtype.equal t.dtype (Value.dtype v) with
      | None ->
          invalid_arg
            (Printf.sprintf "Backend_interp: input %S has dtype %s, expected %s" pname
               (Dtype.name (Value.dtype v))
               (Dtype.name t.dtype))
      | Some Dtype.Equal ->
          if not (Shape.equal t.shape (Value.shape v)) then
            invalid_arg
              (Printf.sprintf "Backend_interp: input %S has shape %s, expected %s" pname
                 (Shape.to_string (Value.shape v))
                 (Shape.to_string t.shape));
          (* The input value itself: inputs are never copied and never
             mutated. *)
          v)

(* Recursive and memoised rather than driven by [Graph.topological_order]:
   dependencies are respected by construction and unreachable nodes are never
   evaluated. The memo is keyed by [Uid.to_int]. *)
let rec eval_node : type a. memo -> inputs -> a Tensor.t -> a Value.t =
 fun memo inputs t ->
  let key = Uid.to_int t.uid in
  match Hashtbl.find_opt memo key with
  | Some (Value.P v) -> (
      match Dtype.equal t.dtype (Value.dtype v) with
      | Some Dtype.Equal -> v
      | None -> invalid_arg "Backend_interp: memo dtype mismatch")
  | None ->
      let v = compute memo inputs t in
      Hashtbl.replace memo key (Value.P v);
      v

and compute : type a. memo -> inputs -> a Tensor.t -> a Value.t =
 fun memo inputs t ->
  match t.node with
  | Tensor.Param pname -> input_of inputs pname t
  | Tensor.Iota ->
      let out = Value.create Dtype.I32 t.shape in
      for i = 0 to Value.numel out - 1 do
        Value.set out i (Int32.of_int i)
      done;
      out
  | Tensor.Map (fn, src) ->
      let vs = eval_node memo inputs src in
      let out = Value.create t.dtype t.shape in
      for i = 0 to Value.numel out - 1 do
        Value.set out i (eval [ B (src.dtype, Value.get vs i) ] ~index:i fn.body1)
      done;
      out
  | Tensor.Map2 (fn, a, b) ->
      let va = eval_node memo inputs a in
      let vb = eval_node memo inputs b in
      let out = Value.create t.dtype t.shape in
      for i = 0 to Value.numel out - 1 do
        let env = [ B (a.dtype, Value.get va i); B (b.dtype, Value.get vb i) ] in
        Value.set out i (eval env ~index:i fn.body2)
      done;
      out
  | Tensor.Reduce (fn, init, src) ->
      (* Along the last axis: one strictly sequential left fold per row, in
         index order. The row count comes from the OUTPUT shape, which [Dsl]
         computed; an empty row yields [init]. [Index] stays the flat source
         index, as it is everywhere else. *)
      let vs = eval_node memo inputs src in
      let n = row_length src.shape in
      let out = Value.create t.dtype t.shape in
      let rows = Value.numel out in
      for r = 0 to rows - 1 do
        let acc = ref (eval [] ~index:0 init) in
        for j = 0 to n - 1 do
          let i = (r * n) + j in
          let env = [ B (src.dtype, !acc); B (src.dtype, Value.get vs i) ] in
          acc := eval env ~index:i fn.body2
        done;
        Value.set out r !acc
      done;
      out
  | Tensor.Scan (fn, init, src) ->
      (* Inclusive, and per row: the accumulator is stored after each step
         and restarts from [init] at the head of every row. *)
      let vs = eval_node memo inputs src in
      let out = Value.create t.dtype t.shape in
      let n = row_length src.shape in
      let rows = if n = 0 then 0 else Value.numel vs / n in
      for r = 0 to rows - 1 do
        let acc = ref (eval [] ~index:0 init) in
        for j = 0 to n - 1 do
          let i = (r * n) + j in
          let env = [ B (src.dtype, !acc); B (src.dtype, Value.get vs i) ] in
          acc := eval env ~index:i fn.body2;
          Value.set out i !acc
        done
      done;
      out
  | Tensor.Gather (idx, src) ->
      let vidx = eval_node memo inputs idx in
      let vs = eval_node memo inputs src in
      let n = Value.numel vs in
      let out = Value.create t.dtype t.shape in
      for i = 0 to Value.numel out - 1 do
        let j = Int32.to_int (Value.get vidx i) in
        if j < 0 || j >= n then
          invalid_arg
            (Printf.sprintf
               "Backend_interp: gather index out of bounds: %d at %d (source has %d \
                elements)"
               j i n);
        Value.set out i (Value.get vs j)
      done;
      out
  | Tensor.Reshape (shape, src) ->
      let vs = eval_node memo inputs src in
      let out = Value.create t.dtype shape in
      for i = 0 to Value.numel out - 1 do
        Value.set out i (Value.get vs i)
      done;
      out
  | Tensor.Broadcast (shape, src) ->
      (* [Dsl.broadcast] guarantees the source has exactly one element, so
         element 0 is the only one there is to read. *)
      let vs = eval_node memo inputs src in
      let x = Value.get vs 0 in
      let out = Value.create t.dtype shape in
      for i = 0 to Value.numel out - 1 do
        Value.set out i x
      done;
      out

let run g ~inputs =
  let memo : memo = Hashtbl.create 64 in
  List.map
    (fun (out_name, p) ->
      match p with Tensor.P t -> (out_name, Value.P (eval_node memo inputs t)))
    (Graph.outputs g)
