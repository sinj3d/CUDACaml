(** Symbolic differentiation of [Expr.t].

    The whole module is structural recursion over the expression GADT. Two
    things make it slightly unusual:

    - it is polymorphic-recursive (like [Backend_interp.eval]): [Cast]
      recurses at a different element type, so every walker is annotated
      [type t. t Expr.t -> ...];
    - building a literal at an unknown element type [t] needs a witness
      that [t = float], which is what the match on [Dtype.F32 | F64] in
      [fconst] provides. Every other dtype raises, which is exactly the
      contract [d] advertises. *)

open Ocaml_cuda_ir

(* ------------------------------------------------------------------ *)
(* Literals at an element type                                          *)
(* ------------------------------------------------------------------ *)

(* The only way to get from a [float] to an ['a Expr.t]: matching the dtype
   refines ['a] to [float] in the two float arms. The integer arms are the
   single place where "not differentiable" is decided. *)
let fconst : type t. t Dtype.t -> float -> t Expr.t =
 fun dt v ->
  match dt with
  | Dtype.F32 -> Dsl.const Dtype.F32 v
  | Dtype.F64 -> Dsl.const Dtype.F64 v
  | Dtype.I32 | Dtype.I64 | Dtype.Bool ->
      invalid_arg
        (Printf.sprintf "Deriv: dtype %s is not differentiable" (Dtype.name dt))

let two_over_sqrt_pi = 2.0 /. Float.sqrt Float.pi
let sqrt_pi_over_two = Float.sqrt Float.pi /. 2.0

(* ------------------------------------------------------------------ *)
(* Simplification                                                       *)
(* ------------------------------------------------------------------ *)

(* Same rounding rule as the interpreter, duplicated on purpose: this
   library sits at layer 2 and must not depend on a backend. *)
let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Folding is arithmetic only. [Min]/[Max] are left alone (their NaN
   behaviour is the backend's business) and so is a division by a zero
   constant, which must keep producing inf/nan at run time. *)
let fold_arith op (x : float) (y : float) : float option =
  match op with
  | Expr.Add -> Some (x +. y)
  | Expr.Sub -> Some (x -. y)
  | Expr.Mul -> Some (x *. y)
  | Expr.Div -> if y = 0.0 then None else Some (x /. y)
  | Expr.Min | Expr.Max -> None
  | Expr.Bit_and | Expr.Bit_or | Expr.Bit_xor | Expr.Shl | Expr.Shr -> None

let fold_binop : type t. t Dtype.t -> Expr.binop -> t -> t -> t option =
 fun dt op x y ->
  match dt with
  | Dtype.F32 -> (
      match fold_arith op x y with Some r -> Some (round_f32 r) | None -> None)
  | Dtype.F64 -> fold_arith op x y
  | Dtype.I32 | Dtype.I64 | Dtype.Bool -> None

(* [is_lit e v]: [e] is a float constant equal to [v]. Integer constants are
   never matched, so the algebraic rewrites below only ever fire on the
   float arithmetic they were stated for. *)
let is_lit : type t. t Expr.t -> float -> bool =
 fun e v ->
  match e.Expr.node with
  | Expr.Const c -> (
      match e.Expr.dtype with
      | Dtype.F32 -> c = v
      | Dtype.F64 -> c = v
      | Dtype.I32 | Dtype.I64 | Dtype.Bool -> false)
  | Expr.Arg _ | Expr.Index | Expr.Binop _ | Expr.Unop _ | Expr.Cmp _
  | Expr.Logic _ | Expr.Not _ | Expr.Select _ | Expr.Cast _ ->
      false

let rec simplify : type t. t Expr.t -> t Expr.t =
 fun e ->
  let dt = e.Expr.dtype in
  match e.Expr.node with
  | Expr.Const _ | Expr.Arg _ | Expr.Index -> e
  | Expr.Binop (op, a0, b0) ->
      let a = simplify a0 and b = simplify b0 in
      let keep () =
        if a == a0 && b == b0 then e else Expr.make dt (Expr.Binop (op, a, b))
      in
      let folded =
        match (a.Expr.node, b.Expr.node) with
        | Expr.Const x, Expr.Const y -> fold_binop dt op x y
        | _ -> None
      in
      (match folded with
      | Some r -> Expr.make dt (Expr.Const r)
      | None -> (
          match op with
          | Expr.Mul ->
              if is_lit b 0.0 then b
              else if is_lit a 0.0 then a
              else if is_lit b 1.0 then a
              else if is_lit a 1.0 then b
              else keep ()
          | Expr.Add ->
              if is_lit b 0.0 then a else if is_lit a 0.0 then b else keep ()
          | Expr.Sub ->
              if is_lit b 0.0 then a
              else if is_lit a 0.0 then Expr.make dt (Expr.Unop (Expr.Neg, b))
              else keep ()
          | Expr.Div | Expr.Min | Expr.Max | Expr.Bit_and | Expr.Bit_or
          | Expr.Bit_xor | Expr.Shl | Expr.Shr ->
              keep ()))
  | Expr.Unop (op, a0) -> (
      let a = simplify a0 in
      let keep () =
        if a == a0 then e else Expr.make dt (Expr.Unop (op, a))
      in
      match (op, a.Expr.node) with
      | Expr.Neg, Expr.Unop (Expr.Neg, inner) -> inner
      | ( ( Expr.Neg | Expr.Sqrt | Expr.Exp | Expr.Log | Expr.Abs | Expr.Sin
          | Expr.Cos | Expr.Erf | Expr.Erfinv ),
          _ ) ->
          keep ())
  | Expr.Cmp (op, a0, b0) ->
      let a = simplify a0 and b = simplify b0 in
      if a == a0 && b == b0 then e else Expr.make dt (Expr.Cmp (op, a, b))
  | Expr.Logic (op, a0, b0) ->
      let a = simplify a0 and b = simplify b0 in
      if a == a0 && b == b0 then e else Expr.make dt (Expr.Logic (op, a, b))
  | Expr.Not a0 ->
      let a = simplify a0 in
      if a == a0 then e else Expr.make dt (Expr.Not a)
  | Expr.Select (c0, a0, b0) ->
      let c = simplify c0 and a = simplify a0 and b = simplify b0 in
      if Uid.equal a.Expr.uid b.Expr.uid then a
      else if c == c0 && a == a0 && b == b0 then e
      else Expr.make dt (Expr.Select (c, a, b))
  | Expr.Cast (x0, target) -> (
      let x = simplify x0 in
      match Dtype.equal x.Expr.dtype target with
      | Some Dtype.Equal -> x
      | None -> if x == x0 then e else Expr.make dt (Expr.Cast (x, target)))

(* ------------------------------------------------------------------ *)
(* Differentiation                                                      *)
(* ------------------------------------------------------------------ *)

(* Kink conventions (Min, Max, Abs, Select): the derivative is that of the
   branch taken at the point, ties going to the first operand. That is a
   subgradient, and it is what pathwise Greeks use. *)
let rec deriv : type t. t Expr.t -> wrt:int -> t Expr.t =
 fun e ~wrt ->
  let dt = e.Expr.dtype in
  match e.Expr.node with
  | Expr.Const _ -> fconst dt 0.0
  | Expr.Arg j -> fconst dt (if j = wrt then 1.0 else 0.0)
  (* [Index] is I32, so [fconst] rejects it. It is unreachable from a float
     root except under a [Cast], which never recurses into a non-float. *)
  | Expr.Index -> fconst dt 0.0
  | Expr.Binop (op, a, b) -> (
      match op with
      | Expr.Add -> Dsl.add (deriv a ~wrt) (deriv b ~wrt)
      | Expr.Sub -> Dsl.sub (deriv a ~wrt) (deriv b ~wrt)
      | Expr.Mul ->
          Dsl.add (Dsl.mul (deriv a ~wrt) b) (Dsl.mul a (deriv b ~wrt))
      | Expr.Div ->
          Dsl.div
            (Dsl.sub (Dsl.mul (deriv a ~wrt) b) (Dsl.mul a (deriv b ~wrt)))
            (Dsl.mul b b)
      | Expr.Min -> Dsl.select (Dsl.le a b) (deriv a ~wrt) (deriv b ~wrt)
      | Expr.Max -> Dsl.select (Dsl.ge a b) (deriv a ~wrt) (deriv b ~wrt)
      | Expr.Bit_and -> invalid_arg "Deriv.d: bit_and is integer only"
      | Expr.Bit_or -> invalid_arg "Deriv.d: bit_or is integer only"
      | Expr.Bit_xor -> invalid_arg "Deriv.d: bit_xor is integer only"
      | Expr.Shl -> invalid_arg "Deriv.d: shl is integer only"
      | Expr.Shr -> invalid_arg "Deriv.d: shr is integer only")
  | Expr.Unop (op, a) -> (
      let da = deriv a ~wrt in
      match op with
      | Expr.Neg -> Dsl.neg da
      | Expr.Sqrt -> Dsl.div da (Dsl.mul (fconst dt 2.0) (Dsl.sqrt a))
      | Expr.Exp -> Dsl.mul (Dsl.exp a) da
      | Expr.Log -> Dsl.div da a
      | Expr.Abs -> Dsl.select (Dsl.lt a (fconst dt 0.0)) (Dsl.neg da) da
      | Expr.Sin -> Dsl.mul (Dsl.cos a) da
      | Expr.Cos -> Dsl.mul (Dsl.neg (Dsl.sin a)) da
      | Expr.Erf ->
          (* d/da erf a = (2/sqrt pi) * exp (-a^2) *)
          Dsl.mul
            (Dsl.mul
               (fconst dt two_over_sqrt_pi)
               (Dsl.exp (Dsl.neg (Dsl.mul a a))))
            da
      | Expr.Erfinv ->
          (* the reciprocal of erf' at the image point *)
          let y = Dsl.erfinv a in
          Dsl.mul
            (Dsl.mul (fconst dt sqrt_pi_over_two) (Dsl.exp (Dsl.mul y y)))
            da)
  (* Bool-valued nodes: the root check in [d] rejects them, and inside a
     [Select] the condition is not differentiated. *)
  | Expr.Cmp _ -> invalid_arg "Deriv.d: a comparison has dtype bool"
  | Expr.Logic _ -> invalid_arg "Deriv.d: a logical operator has dtype bool"
  | Expr.Not _ -> invalid_arg "Deriv.d: a negation has dtype bool"
  | Expr.Select (c, a, b) -> Dsl.select c (deriv a ~wrt) (deriv b ~wrt)
  | Expr.Cast (x, target) ->
      if Dtype.is_float x.Expr.dtype then Dsl.cast target (deriv x ~wrt)
      else fconst dt 0.0

let d e ~wrt =
  if not (Dtype.is_float e.Expr.dtype) then
    invalid_arg
      (Printf.sprintf "Deriv.d: expression has dtype %s, a float dtype is required"
         (Dtype.name e.Expr.dtype));
  simplify (deriv e ~wrt)

let fn1 (f : ('a, 'a) Expr.fn1) : ('a, 'a) Expr.fn1 =
  { Expr.arg1 = f.Expr.arg1; body1 = d f.Expr.body1 ~wrt:0 }

let fn2 (f : ('a, 'a, 'a) Expr.fn2) ~wrt : ('a, 'a, 'a) Expr.fn2 =
  if wrt <> 0 && wrt <> 1 then
    invalid_arg (Printf.sprintf "Deriv.fn2: wrt must be 0 or 1, got %d" wrt);
  { Expr.arg_a = f.Expr.arg_a; arg_b = f.Expr.arg_b; body2 = d f.Expr.body2 ~wrt }

(* ------------------------------------------------------------------ *)
(* Substitution                                                         *)
(* ------------------------------------------------------------------ *)

(* Every node is rebuilt with a fresh uid, including the leaves: [Graph] and
   [Lower] key on uids, so a uid shared between two structurally different
   trees would be a silent miscompile downstream. *)
let rec subst : type t. (int * Expr.packed) list -> t Expr.t -> t Expr.t =
 fun args e ->
  let dt = e.Expr.dtype in
  let go : type u. u Expr.t -> u Expr.t = fun x -> subst args x in
  match e.Expr.node with
  | Expr.Arg i -> (
      match List.assoc_opt i args with
      | None -> Expr.make dt (Expr.Arg i)
      | Some (Expr.P s) -> (
          match Dtype.equal dt s.Expr.dtype with
          | Some Dtype.Equal -> s
          | None ->
              invalid_arg
                (Printf.sprintf
                   "Deriv.apply: Arg %d has dtype %s but the substitute has dtype %s"
                   i (Dtype.name dt) (Dtype.name s.Expr.dtype))))
  | Expr.Const c -> Expr.make dt (Expr.Const c)
  | Expr.Index -> Expr.make dt Expr.Index
  | Expr.Binop (op, a, b) -> Expr.make dt (Expr.Binop (op, go a, go b))
  | Expr.Unop (op, a) -> Expr.make dt (Expr.Unop (op, go a))
  | Expr.Cmp (op, a, b) -> Expr.make dt (Expr.Cmp (op, go a, go b))
  | Expr.Logic (op, a, b) -> Expr.make dt (Expr.Logic (op, go a, go b))
  | Expr.Not a -> Expr.make dt (Expr.Not (go a))
  | Expr.Select (c, a, b) -> Expr.make dt (Expr.Select (go c, go a, go b))
  | Expr.Cast (x, target) -> Expr.make dt (Expr.Cast (go x, target))

let check_arg what (a : _ Expr.t) (x : _ Expr.t) =
  match Dtype.equal a.Expr.dtype x.Expr.dtype with
  | Some Dtype.Equal -> ()
  | None ->
      invalid_arg
        (Printf.sprintf "Deriv.%s: argument has dtype %s, substitute has dtype %s"
           what (Dtype.name a.Expr.dtype) (Dtype.name x.Expr.dtype))

let apply1 (f : ('a, 'b) Expr.fn1) (x : 'a Expr.t) : 'b Expr.t =
  check_arg "apply1" f.Expr.arg1 x;
  subst [ (0, Expr.P x) ] f.Expr.body1

let apply2 (f : ('a, 'b, 'c) Expr.fn2) (x : 'a Expr.t) (y : 'b Expr.t) : 'c Expr.t =
  check_arg "apply2" f.Expr.arg_a x;
  check_arg "apply2" f.Expr.arg_b y;
  subst [ (0, Expr.P x); (1, Expr.P y) ] f.Expr.body2
