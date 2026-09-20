(* Deriv. Every rule is checked against a central finite difference on
   the interpreter in F64; structural checks pin the simplifier. No GPU. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let f64 l = Value.P (Value.of_list Dtype.F64 (vec (List.length l)) l)
let k v = const Dtype.F64 v

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.F64 -> Value.to_list v | _ -> failwith "not f64")

(* f evaluated at every point *)
let eval_f f xs =
  let x = param "x" Dtype.F64 (vec (List.length xs)) in
  floats (run (Graph.create ~name:"f" ~outputs:[ ("r", Tensor.P (map f x)) ]) [ ("x", f64 xs) ]) "r"

(* f' built by Deriv.fn1, evaluated at every point *)
let eval_df f xs =
  let x = param "x" Dtype.F64 (vec (List.length xs)) in
  let fn = Expr.fn1 Dtype.F64 f in
  let dfn = Deriv.fn1 fn in
  let r = Tensor.make Dtype.F64 (vec (List.length xs)) (Tensor.Map (dfn, x)) in
  floats (run (Graph.create ~name:"df" ~outputs:[ ("r", Tensor.P r) ]) [ ("x", f64 xs) ]) "r"

let h = 1e-6

let fd f xs =
  let plus = eval_f f (List.map (fun x -> x +. h) xs) and minus = eval_f f (List.map (fun x -> x -. h) xs) in
  List.map2 (fun p m -> (p -. m) /. (2.0 *. h)) plus minus

let close ~what expect got =
  List.iteri
    (fun i (e, g) ->
      if Float.abs (e -. g) > 1e-6 *. (1.0 +. Float.abs e) then C.fail "%s[%d]: fd %g, deriv %g" what i e g)
    (List.combine expect got)

let check name f xs = C.test ("d/dx " ^ name) (fun () -> close ~what:name (fd f xs) (eval_df f xs))

let rec has_arg : type a. a Expr.t -> bool =
 fun e ->
  match e.Expr.node with
  | Expr.Arg _ -> true
  | Expr.Binop (_, a, b) -> has_arg a || has_arg b
  | Expr.Unop (_, a) -> has_arg a
  | Expr.Cmp (_, a, b) -> has_arg a || has_arg b
  | Expr.Logic (_, a, b) -> has_arg a || has_arg b
  | Expr.Not a -> has_arg a
  | Expr.Select (c, a, b) -> has_arg c || has_arg a || has_arg b
  | Expr.Cast (a, _) -> has_arg a
  | Expr.Const _ | Expr.Index -> false

let rec uids : type a. a Expr.t -> int list =
 fun e ->
  let me = Uid.to_int e.Expr.uid in
  match e.Expr.node with
  | Expr.Binop (_, a, b) -> (me :: uids a) @ uids b
  | Expr.Unop (_, a) -> me :: uids a
  | Expr.Cmp (_, a, b) -> (me :: uids a) @ uids b
  | Expr.Logic (_, a, b) -> (me :: uids a) @ uids b
  | Expr.Not a -> me :: uids a
  | Expr.Select (c, a, b) -> (me :: uids c) @ uids a @ uids b
  | Expr.Cast (a, _) -> me :: uids a
  | Expr.Const _ | Expr.Index | Expr.Arg _ -> [ me ]

let () =
  check "3x^2 - 2x + 1" (fun x -> add (sub (mul (k 3.0) (mul x x)) (mul (k 2.0) x)) (k 1.0)) [ -2.; -0.5; 0.; 1.; 3. ];
  check "x / (1 + x^2)" (fun x -> div x (add (k 1.0) (mul x x))) [ -2.; -0.5; 0.; 1.; 3. ];
  check "sqrt" sqrt [ 0.5; 1.; 4. ];
  check "exp" exp [ -1.; 0.; 2. ];
  check "log" log [ 0.5; 1.; 3. ];
  check "sin" sin [ 0.; 1.; -2. ];
  check "cos" cos [ 0.; 1.; -2. ];
  check "erf" erf [ -1.; 0.; 0.5 ];
  check "erfinv" erfinv [ -0.9; -0.3; 0.; 0.5; 0.9 ];
  check "abs (away from 0)" (fun x -> Expr.make Dtype.F64 (Expr.Unop (Expr.Abs, x))) [ -2.; -0.5; 1.; 3. ];
  check "min (x, 1)" (fun x -> min x (k 1.0)) [ -1.; 0.; 2.; 3. ];
  check "max (x, 1)" (fun x -> max x (k 1.0)) [ -1.; 0.; 2.; 3. ];
  check "select (x<0) x^2 (exp x)" (fun x -> select (lt x (k 0.0)) (mul x x) (exp x)) [ -2.; -1.; 1.; 2. ];
  check "composite exp(sin x) * log(1+x^2)" (fun x -> mul (exp (sin x)) (log (add (k 1.0) (mul x x)))) [ -1.5; 0.3; 2. ];
  C.test "fn2 partials of x*y + sin x" (fun () ->
      let f a b = add (mul a b) (sin a) in
      let fn = Expr.fn2 Dtype.F64 Dtype.F64 f in
      let xs = [ 1.0; -0.5 ] and ys = [ 2.0; 3.0 ] in
      let x = param "x" Dtype.F64 (vec 2) and y = param "y" Dtype.F64 (vec 2) in
      let dx = Tensor.make Dtype.F64 (vec 2) (Tensor.Map2 (Deriv.fn2 fn ~wrt:0, x, y)) in
      let dy = Tensor.make Dtype.F64 (vec 2) (Tensor.Map2 (Deriv.fn2 fn ~wrt:1, x, y)) in
      let o = run (Graph.create ~name:"g" ~outputs:[ ("dx", Tensor.P dx); ("dy", Tensor.P dy) ]) [ ("x", f64 xs); ("y", f64 ys) ] in
      C.floats ~tol:1e-12 ~expect:(List.map2 (fun a b -> b +. Stdlib.cos a) xs ys) (floats o "dx");
      C.floats ~tol:1e-12 ~expect:xs (floats o "dy");
      C.raises (fun () -> ignore (Deriv.fn2 fn ~wrt:2)));
  C.test "cast of an int expression has derivative 0; index * x -> index" (fun () ->
      C.floats ~tol:1e-12 ~expect:[ 0.; 1.; 2. ] (eval_df (fun x -> mul (cast Dtype.F64 (index ())) x) [ 5.; 5.; 5. ]));
  C.test "d on an integer expression raises Invalid_argument" (fun () ->
      let fn = Expr.fn1 Dtype.I32 (fun x -> mul x x) in
      C.raises (fun () -> ignore (Deriv.d fn.Expr.body1 ~wrt:0)));
  C.test "structural: constants and args" (fun () ->
      let is_const v (e : float Expr.t) = match e.Expr.node with Expr.Const c -> C.float ~tol:0.0 ~expect:v c | _ -> C.fail "not a Const" in
      is_const 0.0 (Deriv.d (k 5.0) ~wrt:0);
      is_const 1.0 (Deriv.d (Expr.make Dtype.F64 (Expr.Arg 0)) ~wrt:0);
      is_const 0.0 (Deriv.d (Expr.make Dtype.F64 (Expr.Arg 1)) ~wrt:0));
  C.test "apply1 substitutes; dtype mismatch raises; no shared uids" (fun () ->
      let fn = Expr.fn1 Dtype.F64 (fun x -> add (mul x x) (k 1.0)) in
      let r = Deriv.apply1 fn (k 2.0) in
      C.bool ~expect:false (has_arg r);
      C.raises (fun () -> ignore (Deriv.apply1 fn (const Dtype.F32 2.0)));
      let shared = List.filter (fun u -> List.mem u (uids fn.Expr.body1)) (uids (Deriv.apply1 fn (Expr.make Dtype.F64 (Expr.Arg 3)))) in
      C.int ~expect:0 (List.length shared));
  C.test "apply2 substitutes both arguments" (fun () ->
      let fn = Expr.fn2 Dtype.F64 Dtype.F64 (fun a b -> sub a b) in
      let r = Deriv.apply2 fn (k 5.0) (k 3.0) in
      C.bool ~expect:false (has_arg r);
      C.floats ~expect:[ 2.0 ] (eval_f (fun _ -> r) [ 0.0 ]));
  C.test "simplify" (fun () ->
      let x = Expr.make Dtype.F64 (Expr.Arg 0) and y = Expr.make Dtype.F64 (Expr.Arg 1) in
      (match (Deriv.simplify (add (mul x (k 0.0)) (mul y (k 1.0)))).Expr.node with
      | Expr.Arg 1 -> ()
      | _ -> C.fail "x*0 + y*1 should simplify to y");
      (match (Deriv.simplify (neg (neg x))).Expr.node with Expr.Arg 0 -> () | _ -> C.fail "neg neg x");
      (match (Deriv.simplify (add (k 2.0) (k 3.0))).Expr.node with
      | Expr.Const v -> C.float ~tol:0.0 ~expect:5.0 v
      | _ -> C.fail "2+3 should fold");
      match (Deriv.simplify (div (k 1.0) (k 0.0))).Expr.node with
      | Expr.Binop (Expr.Div, _, _) -> ()
      | _ -> C.fail "division by constant zero must not fold");
  C.run ()
