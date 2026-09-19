# T03 — `Backend_interp`: the reference evaluator

## Goal

Implement `lib/backend_interp/ocaml_cuda_backend_interp.ml`: a pure-OCaml
evaluator for `Graph.t` that walks the topological order and computes every
tensor element by element on `Value.t`s. It is the **oracle** every other
backend and every optimisation is checked against, so correctness and
faithfulness to C/CUDA semantics matter; speed does not.

Depends on: T01, T02.

## Files you own

- `lib/backend_interp/ocaml_cuda_backend_interp.ml` — replace the stub.

Do not edit the `.mli` or anything in `lib/ir`.

## Interfaces

Contract (`lib/backend_interp/ocaml_cuda_backend_interp.mli`):
`include Ocaml_cuda_backend.Backend.S`, which is:

```ocaml
val name : string
type compiled
val compile : Graph.t -> compiled
val run : compiled -> inputs:(string * Value.packed) list -> (string * Value.packed) list
```

Keep `type compiled = Graph.t` and `let compile g = g` (already in the stub).

What you evaluate (`lib/ir/expr.ml`):

```ocaml
type binop = Add | Sub | Mul | Div | Min | Max
type unop = Neg | Sqrt | Exp | Log | Abs
type cmp = Eq | Ne | Lt | Le | Gt | Ge
type logic = And | Or
type 'a t = { uid : Uid.t; dtype : 'a Dtype.t; node : 'a node }
and _ node =
  | Const : 'a -> 'a node
  | Arg : int -> 'a node          (* i-th argument of the element function *)
  | Index : int32 node            (* flat index of the element being computed *)
  | Binop : binop * 'a t * 'a t -> 'a node
  | Unop : unop * 'a t -> 'a node
  | Cmp : cmp * 'b t * 'b t -> bool node
  | Logic : logic * bool t * bool t -> bool node
  | Not : bool t -> bool node
  | Select : bool t * 'a t * 'a t -> 'a node
  | Cast : 'b t * 'a Dtype.t -> 'a node
type ('a, 'b) fn1 = { arg1 : 'a t; body1 : 'b t }          (* arg1.node = Arg 0 *)
type ('a, 'b, 'c) fn2 = { arg_a : 'a t; arg_b : 'b t; body2 : 'c t }  (* Arg 0, Arg 1 *)
```

and `lib/ir/tensor.ml` (see T02 for the full `node` type). Helpers you
have: `Dtype.equal : 'a Dtype.t -> 'b Dtype.t -> ('a, 'b) Dtype.eq option`
(returns `Some Equal` when the dtypes are the same constructor),
`Dtype.is_float`, `Value.create/get/set/numel/shape/dtype`,
`Graph.topological_order/params/outputs`, `Shape.equal/numel`.

## Implementation

### Environment for element functions

`Arg i` values have different types, so the environment is a list of
packed typed values and lookup uses the equality witness:

```ocaml
type binding = B : 'a Dtype.t * 'a -> binding

let lookup : type a. binding list -> int -> a Dtype.t -> a =
 fun env i want ->
  match List.nth env i with
  | B (have, v) -> ( match Dtype.equal want have with Some Dtype.Equal -> v | None -> failwith "Arg type mismatch")
```

### Scalar operations, one function per family, all `type a.`

```ocaml
let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let binop : type a. a Dtype.t -> Expr.binop -> a -> a -> a = fun d op x y ->
  match d with
  | Dtype.F32 -> round_f32 (float_binop op x y)
  | Dtype.F64 -> float_binop op x y
  | Dtype.I32 -> (match op with Add -> Int32.add x y | Sub -> Int32.sub x y | Mul -> Int32.mul x y
                  | Div -> Int32.div x y | Min -> if x < y then x else y | Max -> if x > y then x else y)
  | Dtype.I64 -> (* same with Int64 *)
  | Dtype.Bool -> invalid_arg "binop on bool"
```
where `float_binop` is `+. -. *. /. Float.min Float.max`. **F32 results are
rounded through `round_f32` after every operation** so the interpreter
matches single-precision device arithmetic. `Int32.div` truncates toward
zero like C; do not use `Int32.rem`-based flooring. Integer division by
zero raises `Division_by_zero`; let it.

`unop`: `Neg` for all numeric types (`Int32.neg` etc.); `Sqrt/Exp/Log` only
for F32/F64 (`invalid_arg` for ints); `Abs` for all numeric. Round F32.

`cmp`: use OCaml's polymorphic `<`, `<=`, `=`, `<>` on the two values.
This gives C semantics for NaN (`nan < x` is false, `nan <> nan` is true).

`cast : type a b. from:a Dtype.t -> to_:b Dtype.t -> a -> b`: go through an
intermediate: floats → `float`, ints → `Int64`. float→int truncates
toward zero (`Int64.of_float`, then `Int64.to_int32` for I32); int→float
uses `Int64.to_float` then `round_f32` for F32; bool→numeric is 1/0;
numeric→bool is `<> 0`. Same-type cast is identity.

### Expression evaluation

```ocaml
let rec eval : type a. binding list -> index:int -> a Expr.t -> a = fun env ~index e ->
  match e.node with
  | Expr.Const v -> v
  | Expr.Arg i -> lookup env i e.dtype
  | Expr.Index -> Int32.of_int index
  | Expr.Binop (op, x, y) -> binop e.dtype op (eval env ~index x) (eval env ~index y)
  | Expr.Unop (op, x) -> unop e.dtype op (eval env ~index x)
  | Expr.Cmp (op, x, y) -> cmp op (eval env ~index x) (eval env ~index y)
  | Expr.Logic (And, x, y) -> eval env ~index x && eval env ~index y   (* Or likewise *)
  | Expr.Not x -> not (eval env ~index x)
  | Expr.Select (c, x, y) -> if eval env ~index c then eval env ~index x else eval env ~index y
  | Expr.Cast (x, d) -> cast ~from:x.dtype ~to_:d (eval env ~index x)
```

### Node evaluation with memoisation

```ocaml
let eval_node : type a. memo -> inputs -> a Tensor.t -> a Value.t
```
where `memo : (int, Value.packed) Hashtbl.t` keyed by `Uid.to_int`. On a
hit, unpack with `Dtype.equal` against `t.dtype` to recover the typed
value. On a miss compute, store, return:

- `Param name` → `List.assoc_opt name inputs`; missing → `invalid_arg`.
  Check `Dtype.equal` (else `invalid_arg "dtype"`) and `Shape.equal`
  (else `invalid_arg "shape"`). Return the input value itself (no copy).
- `Iota` → `Value.create I32 shape`, element i = `Int32.of_int i`.
- `Map (fn, a)` → `va = eval_node a`; out = `Value.create t.dtype t.shape`;
  for each i: `set out i (eval [B (a.dtype, get va i)] ~index:i fn.body1)`.
- `Map2 (fn, a, b)` → env `[B (a.dtype, get va i); B (b.dtype, get vb i)]`.
- `Reduce (fn, init, src)` → `acc = eval [] ~index:0 init`; for i in
  0..n-1 **in order**: `acc <- eval [B (d, acc); B (d, get vs i)] ~index:i fn.body2`.
  Output is `Value.create d Shape.scalar` with element 0 = acc. An empty
  source yields `init`.
- `Scan` → same loop, but store `acc` into out[i] after each step
  (inclusive scan; same shape as source).
- `Gather (idx, src)` → out shape = idx shape; `j = Int32.to_int (get vidx i)`;
  if `j < 0 || j >= numel src` → `invalid_arg "gather index out of bounds"`;
  out[i] = src[j].
- `Reshape (shape, a)` → copy all elements into a new value of `shape`.

### `run`

```ocaml
let run g ~inputs =
  let memo = Hashtbl.create 64 in
  List.map (fun (name, Tensor.P t) -> (name, Value.P (eval_node memo inputs t))) (Graph.outputs g)
```
Outputs in `Graph.outputs` order. Shared nodes are computed once thanks to
`memo`. Do **not** call `Graph.topological_order` to drive evaluation — a
recursive `eval_node` with memo already respects dependencies and avoids
evaluating unreachable nodes.

## Invariants

- Runs the graph **exactly as written**: no fusion, no reordering of
  reductions, no passes. Sequential left fold for `Reduce`/`Scan`.
- F32 arithmetic is rounded after every op. This is what makes the
  `0.1 + 0.2 = 0.300000011920929` test pass.
- Never mutates an input `Value`.

## Failure modes to avoid

- Omitting `type a.` on `binop`/`unop`/`cast`/`eval`/`eval_node`: GADT
  refinement will not work and you will get "this expression has type
  float but an expression was expected of type a".
- Or-patterns across GADT constructors (`| Dtype.F32 | Dtype.F64 -> ...`)
  do not refine the type; write separate branches.
- Using `List.assoc` on `inputs` with an exception instead of a clear
  `invalid_arg` (tests check an exception is raised; a clear message
  helps humans).
- Computing `Iota` as `float` or forgetting `Int32.of_int`.
- Skipping the F32 rounding on `cast` and `unop`.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_interp.exe
```

Expected: `19 tests, 0 failures`.

## Tests (already written: `test/unit/test_interp.ml`)

map; map2 over a fused chain; iota+cast; `index ()` inside a map; reduce
sum from a non-zero init; reduce max; reduce of empty = init; inclusive
scan; gather; gather OOB raises; reshape keeps data and changes shape;
select with comparison; i32 division truncates toward zero (`-7/2 = -3`);
f32 rounding (`0.1 + 0.2`); shared intermediate evaluated once and outputs
ordered; missing input raises; wrong-shape input raises; wrong-dtype input
raises; every program in `examples/programs.ml` runs without exception.
