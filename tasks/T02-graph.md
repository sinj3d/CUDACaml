# T02 — `Graph` and `Dsl` validation

## Goal

Implement `lib/ir/graph.ml` (program container, topological order, fan-out,
Graphviz dump) and add shape validation to two functions in `lib/ir/dsl.ml`.
Every later layer walks `Graph.topological_order` and asks `Graph.fan_out`,
so these must be correct and deterministic.

Depends on: T01 (tests use `Value` indirectly through examples).

## Files you own

- `lib/ir/graph.ml` — replace the stub.
- `lib/ir/dsl.ml` — edit **only** `map2` and `reshape` (add checks).

Do not edit any `.mli`, `tensor.ml`, `expr.ml`.

## Interfaces

`lib/ir/graph.mli` (verbatim; implement all of it):

```ocaml
type t
val create : name:string -> outputs:(string * Tensor.packed) list -> t
val name : t -> string
val params : t -> (string * Tensor.packed) list
val outputs : t -> (string * Tensor.packed) list
val topological_order : t -> Tensor.packed list   (* deps before dependents; stable *)
val iter : t -> f:(Tensor.packed -> unit) -> unit
val fold : t -> init:'acc -> f:('acc -> Tensor.packed -> 'acc) -> 'acc
val fan_out : t -> Tensor.packed -> int            (* number of USES (edges), not consumers *)
val to_dot : t -> string
```

What you traverse (`lib/ir/tensor.ml`, already implemented):

```ocaml
type 'a t = { uid : Uid.t; dtype : 'a Dtype.t; shape : Shape.t; node : 'a node }
and _ node =
  | Param : string -> 'a node
  | Iota : int32 node
  | Map : ('a, 'b) Expr.fn1 * 'a t -> 'b node
  | Map2 : ('a, 'b, 'c) Expr.fn2 * 'a t * 'b t -> 'c node
  | Reduce : ('a, 'a, 'a) Expr.fn2 * 'a Expr.t * 'a t -> 'a node
  | Scan : ('a, 'a, 'a) Expr.fn2 * 'a Expr.t * 'a t -> 'a node
  | Gather : int32 t * 'a t -> 'a node
  | Reshape : Shape.t * 'a t -> 'a node
type packed = P : _ t -> packed
val uid : packed -> Uid.t
val shape : packed -> Shape.t
val deps : packed -> packed list      (* direct inputs, in argument order *)
```

`Uid.to_int : Uid.t -> int`, `Uid.to_string : Uid.t -> string` (e.g. `"n17"`).

## Implementation

Compute everything once in `create` and store it:

```ocaml
type t = {
  name : string;
  outputs : (string * Tensor.packed) list;
  params : (string * Tensor.packed) list;
  order : Tensor.packed list;
  fan_out : (int, int) Hashtbl.t;   (* keyed by Uid.to_int *)
}
```

`create ~name ~outputs`:
1. Validate: `outputs = []` → `invalid_arg`. Duplicate output names →
   `invalid_arg`.
2. Topological order by depth-first post-order:
   ```
   visited : (int, unit) Hashtbl.t ; order : packed list ref (built reversed)
   visit p =
     if not visited(uid p) then
       mark visited; List.iter visit (Tensor.deps p); order := p :: !order
   List.iter (fun (_, p) -> visit p) outputs
   order = List.rev !order
   ```
   Post-order guarantees every dep precedes its dependent. Visiting
   outputs in list order and deps in argument order makes it stable.
3. `fan_out`: for every `p` in `order`, for every `d` in `Tensor.deps p`,
   increment the count for `uid d`. A node used twice by one consumer
   (`Map2 (f, s, s)`) therefore counts 2. Nodes never used count 0
   (`Hashtbl.find_opt` → default 0 in `fan_out`).
4. `params`: filter `order` for nodes whose `node` is `Param name`, in
   order. Then check: if two entries have the same name but different
   uids → `invalid_arg "Graph.create: two Param nodes named x"`.
   To inspect the node you need to open the existential:
   ```ocaml
   let param_name (Tensor.P t) = match t.node with Tensor.Param n -> Some n | _ -> None
   ```

`iter`/`fold` walk `order`. `topological_order t = t.order`.

`to_dot t`: one line per node and one per edge, e.g.

```
digraph saxpy {
  n3 [label="n3 Param x [1024] f32"];
  n5 [label="n5 Map [1024] f32"];
  n3 -> n5;
}
```
The label must contain `Uid.to_string`, the constructor name, the shape
(`Shape.to_string`) and `Dtype.name`. Edges go from dependency to
dependent (`dep -> node`). Exact whitespace is not tested; the strings
`digraph`, the graph name, every uid, and `->` are.

`dsl.ml` edits:
- `map2`: if `not (Shape.equal a.shape b.shape)` → `invalid_arg` with both
  shapes in the message. No broadcasting.
- `reshape`: if `Shape.numel shape <> Shape.numel src.shape` → `invalid_arg`.

## Invariants

- `topological_order` contains each node exactly once, deps before
  dependents, and is identical across calls (no Hashtbl iteration order
  anywhere in the output).
- Identity is **uid only**. Never compare `Tensor.packed` values with `=`
  or hash them with `Hashtbl.hash` — they contain large trees; use
  `Uid.to_int (Tensor.uid p)` as the key everywhere.
- `create` never mutates any node.

## Failure modes to avoid

- Pre-order instead of post-order (dependents before deps).
- Using `List.mem`/`List.assoc` on packed nodes (structural comparison of
  GADT trees — slow and wrong).
- Counting fan-out over *distinct consumers* instead of edges.
- Emitting `Hashtbl.iter` output in `to_dot` (nondeterministic). Iterate
  `order` instead.
- Dedup of outputs by node: two output names may legitimately point to the
  same node; that is allowed. Only duplicate *names* are rejected.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_graph.exe
```

Expected: `13 tests, 0 failures`.

## Tests (already written: `test/unit/test_graph.ml`)

- params discovered in first-use order, deduped
- topological order: deps before dependents, no duplicates
- shared node appears once; fan_out counts uses; unused node has 0
- same node used twice by one consumer → fan_out 2
- outputs preserved in order and by name; `name` roundtrips
- two distinct Param nodes with the same name rejected
- duplicate output names rejected; empty outputs rejected
- map2 shape mismatch rejected; reshape numel mismatch rejected (and equal numel accepted)
- gather output takes index shape; reduce output is scalar (these test existing Dsl code)
- to_dot contains `digraph`, name, both uids, `->`
