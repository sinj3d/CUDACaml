# T04 — `Fusion`: decide which tensors get a buffer

## Goal

Implement `lib/lower/fusion.ml`. Fusion in this project is an *analysis*,
not a rewrite: element-wise nodes are inlined by `Lower` (T05) into
whichever kernel consumes them, so the only decision is **which nodes are
materialised** (get their own global buffer and, unless they are Params,
their own kernel). Get this wrong and either work is duplicated across
kernels or a kernel reads a buffer nobody wrote.

Depends on: T02.

## Files you own

- `lib/lower/fusion.ml` — replace the stub.

## Interfaces

Contract (`lib/lower/fusion.mli`):

```ocaml
type plan
val plan : Graph.t -> plan
val is_materialized : plan -> Tensor.packed -> bool
val kernel_roots : plan -> Tensor.packed list
```

You need from `Graph`: `topological_order`, `outputs`, `fan_out`. From
`Tensor`: `uid`, and the `node` constructors (`Param`, `Iota`, `Map`,
`Map2`, `Reduce`, `Scan`, `Gather`, `Reshape` — see T02 for the type).
`Uid.to_int` for hashtable keys.

## Implementation

```ocaml
type plan = { materialized : (int, unit) Hashtbl.t; roots : Tensor.packed list }
```

A node is materialised iff **any** of:
1. it is a `Param` (inputs always live in a buffer);
2. it is a `Reduce` or `Scan` (fusion barriers: their result is computed
   by a whole kernel, not per element);
3. its uid appears in `Graph.outputs` (results must be downloadable);
4. `Graph.fan_out g p >= 2` (used more than once → compute once, store).

Everything else (`Iota`, `Map`, `Map2`, `Gather`, `Reshape` with fan-out
≤ 1 that is not an output) is inlined by the consumer.

`kernel_roots` = the materialised nodes in `Graph.topological_order`,
**excluding Params**. Order must be the topological order (dependencies'
kernels launch first).

To classify a node open the existential:

```ocaml
let is_barrier (Tensor.P t) = match t.node with Tensor.Reduce _ | Tensor.Scan _ -> true | _ -> false
let is_param (Tensor.P t) = match t.node with Tensor.Param _ -> true | _ -> false
```

Build a set of output uids once (`Hashtbl` keyed by `Uid.to_int`), then a
single pass over `topological_order` fills `materialized` and collects
`roots`.

## Invariants

- `kernel_roots` ⊆ materialised, contains no Params, and is in
  topological order.
- The plan is a pure function of the graph; calling `plan` twice gives
  identical results.
- Every node reachable from a root through non-materialised nodes is
  element-wise (this is what lets Lower inline it). That follows from rule
  2 and needs no extra code, but do not "optimise" rule 2 away.

## Failure modes to avoid

- Treating `Gather` as a barrier: it is element-wise (`out[i] = src[idx[i]]`)
  and must be inlineable; `test_fusion` checks it.
- Materialising by *consumer count* instead of `Graph.fan_out` (which counts
  edges): `map2 add s s` must materialise `s`.
- Forgetting that an output can also be an intermediate (rule 3 and rule 4
  are independent; either alone materialises).
- Using structural equality on `Tensor.packed`; key by uid.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_fusion.exe
```

Expected: `8 tests, 0 failures`.

## Tests (already written: `test/unit/test_fusion.ml`)

- chain of three maps: only the last is a root; intermediates not materialised
- params are materialised but never roots
- map feeding a reduce is inlined; reduce is the only root
- fan_out ≥ 2 forces materialisation; roots in topological order `[s; a; b]`
- an intermediate that is also an output is materialised
- scan is a barrier; a map over it is a separate root, after it
- iota, gather and reshape are inlineable (single root)
- node used twice by one `map2` is materialised
