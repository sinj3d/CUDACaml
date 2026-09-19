# T13 — `Broadcast`: a scalar tensor everywhere

## Goal

Add one node to the IR: `Broadcast (shape, src)` where `src` has exactly one
element, `out[i] = src[0]`. Today a scalar market input (spot, vol, rate)
either has to be padded to a length-n vector on the host or baked in as an
`Expr.Const`, and constants are not differentiable. T18 (reverse-mode AD)
needs this for the adjoint of a reduction, and T16/T20 need it to pass a
seed or a spot price as a `Param` of shape `Shape.scalar`.

Also add `Dsl.full`, a constant tensor built from `iota`, so a "tensor of
ones" needs no buffer and no input.

Depends on: T02–T06 (v1). Phase 1.

## Files you own

- `lib/ir/tensor.ml`, `lib/ir/graph.ml`, `lib/ir/dsl.ml`, `lib/ir/dsl.mli` (you may edit this `.mli`)
- `lib/backend_interp/ocaml_cuda_backend_interp.ml`
- `lib/lower/fusion.ml`, `lib/lower/lower.ml`
- `test/staged/unit/test_broadcast.ml` → promote

## Interfaces

`Tensor.node` gains:

```ocaml
| Broadcast : Shape.t * 'a t -> 'a node
    (** [src] has numel 1 (shape [] or [1] or [1;1]...); out[i] = src[0].
        Result shape is the first component. *)
```

`Tensor.deps`: `[P src]`. `Graph.node_kind`: `"Broadcast"`.

`dsl.mli` gains:

```ocaml
(** [broadcast shape s]: [s] must have exactly one element, otherwise
    [Invalid_argument]. Result has [shape]. *)
val broadcast : Shape.t -> 'a Tensor.t -> 'a Tensor.t

(** [full dtype shape v]: every element is [v]. No Param, no buffer: it is
    [map (fun _ -> const dtype v) (iota shape)] and is always inlined. *)
val full : 'a Dtype.t -> Shape.t -> 'a -> 'a Tensor.t

(** [scalar name dtype] = [param name dtype Shape.scalar]. *)
val scalar : string -> 'a Dtype.t -> 'a Tensor.t
```

## Implementation

- **Interp**: evaluate `src`, `Value.get vs 0`, fill an output of the
  broadcast shape.
- **Fusion**: `Broadcast` is element-wise. It is inlineable exactly like
  `Reshape`; nothing in `fusion.ml` should need more than adding it to the
  list of element-wise kinds in the comment. It is materialised only by the
  generic rules (output, fan-out ≥ 2).
- **Lower** `inline_node`: `Tensor.Broadcast (_, a) -> elem plan bufs (P a) ~index:(int_lit 0)`.
  `kernel_of`: add `Broadcast` to the element-wise arm (grid-stride loop
  over the *output* numel).
- Note the interaction with fusion: if `src` is a `Reduce` (materialised),
  the load is `t42[0]` inside the consumer's loop, which the compiler hoists.
  If `src` is a `Param`, it is `p_s[0]`. Either way there is no broadcast
  kernel unless the broadcast itself is an output.

## Invariants

- `Shape.numel (Tensor.shape (P src)) = 1` is checked in `Dsl.broadcast`,
  not in `Lower` or the interpreter; those trust the graph.
- `Dsl.full` creates no `Param` and `Lower.program` of a graph whose only
  input is a `full` has zero `Upload` ops.

## Failure modes to avoid

- Reading `src[index]` instead of `src[0]` in `inline_node`: the tests
  broadcast a *reduce result* precisely to catch this (the reduce buffer
  has numel 1, so an out-of-bounds read would be silent on the GPU).
- Making `Broadcast` a barrier in `Fusion`. It must inline: `map2 mul x
  (broadcast shape s)` is one kernel.
- Forgetting `Graph.node_kind`; `to_dot` is exhaustive.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_broadcast.exe
```

## Tests (already written: `test/staged/unit/test_broadcast.ml`)

- interp: `broadcast [4] (scalar s)` with s = 2.5 gives four 2.5s
- interp: `map2 mul x (broadcast (shape x) (reduce add x))` = x * sum x
- `Dsl.broadcast` rejects a source with numel ≠ 1
- `Dsl.full F32 [3] 7.0` evaluates to `[7;7;7]` and its graph has no params
- fusion: broadcast with fan-out 1 is not materialised; the map2 is the only root
- lower: the consumer kernel contains a `Load` of the reduce buffer at
  literal index 0 and no other index into it
- lower: broadcasting a param yields a kernel with params `[p_s; out]` and
  a grid-stride loop over the output numel
- `Graph.to_dot` of a graph with a broadcast contains `Broadcast`
- deps of a Broadcast node is exactly its source
