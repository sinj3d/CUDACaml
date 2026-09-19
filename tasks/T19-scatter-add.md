# T19 — `Scatter_add`, atomics, and the `Gather` adjoint

## Goal

The adjoint of `out[i] = src[idx[i]]` is `adj_src[idx[i]] += adj_out[i]`:
a scatter with addition, which the IR cannot express. Add it as a node,
lower it with `atomicAdd`, and complete `Grad` for `Gather`. It is also the
primitive for bucketing and histograms.

This task also makes the first small generalisation of `Lower`: a kernel
root may produce **more than one kernel** (zero-fill, then scatter). T21
builds on the same hook.

Depends on: T18. Phase 3.

## Files you own

- `lib/ir/tensor.ml`, `lib/ir/graph.ml`, `lib/ir/dsl.ml`, `lib/ir/dsl.mli` (you may edit)
- `lib/backend_interp/ocaml_cuda_backend_interp.ml`
- `lib/lower/fusion.ml`, `lib/lower/fusion.mli` (doc comment only), `lib/lower/kernel_ir.ml`, `lib/lower/lower.ml`
- `lib/backend_cuda/emit.ml`
- `lib/ad/grad.ml`
- `test/unit/test_lower.ml`, `test/unit/test_emit.ml` — **compile fixes only** (new `Kernel_ir.stmt` arm)
- `test/staged/unit/test_grad.ml` — replace the single assertion "Gather raises Not_differentiable" with the two gather-gradient checks listed below; nothing else in that file changes
- `test/staged/unit/test_scatter.ml` → promote

## Interfaces

`Tensor.node`:

```ocaml
| Scatter_add : int32 t * 'a t * Shape.t -> 'a node
    (** [Scatter_add (idx, src, shape)]: out = zeros shape; for every i,
        out[idx[i]] += src[i]. [idx] and [src] have the same shape.
        Float or I32 element type; summation order is unspecified. *)
```

`Tensor.deps`: `[P idx; P src]`. `Graph.node_kind`: `"Scatter_add"`.

`dsl.mli`:

```ocaml
(** [scatter_add idx src shape]; [Invalid_argument] unless
    [shape idx = shape src] and the dtype is F32, F64 or I32. *)
val scatter_add : int32 Tensor.t -> 'a Tensor.t -> Shape.t -> 'a Tensor.t
```

`Kernel_ir.stmt`:

```ocaml
| Atomic_add of { buf : buffer; index : expr; value : expr }   (** atomicAdd(&buf[index], value) *)
```

`Emit`: `atomicAdd(&%s[%s], %s);`. For `I64`/`Bool` buffers `failwith`.

**Fusion**: `Scatter_add` is a barrier (always materialised), like
`Reduce`/`Scan`. Update the doc comment in `fusion.mli` accordingly.

## Implementation

**Interp**: allocate zeros (`Value.create` is zero-initialised: verify, and
fill explicitly if not); loop `i`, bounds-check `idx[i]` against the
output numel exactly like `Gather` does, accumulate with the dtype's
`binop Add`.

**Lower**: refactor `kernel_of : plan -> buffers -> root -> K.kernel` into
`kernels_of : ... -> K.kernel list`, and have `program` concatenate. The
plan's `Launch` ops follow the same order. Every existing root returns a
singleton list, so no v1 test changes.

For `Scatter_add (idx, src, _)` with `m = numel out`, `n = numel src`:

1. `k_<uid>_zero`: grid-stride over `m`, `Store out[i] = 0` (`Schedule.grid_stride ~numel:m`).
2. `k_<uid>`: grid-stride over `n`, `Atomic_add { buf = out; index = elem idx i; value = elem src i }`
   (`Schedule.grid_stride ~numel:n`). Params: the materialised inputs of
   both `idx` and `src` (via `inputs_of`), then `out` last.

Kernel launches on the same stream are ordered, so no explicit sync is
needed between the two.

**Grad**: replace the `Gather` arm with
`accumulate src (scatter_add idx ā (shape src))` when `src` is float.

## Invariants

- The output buffer is written by two kernels; the zero kernel is always
  emitted first, and both appear in `program.kernels` adjacent and in that order.
- On the interpreter results are exact; on the device float results depend
  on atomic ordering and tests use a tolerance. I32 results are exact everywhere.

## Failure modes to avoid

- Fusing `Scatter_add` (it has no per-element expression).
- Zero-filling inside the scatter kernel: other blocks may already have added.
- Forgetting to add the `Atomic_add` arm to `collect_stmt`/`rename_stmt` in `emit.ml`.
- Bounds: the interpreter must raise on an out-of-range index; the device kernel is unguarded and the docs say so.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_scatter.exe && dune exec test/unit/test_grad.exe
```

## Tests (already written: `test/staged/unit/test_scatter.ml`)

- interp: histogram of `[0;1;1;3;3;3]` into 4 buckets of F32 ones → `[1;2;0;3]`; I32 variant exact
- interp: out-of-range index raises `Invalid_argument`
- `Dsl.scatter_add` rejects a shape mismatch between idx and src
- fusion: a `scatter_add` is materialised even with fan-out 1
- lower: exactly 2 kernels for a lone scatter root; the first has a body with a single `Store` of a zero literal; the second contains an `Atomic_add` into the same buffer; plan has 2 `Launch` ops in that order
- emit: source contains `atomicAdd(&`
- `Graph.to_dot` mentions `Scatter_add`

Added to `test/staged/unit/test_grad.ml` by this task:

- `sum (gather idx x)` with `idx = [0;0;1]`, `x : [3]` → `[2; 1; 0]`
- the `reverse` example (gather by `n-1-i`) with weights: gradient is the reversed weights, checked against `fd`
