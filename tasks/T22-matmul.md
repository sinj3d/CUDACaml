# T22 — `Matmul`: a 2-D node, a 2-D launch, and its adjoint

## Goal

Dense `[m; k] × [k; n] → [m; n]` as a node with a tiled shared-memory
kernel. Sizes in pricing are modest (regression design matrices, small
correlation factors), so an in-house kernel is right and cuBLAS is not
bound. This is the one task that widens `Schedule.launch` to two
dimensions.

Depends on: T14 (`transpose`), T18 (adjoint arm). Phase 4.

## Files you own

- `lib/ir/tensor.ml`, `lib/ir/graph.ml`, `lib/ir/dsl.ml`, `lib/ir/dsl.mli` (you may edit)
- `lib/backend_interp/ocaml_cuda_backend_interp.ml`
- `lib/lower/fusion.ml`, `lib/lower/kernel_ir.ml`, `lib/lower/schedule.ml`, `lib/lower/schedule.mli` (you may edit), `lib/lower/lower.ml`
- `lib/backend_cuda/emit.ml`, `lib/backend_cuda/executor.ml`
- `lib/runtime/launch.ml`, `lib/runtime/launch.mli` (you may edit)
- `lib/ad/grad.ml`
- `test/unit/test_lower.ml`, `test/unit/test_emit.ml`, `test/unit/test_runtime.ml` — **compile fixes only** for the new `launch` fields and `Kernel_ir` arms
- `test/staged/unit/test_matmul.ml` → promote

## Interfaces

`Tensor.node`: `| Matmul : 'a t * 'a t -> 'a node  (** row-major [m;k] x [k;n] -> [m;n] *)`.
`deps`: `[P a; P b]`. `node_kind`: `"Matmul"`.

`dsl.mli`: `val matmul : 'a Tensor.t -> 'a Tensor.t -> 'a Tensor.t`
(`Invalid_argument` unless both rank 2 and inner dims agree; any numeric dtype).

`schedule.mli`:

```ocaml
type launch = { grid : int; grid_y : int; block : int; block_y : int; shared_bytes : int }
(* every existing constructor sets grid_y = 1, block_y = 1 *)

val tile : int                                   (* 16 *)
val tiled_2d : m:int -> n:int -> launch          (* grid = ceil(n/tile); grid_y = ceil(m/tile); block = block_y = tile *)
```

`Kernel_ir.expr`: `| Block_id_y  (** blockIdx.y *)` and `| Local_thread_id_y  (** threadIdx.y *)`.
`Emit` prints them; `collect_expr`/`rename_expr` pass them through.

`launch.mli`:

```ocaml
val run :
  Jit.kernel -> grid:int -> ?grid_y:int -> block:int -> ?block_y:int -> shared_bytes:int -> Buffer.t list -> unit
```

(cudajit's `Stream.launch_kernel` already takes `?grid_dim_y` / `?block_dim_y`.)
`Executor` passes both dimensions from `k.launch`.

## Implementation

**Interp**: `out[i*n + j] = fold_{p = 0..k-1} acc + a[i*k + p] * b[p*n + j]`
using the dtype's `binop` for both `Mul` and `Add` so F32 rounds per step,
starting from the dtype's zero, `p` ascending.

**Fusion**: `Matmul` is a barrier (materialised). Its operands are *not*
special: an operand that is an inlineable map is inlined into the tile
loads through `elem`, as for every other consumer.

**Lower**: launch `Schedule.tiled_2d ~m ~n`; two `Shared` buffers
`sa`, `sb` of `tile*tile`; body:

```
row = Block_id_y*tile + Local_thread_id_y ; col = Block_id*tile + Local_thread_id
acc = 0
for t = 0; t < ceil(k/tile); t++:
  ka = t*tile + Local_thread_id ; kb = t*tile + Local_thread_id_y
  sa[ty*tile + tx] = (row < m && ka < k) ? elem a [row*k + ka] : 0
  sb[ty*tile + tx] = (kb < k && col < n) ? elem b [kb*n + col] : 0
  Sync_threads
  for p = 0; p < tile; p++: acc = acc + sa[ty*tile + p] * sb[p*tile + tx]
  Sync_threads
if row < m && col < n: out[row*n + col] = acc
```

`acc` is a `Let` before the loop and `Assign` inside; `Store`/`If` as
usual. The `elem` calls are where operand fusion happens.

**Grad**: for `C = A·B` with adjoint `ā`: `accumulate A (matmul ā (transpose B))`
and `accumulate B (matmul (transpose A) ā)`, each only when that operand is float.

## Invariants

- `Shape.dims (shape (matmul a b)) = [m; n]`, computed in `Dsl`.
- Per-element accumulation order is `p` ascending on both backends; the
  device may contract `a*b + c` into an FMA, so float tolerances apply
  (1e-4 relative for F32 at k ≤ 256 in the system test; F64 1e-9).
- Every pre-existing `Schedule` constructor yields `grid_y = block_y = 1`,
  and `Executor` behaviour for them is unchanged.

## Failure modes to avoid

- Swapping the roles of `Local_thread_id` and `Local_thread_id_y` in the
  tile loads (the tests use non-square shapes so a swap fails).
- A single `Sync_threads` per tile step: the second one protects the next
  step's overwrite of `sa`/`sb`.
- Bounds guards missing on the loads when `m`, `n`, `k` are not multiples
  of 16 (the tests use 5×7 by 7×3).
- Forgetting `Block_id_y` in Emit's `collect_expr`: the mangler's
  traversal is exhaustive and the build will tell you.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_matmul.exe && dune exec test/unit/test_grad.exe
```

## Tests (already written: `test/staged/unit/test_matmul.ml`)

- interp: `[[1;2;3];[4;5;6]] × [[7;8];[9;10];[11;12]]` = `[[58;64];[139;154]]`; identity leaves a 3×3 unchanged; 5×7 · 7×3 against a hand-written reference in the test
- `Dsl.matmul` rejects rank ≠ 2 and inner-dim mismatch
- fusion: `matmul` is materialised; `matmul (map f a) b` is one kernel whose params are `p_a`, `p_b`, out
- lower: launch grid = ceil(n/16), grid_y = ceil(m/16), block = block_y = 16; two `Shared` buffers of 256; body contains `Block_id_y` and `Local_thread_id_y` and ≥ 2 `Sync_threads`
- `Schedule.grid_stride`/`rows_block` still give `grid_y = block_y = 1`
- emit: source contains `blockIdx.y`, `threadIdx.y`, two `__shared__` arrays
- grad: `sum (matmul A B)` gradients vs `fd` for A `[2;3]`, B `[3;2]`; `sum (matmul (map exp A) B)` likewise
