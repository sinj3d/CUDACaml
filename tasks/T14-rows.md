# T14 — Last-axis `Reduce` and `Scan`, and `transpose`

## Goal

Monte Carlo paths are a 2-D tensor `[n_paths; n_steps]`. The cumulative
log-return is a scan along the step axis of every path; a per-path payoff
reduction is a reduce along that axis. Today `Reduce` and `Scan` see a
tensor as one flat run of elements, so neither is expressible.

Change the meaning of the two constructors to **"along the last axis"**:

- `Reduce (f, init, src)` with `src : [d0; ...; dk-1; n]` yields `[d0; ...; dk-1]`;
  a rank-1 source yields `Shape.scalar` (unchanged behaviour).
- `Scan (f, init, src)` scans each row of length `n` independently; shape unchanged.

The existing `Dsl.reduce` / `Dsl.scan` keep their **whole-tensor** meaning
by flattening first. New `reduce_rows` / `scan_rows` expose the row form.
Other axes are reached through `transpose`, which is a fused gather.

Depends on: T13. Phase 1.

## Files you own

- `lib/ir/tensor.ml` (doc comments only), `lib/ir/dsl.ml`, `lib/ir/dsl.mli` (you may edit)
- `lib/backend_interp/ocaml_cuda_backend_interp.ml`
- `lib/lower/kernel_ir.ml`, `lib/lower/schedule.ml`, `lib/lower/schedule.mli` (you may edit), `lib/lower/lower.ml`
- `lib/backend_cuda/emit.ml`
- `test/unit/test_lower.ml` — **compile fix only**: `loads_in_expr` must gain a `K.Block_id` arm. No assertion changes.
- `test/staged/unit/test_rows.ml` → promote

## Interfaces

`dsl.mli`:

```ocaml
(** Whole-tensor reduction to a scalar. Rank ≥ 2 sources are flattened with
    [reshape] first; the result is [Shape.scalar] as before. *)
val reduce : ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Whole-tensor inclusive scan in flat index order; shape preserved. Rank ≥ 2
    sources are flattened and reshaped back. *)
val scan : ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Reduce every row (last axis). Rank ≥ 2 required, else [Invalid_argument].
    [[d0;..;dk-1;n]] -> [[d0;..;dk-1]]. *)
val reduce_rows : ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Inclusive scan of every row (last axis). Rank ≥ 2 required. *)
val scan_rows : ('a Expr.t -> 'a Expr.t -> 'a Expr.t) -> init:'a Expr.t -> 'a Tensor.t -> 'a Tensor.t

(** Rank-2 only. [transpose x] is a [gather] over an [iota] index tensor
    ([out[i*m + j] = x[j*n + i]]), so it fuses into its consumer and never
    owns a buffer unless fan-out forces it. *)
val transpose : 'a Tensor.t -> 'a Tensor.t
```

`kernel_ir.ml` `expr` gains `| Block_id  (** blockIdx.x *)`. `emit.ml`
prints it as `blockIdx.x` and handles it in `collect_expr`/`rename_expr`.

`schedule.mli`:

```ocaml
(** Reduction kernels: one block of [block_size] threads PER ROW. *)
val rows_block : rows:int -> launch      (* { grid = rows; block = block_size; shared_bytes = 0 } *)

(** Scan kernels (sequential per row until T21): one thread per row. *)
val rows_thread : rows:int -> launch     (* { grid = rows; block = 1; ... } *)
```

Remove `single_block` and `single_thread`, or keep them as
`rows_block ~rows:1` / `rows_thread ~rows:1`; either is fine, but
`test_lower.ml` uses neither by name.

## Implementation

`row_length (P t)` = last dim of `Tensor.shape`; `rows` = numel / row_length
(1 for rank-1). Rank-0 sources never reach here: `Dsl` rejects them.

**Interp**: outer loop over rows, inner sequential fold as today; write
`out[row]` (reduce) or `out[row*n + i]` (scan).

**Lower, Reduce**: keep the block-tree kernel, launched with
`Schedule.rows_block ~rows`. Inside: `row = Block_id`; the strided loop is
`i = row*n + tid; i < row*n + n; i += Block_dim`; the final store is
`out[row] = sdata[0]`. For rows = 1 the emitted body differs from v1 only
by `blockIdx.x * n` terms that fold to zero; `test_lower` "reduce kernel
shape" asserts `grid = 1` and still passes.

**Lower, Scan**: `Schedule.rows_thread ~rows`; `row = Block_id`; the loop
runs `i` from `row*n` to `row*n + n`. `test_lower` "scan kernel runs on one
thread" asserts grid 1, block 1 for a rank-1 scan; still true.

**Dsl.reduce**: if rank = 1 build `Reduce` directly (so every v1 structural
test, including fusion's "map feeding a reduce is inlined; reduce is the
only root", sees the same nodes). If rank ≥ 2, `Reduce` over
`reshape (vec numel) src`. **Dsl.scan**: same, then `reshape src.shape`.

**Dsl.transpose**: `gather idx x` with `idx = map (fun k -> (k mod m) * n + k / m) (iota [n; m])`
for `x : [m; n]`. Integer `div`/`mod` on `I32` expressions: `mod` is not in
`Expr.binop`; use `k - (k / m) * m`.

## Invariants

- `Tensor.Reduce` output shape is computed in `Dsl` (`Shape.of_dims (all but last)`),
  never in `Lower`; `Lower` derives rows from shapes.
- A rank-1 `reduce` builds exactly the same graph as in v1.
- `reduce_rows (transpose x)` equals column sums; the test checks it on the interpreter.

## Failure modes to avoid

- Using `Global_thread_id` inside the row-reduce loop: it is `blockIdx.x * blockDim.x + threadIdx.x`
  and would skew every row after the first.
- Forgetting the `Sync_threads` between tree steps still applies per block; nothing changes there.
- Letting `Dsl.reduce` on a rank-2 tensor return shape `[d0]`; it must return a scalar.
- `transpose` on a non-rank-2 tensor must raise `Invalid_argument`.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_rows.exe
```

## Tests (already written: `test/staged/unit/test_rows.ml`)

- interp: `reduce_rows add` on `[[1;2;3];[4;5;6]]` → `[6; 15]`; result shape `[2]`
- interp: `reduce add` on the same 2-D tensor → scalar 21
- interp: `scan_rows add` → `[[1;3;6];[4;9;15]]`; `scan add` (flat) → `[1;3;6;10;15;21]` reshaped to `[2;3]`
- interp: `reduce_rows max` picks the row max with a finite identity
- interp: `transpose` of `[2;3]` → `[3;2]` with the right elements; `reduce_rows add (transpose x)` = `[5; 7; 9]`
- rank-1 `reduce_rows` / `scan_rows` and rank-≠2 `transpose` raise `Invalid_argument`
- lower: `reduce_rows` over `[2;3]` gives one kernel with `launch.grid = 2`, block = `Schedule.block_size`, output numel 2, and a body that mentions `K.Block_id`
- lower: `scan_rows` over `[4;10]` gives `grid = 4`, `block = 1`
- lower: rank-1 `reduce` still gives `grid = 1` (v1 unchanged)
- emit: the source of a rows reduce contains `blockIdx.x` and its kernel is `extern "C"`
- fusion: `transpose` with fan-out 1 is not materialised; `reduce_rows (map f (transpose x))` is one kernel
