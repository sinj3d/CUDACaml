# T21 — Grid-wide reduction and parallel scan

## Goal

Today a `Reduce` runs on **one block** per row (256 threads walk the whole
row) and a `Scan` runs on **one thread** per row. At 2^24 elements that is
the slowest code in the project by a wide margin. Both become multi-kernel
schedules using the root-to-kernel-list hook from T19, with scratch buffers
in the host plan. Small inputs keep the single-kernel form so every v1
structural test still holds.

Depends on: T14, T19. Phase 4.

## Files you own

- `lib/lower/schedule.ml`, `lib/lower/schedule.mli` (you may edit), `lib/lower/lower.ml`
- `test/staged/unit/test_multikernel.ml` → promote

No IR or Emit change. `Kernel_ir` already has `Block_id`, `Local_thread_id`,
`Block_dim`, `Global_thread_id`, `Global_size`, `Sync_threads`, `Shared`.

## Interfaces (`schedule.mli` additions)

```ocaml
(** Reduce: blocks per row. 1 when [row_len <= reduce_threshold], else
    [min 128 (ceil (row_len / (block_size * 8)))]. *)
val reduce_blocks : row_len:int -> int
val reduce_threshold : int      (* 4096 *)

(** Scan: elements per block-scan chunk; equals [block_size]. *)
val scan_chunk : int

(** Scan launch for [rows] rows of [row_len]: 1 kernel if
    [row_len <= scan_chunk], else 3. *)
val scan_kernels : row_len:int -> int
```

## Implementation

### Reduce, `G = reduce_blocks ~row_len:n`, rows `R`

- `G = 1`: today's kernel, unchanged.
- `G > 1`: two kernels and one scratch buffer `t<uid>_partials` (Global, numel `R*G`, dtype of the node).
  1. `k_<uid>_partial`: launch `{ grid = R*G; block = block_size }`.
     `row = Block_id / G`, `blk = Block_id - row*G`. Each thread folds
     `i = row*n + blk*block_size + tid; i < row*n + n; i += G*block_size`,
     then the existing shared-memory tree; thread 0 stores `partials[Block_id]`.
  2. `k_<uid>`: launch `{ grid = R; block = block_size }` (`Schedule.rows_block`),
     folds `partials[row*G + tid]` for `tid < G` with stride `Block_dim`,
     tree, stores `out[row]`.
  Plan: `Alloc partials` before the first launch, `Free partials` right
  after the second; both between the existing param uploads and the first
  download.

### Scan (Hillis–Steele per chunk), `C = scan_chunk`, `B = ceil (n / C)` chunks per row

- `B = 1`: one kernel, launch `{ grid = R; block = C; shared = C }`:
  thread `tid` loads `x = (tid < n) ? src[row*n + tid] : identity`
  into `s[tid]`; for `off = 1, 2, 4, … < C`: `Sync_threads`; `v = (tid >= off) ? combine s[tid-off] s[tid] : s[tid]`; `Sync_threads`; `s[tid] = v`;
  finally `if tid < n then out[row*n + tid] = s[tid]`.
  Note the two syncs per step: read-all then write-all, so no lane reads a
  value from the current step. The `v` is a `Let` at loop scope.
- `B > 1`: three kernels, scratch `t<uid>_sums` (numel `R*B`):
  1. `k_<uid>_chunks`: `grid = R*B`, `block = C`: as above on chunk
     `Block_id mod B` of row `Block_id / B`; stores the inclusive scan of the
     chunk into `out` and its last element into `sums[Block_id]`.
  2. `k_<uid>_sums`: `grid = R`, `block = 1`: sequential inclusive scan of
     `sums[row*B .. row*B+B)` in place (the v1 scan body).
  3. `k_<uid>`: `grid_stride ~numel:(R*n)`: for element `i`, `row = i / n`,
     `blk = (i - row*n) / C`; `if blk > 0 then out[i] = combine sums[row*B + blk - 1] out[i]`.
  Since `out` is read and written by the same thread only, no atomics.
  Plan: `Alloc sums` / `Free sums` around the three launches.

`combine a b` is `expr_of fn.body2 ~env:[a; b]` exactly as today; the
operator is only required to be associative, and `init` its identity
(the `Dsl.scan` contract; note `combine sums[...] out[i]` puts the earlier
chunk on the left, which matters for non-commutative operators).

`Reduce`/`Scan` roots return their kernel list from `kernels_of`. Kernel
names: the *last* kernel keeps `k_<uid>` so any code mapping roots to
kernels by name still finds the one that writes the output.

## Invariants

- `Lower.program` on every v1 example is unchanged: all their rows are
  ≤ 4096 elements for reduce, and only `prefix 300` scans, which fits one
  chunk (its kernel changes from 1 thread to a 256-thread block; the v1
  test asserting `grid = 1, block = 1` is on a 10-element scan and is
  **updated by this task** to `block = Schedule.scan_chunk`; that is the
  one permitted assertion change and must be stated in the report).
- Every buffer named in a `Launch` is `Alloc`'d earlier in the plan and
  `Free`'d later; the staged test checks this for every program it lowers.
- Float results differ from the interpreter by reassociation only; the
  system test uses tolerance 1e-3 at 2^22, as S3 always has.

## Failure modes to avoid

- One `Sync_threads` per Hillis–Steele step.
- Using `Global_thread_id` for the chunk-local index; it is `Local_thread_id`.
- `sums` scanned exclusively: kernel 3 must add `sums[blk-1]`, the
  inclusive total of the *previous* chunks.
- Threshold that changes `saxpy 1000`'s kernel count (`test_lower` asserts 2).
- Reusing `out` as the partials buffer in the reduce; `out` has `R` elements, not `R*G`.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_multikernel.exe
```

## Tests (already written: `test/staged/unit/test_multikernel.ml`)

Structural, no GPU. `well_formed plan` in the test asserts the Alloc/Launch/Free discipline.

- `reduce_blocks ~row_len:1000 = 1`; `~row_len:1_000_000 = 128`; `~row_len:8192 = 4`
- `reduce add` over `[1000]`: 1 kernel (v1). Over `[1 lsl 20]`: 2 kernels, one scratch `Alloc` and `Free`, first kernel grid 128, second grid 1; last kernel is named `k_<uid>` (ends the list and writes the output buffer)
- `reduce_rows add` over `[4; 1 lsl 16]`: first kernel grid = 4·`reduce_blocks`, second grid 4
- `scan add` over `[100]`: 1 kernel, block = `scan_chunk`, one `Shared` buffer of `scan_chunk` elements, at least 2 `Sync_threads` per loop body
- `scan add` over `[5000]`: 3 kernels, grid of the first = 20, second grid 1 block 1, third is grid-stride; scratch `Alloc`/`Free`
- `scan_rows add` over `[3; 700]`: 3 kernels, first grid 9, second grid 3
- `well_formed` holds for every example in `Programs.all` and for all graphs above
- saxpy still lowers to exactly 2 kernels
