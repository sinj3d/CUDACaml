# T24 — Persistent executor: allocate once, run many

## Goal

`Executor.run` today allocates every device buffer, uploads, launches,
downloads and frees, per run. The measured cost of that on a 2^24-element
output is tens of milliseconds, more than the kernels. Make a compiled
program own its device buffers for its lifetime: `Alloc` at compile,
`Free` at release, and a `run` that only moves data and launches.

Depends on: T08 (v1). Phase 5. Independent of Phases 1–4; can be done
early.

## Files you own

- `lib/backend_cuda/executor.ml`, `lib/backend_cuda/executor.mli` (you may edit)
- `lib/backend_cuda/ocaml_cuda_backend_cuda.ml`, `lib/backend_cuda/ocaml_cuda_backend_cuda.mli` (you may edit)
- `lib/runtime/buffer.ml`, `lib/runtime/buffer.mli` (you may edit)
- `test/unit/test_executor.ml` — **compile fixes only** if it calls `Executor.run` directly
- `test/staged/unit/test_pool.ml` → promote

## Interfaces

`buffer.mli` addition:

```ocaml
(** Allocations minus frees since process start. For tests and leak checks. *)
val live_count : unit -> int
```

`executor.mli`:

```ocaml
type t

(** Runs every [Alloc] in the plan once. Raises if the plan allocates a
    buffer it never frees or frees one it never allocates (the plan is
    validated, not trusted). *)
val create : Kernel_ir.program -> Jit.module_ -> t

(** [Upload], [Launch], [Download] only; [Alloc]/[Free] are skipped.
    Outputs are fresh host [Value]s. Raises [Invalid_argument] on a
    missing input and [Failure] after [release]. *)
val run : t -> inputs:(string * Value.packed) list -> (string * Value.packed) list

(** Frees every buffer. Idempotent. *)
val release : t -> unit

(** For tests: number of device buffers this executor holds. *)
val buffer_count : t -> int
```

`ocaml_cuda_backend_cuda.mli`: `compiled` now holds an `Executor.t`;
add `val release : compiled -> unit` and `val executor : compiled -> Executor.t`.
`Backend.S` is unchanged.

## Implementation

- `create`: walk the plan; on `Alloc b` allocate into the table; on `Free`
  record that the name is freed-at-release; anything else ignored. Check
  that the sets match.
- `run`: as today minus `Alloc`/`Free`. Keep the "sync before first
  download" logic. Since buffers persist, a `Download` of a buffer never
  written on this run would return stale data; that cannot happen with a
  plan produced by `Lower` (every output buffer is written by a launch or
  uploaded), but assert it in a debug check rather than trusting it: the
  set of downloaded buffers must be a subset of those uploaded or launched
  into during this run.
- `release`: free everything, mark released.
- Scratch buffers from T21 (partials/sums) are `Alloc`/`Free` pairs inside
  the plan; they simply become persistent too. Fine.
- Add a finaliser (`Gc.finalise`) that calls `release`, so a dropped
  `compiled` does not leak device memory; but tests call `release`
  explicitly and must not rely on the GC.

## Invariants

- `Buffer.live_count ()` is the same before and after any number of `run`s.
- `release` returns `live_count` to its pre-`create` value.
- Two compiled programs never share a buffer even if their buffer names coincide.

## Failure modes to avoid

- Freeing in `run` on exception: with `Fun.protect ~finally` from v1
  copied over, a failed run would free the pool. Remove it; errors
  propagate and the buffers stay owned.
- Downloading into a cached host `Value`: outputs must be fresh, because
  callers keep them (S5 compares 50 of them).
- `live_count` counting the `max 1 bytes` dummy allocations differently from real ones: every `alloc` is +1, every `free` is −1.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_pool.exe
```

## Tests (already written: `test/staged/unit/test_pool.ml`; GPU, skips without one)

- compile saxpy n=4096; `live_count` rises by `buffer_count`; 50 runs with changing inputs give correct sums and `live_count` does not change
- `release` returns `live_count` to the value before `compile`; a second `release` is a no-op; `run` after `release` raises
- two compiled programs alive at once: both run correctly, interleaved
- outputs of two runs are distinct `Value`s (mutating one does not change the other)
- a run with a missing input raises `Invalid_argument` and leaves `live_count` unchanged
