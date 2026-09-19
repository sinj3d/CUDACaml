# T10 — System tests (GATED)

## Gate

Do **not** start this task until **all** of the following are true:

1. `make unit` exits 0 on the GPU machine — every unit test from T01–T08
   passes (none of them skipped for lack of a device).
2. T09's verify block passes, including the `check` loop over every example.
3. Someone has pasted the outputs of (1) and (2) into the task tracker.

If any of that is false, go back to the failing task. System tests are
not a place to debug unit-level problems.

## What runs

`test/system/test_system.ml` (already written). It refuses to run unless
`OCAML_CUDA_SYSTEM=1` is set and a device is present. `make system`
builds, runs the full unit suite first, and only then runs it:

```
make system 2>&1 | tee system.log
```

The suite:

| ID | What | Pass criterion |
|---|---|---|
| S1 | every example in `examples/programs.ml`, interp vs cuda | `Differential.check` = `Ok` at tol 1e-5 |
| S2 | saxpy at n ∈ {0, 1, 255, 256, 257, 1009, 300000} | Ok — covers empty tensors, single element, block boundaries, real striding |
| S3 | sum over 4 194 304 elements | Ok at tol **1e-3** (float reassociation) |
| S4 | chain of five maps | exactly 1 kernel in `Lower.program` AND Ok end to end |
| S5 | one compiled saxpy run 50× with changing inputs | every sum correct → no stale device state |
| S6 | i32 program with negative division | Ok (exact) |
| S7 | timing saxpy n = 2^24 | informational only; prints interp / cuda-first / cuda-warm / ratio |

Expected last line: `N tests, 0 failures` with N = 10 (examples) + 7 + 5 = 22.

## Triage table — which task owns a failure

| Symptom | Owner |
|---|---|
| NVRTC compile error in the log | T06 (Emit) — paste the generated source via `ocaml-cuda emit <ex>` |
| `get_function` fails / name not found | T06 (`extern "C"`) or T05 (kernel naming) |
| S1 mismatch on `sum`/`maxval`/`fanout` only | T05 reduce kernel (sync, init identity, tree bounds) |
| S1 mismatch on `prefix` only | T05 scan kernel |
| S1 mismatch on `reverse` | T05 gather inlining (index expr) |
| S1 mismatch on `clamp` | T06 select/cmp printing or T03 cmp semantics |
| S1 mismatch on `squares`/`reshape_flat` | T05 iota/reshape inlining |
| S2 fails only at n=0 | T07 (0-byte alloc) or T05 (`grid_stride ~numel:0`) |
| S2 fails only at n=300000 | T05 grid-stride loop bounds |
| S3 fails but S1 `sum` passes | tolerance handling in T08 (must be relative) |
| S4 kernel count ≠ 1 | T04 (Fusion) |
| S5 fails on iteration > 1 | T08 executor state (buffers reused / not freed) or T07 |
| S6 wrong sign on negative division | T06 int literal or T03 `Int32.div` |
| S7 ratio < 1 | not a failure; note it. Likely tiny n or first-run compile included |

## Report

Paste `system.log` in full, the S7 line, `nvidia-smi --query-gpu=name --format=csv,noheader`,
and `ocaml -version`. If everything passes, the project is demo-ready:
`dune exec ocaml-cuda -- emit chain` shows the fused kernel and
`dune exec ocaml-cuda -- check chain` proves it.
