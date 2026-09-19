# Task specs

Each `Txx-*.md` is a self-contained work order. An implementer gets **only**
that file plus the repository. Read your spec top to bottom before opening
any source file.

## v1 (done): T01–T10

```
T01 Value ──▶ T02 Graph+Dsl ──▶ T03 Interp ──┐
                    │                          │
                    └──▶ T04 Fusion ──▶ T05 Lower ──▶ T06 Emit ──┐
                                                                 ├──▶ T08 Executor+Differential ──▶ T09 CLI ──▶ T10 SYSTEM
T07 Runtime (needs a GPU machine; independent of T01–T06) ───────┘
```

## v2: T11–T28 — what a derivatives desk needs

Phase 0 documentation and portability; Phase 1 IR enablers; Phase 2 RNG;
Phase 3 reverse-mode AD; Phase 4 lowering (grid-wide reduce/scan, matmul);
Phase 5 runtime (buffer pool, streams, pinned memory, multi-GPU);
Phase 6 the second gated system suite plus the A100 campaign.

```
Phase 0   T11 info+docs      T12 brev+bench            (independent; any time)

Phase 1   T13 Broadcast ──┬──▶ T14 Rows (last-axis reduce/scan, transpose)
                          │
          T15 Expr ops ───┤
                          │
Phase 2                   └──▶ T16 RNG   (needs T13, T15)

Phase 3   T15 ──▶ T17 Deriv ──▶ T18 Grad (needs T13, T14, T17) ──▶ T19 Scatter_add
                                                                        │
Phase 2+3 T16, T18 ─────────────────────────────────────────────────────┴──▶ T20 Black–Scholes MC + Greeks

Phase 4   T14, T19 ──▶ T21 Multi-kernel reduce/scan
          T14, T18 ──▶ T22 Matmul ──▶ T23 Longstaff–Schwartz (needs T16, T21, T22)

Phase 5   T08 ──▶ T24 Persistent executor ──▶ T25 Streams/async/resident ──▶ T26 Pinned host memory
                                                          │
                                                          └──▶ T27 Multi-GPU

Phase 6   everything ──▶ T28 SYSTEM v2 + A100 campaign (GATED)
```

A task may start when every arrow into it is green (its predecessor's
verify command exits 0). T11, T12 and T24 have no v2 predecessors and can
run first. **T28 runs only after T11–T27 are all green**; see its gate.

## Rules for every task

1. **Build must be warning-free.** `dune build 2>&1` uses the dev profile,
   where unused variables, unused opens and non-exhaustive matches are
   errors. Do not add `[@@@warning "-..."]` to silence them; fix the code.
   Adding a constructor to `Tensor.node`, `Expr.binop`, `Expr.unop` or
   `Kernel_ir` makes every exhaustive match a compile error: that list of
   errors is your to-do list, and every site must be handled, not stubbed.
2. **Never weaken a test.** You may edit a file under `test/` only to fix a
   compile error, and the fix must keep the assertion's meaning. If a test
   looks wrong, say so in your report; do not change the expected value.
   A constant copied from an external reference (a known-answer vector, a
   closed-form price) may be *reported* as a suspected transcription error
   with your evidence; it may not be edited.
3. **Do not edit `.mli` files** unless your spec says so. The `.mli` is the
   contract other tasks build against.
4. **Do not touch files outside your spec's "Files you own" list.** If you
   need something from another layer that does not exist, stop and report.
5. **No new opam dependencies** except where a spec names one.
6. **Verify command must exit 0** before you report done. Paste its last
   ten lines in your report.
7. **Every v1 test must still pass.** `make unit` is part of every verify
   block. If your change legitimately alters a v1 expectation, the spec
   says so explicitly; otherwise it is a regression.
8. Your report: files changed, verify output, anything you could not do
   and why. No prose beyond that.

## Staged tests (v2)

v2 tests are written before the code they test, so they cannot be in
`test/unit/dune` yet: they would not compile. They live in `test/staged/`,
which has no `dune` file and is therefore invisible to the build. Each
spec's **Tests** section names its file(s) and the last step of every task is:

```
git mv test/staged/unit/test_<x>.ml test/unit/test_<x>.ml
```

then add `test_<x>` to the `names` list in `test/unit/dune`. From that
moment the test is part of `make unit` for everyone. `test/staged/README.md`
lists what is still staged.

The staged system suite `test/staged/system/test_system.ml` replaces
`test/system/test_system.ml` in T28 and nowhere else.

## Toolchain

OCaml ≥ 5.1, dune ≥ 3.0. From the repo root:

```
dune build 2>&1          # build everything
dune exec test/unit/test_<name>.exe   # one task's tests
make unit                # all unit tests
make system              # gated system tests (T10 / T28)
make bench               # speedup benchmark (needs a GPU)
```

Tests use `test/lib/check.ml`, a 40-line harness with no dependencies:
`Check.test name (fun () -> ...)`, `Check.int/bool/string/float/floats/contains/raises`,
`Check.skip`, `Check.run ()` exits 1 on any failure.

GPU-dependent unit tests skip themselves when `Runtime.Device.available ()`
is false and count as passed. On a GPU machine nothing may skip.
