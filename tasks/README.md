# Task specs

Each `Txx-*.md` is a self-contained work order. An implementer gets **only**
that file plus the repository. Read your spec top to bottom before opening
any source file.

## Order and dependencies

```
T01 Value ──▶ T02 Graph+Dsl ──▶ T03 Interp ──┐
                    │                          │
                    └──▶ T04 Fusion ──▶ T05 Lower ──▶ T06 Emit ──┐
                                                                 ├──▶ T08 Executor+Differential ──▶ T09 CLI ──▶ T10 SYSTEM
T07 Runtime (needs a GPU machine; independent of T01–T06) ───────┘
```

A task may start when every arrow into it is green (its predecessor's
verify command exits 0). T07 can run in parallel with T01–T06 on the GPU
box. **T10 runs only after T01–T09 are all green** — see its spec for the
gate.

## Rules for every task

1. **Build must be warning-free.** `dune build 2>&1` uses the dev profile,
   where unused variables, unused opens and non-exhaustive matches are
   errors. Do not add `[@@@warning "-..."]` to silence them; fix the code.
2. **Never weaken a test.** You may edit a file under `test/` only to fix a
   compile error, and the fix must keep the assertion's meaning. If a test
   looks wrong, say so in your report; do not change the expected value.
3. **Do not edit `.mli` files** unless your spec says so. The `.mli` is the
   contract other tasks build against.
4. **Do not touch files outside your spec's "Files you own" list.** If you
   need something from another layer that does not exist, stop and report.
5. **No new opam dependencies** except where a spec names one (T07: cudajit).
6. **Verify command must exit 0** before you report done. Paste its last
   ten lines in your report.
7. Your report: files changed, verify output, anything you could not do
   and why. No prose beyond that.

## Toolchain

OCaml ≥ 5.1, dune ≥ 3.0. From the repo root:

```
dune build 2>&1          # build everything
dune exec test/unit/test_<name>.exe   # one task's tests
make unit                # all unit tests
make system              # gated system tests (T10)
```

Tests use `test/lib/check.ml`, a 40-line harness with no dependencies:
`Check.test name (fun () -> ...)`, `Check.int/bool/string/float/floats/contains/raises`,
`Check.run ()` exits 1 on any failure.
