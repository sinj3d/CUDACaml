# SKILLS.md — working in CUDACaml as a coding agent

Read this before editing. It is the short version of
[ARCHITECTURE.md](ARCHITECTURE.md), plus the things that will waste your time
if you learn them by trial.

## The one-sentence model

A typed DAG (`Graph.t`) is rewritten by passes, `Fusion` decides which nodes
get a buffer, `Lower` turns the rest into imperative kernels, `Emit` prints
CUDA C++, and NVRTC compiles it at run time. A pure-OCaml interpreter runs the
same graph and is the oracle for every correctness claim.

## Commands

```
make build          # dune build
make unit           # all unit tests; GPU ones skip themselves without a device
make system         # gated end-to-end suite; needs a GPU and CUDACAML_SYSTEM=1
make env            # print what the build resolved (CUDA_PATH, LD_LIBRARY_PATH)
make record         # benchmark both precisions, append to bench/results/

dune exec cudacaml -- emit <example>    # generated CUDA C++, no GPU needed
dune exec cudacaml -- dot <example>     # graph as Graphviz, no GPU needed
dune exec cudacaml -- check <example>   # differential-test GPU vs interpreter
dune exec cudacaml -- list              # example names
```

`make system` sets `CUDACAML_SYSTEM=1` itself. Run one unit test directly with
`dune exec test/unit/test_value.exe`.

### Three traps

- **`dune exec cudacaml` fails with `libnvrtc.so.12: cannot open shared object
  file` even though `make unit` passes.** The `Makefile` exports
  `LD_LIBRARY_PATH`; bare `dune exec` does not inherit it. Either go through a
  `make` target or export `CUDA_PATH` and `$CUDA_PATH/lib64` yourself. This is
  not a broken build.
- **`make unit` passing does not mean the GPU paths ran.** Seven suites skip
  themselves when no device is present — `test_runtime`, `test_executor`,
  `test_info`, `test_pool`, `test_streams`, `test_pinned`, `test_multigpu`
  (which also skips its second half on a one-card machine) — and a skip still
  reports `0 failures`. Only `make system` fails rather than skips. If you
  changed anything below `lib/lower`, run `make system`.
- **`make unit` printed nothing and exited 0.** dune caches test results, so
  on an unchanged tree it re-runs nothing and shows nothing. `make system`
  passes `--force` for exactly this reason; to see the unit suite run again,
  use `dune test test/unit --force`.

## Layering — the rule that is enforced

Each layer is a separate dune library, so a violation is a build error rather
than a review comment. Dependencies point strictly downward:

```
ir  →  passes  →  lower  →  backend_cuda
                  runtime ↗
       backend / backend_interp
ir  →  rng, ad        (graph libraries built from Dsl calls; ir only)
```

`runtime` never imports `Graph` or `Kernel_ir`. It knows bytes, pointers, PTX
strings and kernel handles. If you find yourself wanting a `Graph` in
`lib/runtime`, the design has gone wrong — put it in `backend_cuda`.

## Invariants — do not quietly break these

1. **`Backend_interp` never runs passes.** It is the oracle, so it must stay
   independent of what it judges. Optimising it to match the CUDA path
   destroys its only purpose.
2. **`Emit` is a pretty-printer.** Any decision that has to be made there
   belongs in `Lower` or `Schedule`. Emitting different code based on a
   heuristic is the bug, not the fix.
3. **Shapes are static per `Graph.t`.** A new size means a new graph. There is
   no dynamic-shape path to extend.
4. **No dtype is "the fast one".** Every dtype-dependent choice is made from
   the `Dtype.t` witness. `F32` and `F64` go through identical machinery.
5. **`Fusion` decides materialisation only** — params, the fusion barriers
   (`Reduce`, `Scan`, `Scatter_add`, `Matmul`), outputs, and anything with
   fan-out ≥ 2 get a buffer; everything else is inlined by `Lower`. That is
   how map/map2/reduce fusion happens without an n-ary node in the IR.

## Adding a node to the IR

`Tensor.node` has eleven variants and is the contract between the front end
and every pass. Adding one means updating, and the compiler will list them:

- `Tensor.deps` — or traversal silently skips your operands
- `Backend_interp` — the meaning, written the obvious slow way
- `Fusion` — does it need a buffer?
- `Lower` — the kernel, or the inline expression
- `Dsl` — the user-facing constructor, with shape validation
- `Grad` (`lib/ad/`) — the adjoint, if it is differentiable

Write the interpreter case **first** and a unit test against it, then make the
CUDA path agree. That ordering is the whole point of having an oracle.

## Testing

`test/lib/check.ml` is the harness: `C.test`, `C.int`, `C.bool`,
`C.float ?tol`, `C.floats ?tol`, `C.string`, `C.contains`, `C.raises`,
`C.fail`, `C.skip`. A suite ends with `C.run ()` and prints
`N tests, 0 failures`.

- **Unit tests** (`test/unit/`, one executable per module) must pass without a
  GPU, or skip themselves explicitly with `C.skip`.
- **System tests** (`test/system/test_system.ml`) are `S1`–`S20`, end to end
  on a real device.
- **Never weaken a tolerance to make a test pass.** If a number is outside its
  band, that is the finding. Report it rather than widening the band.
- New numeric work needs a differential test against the interpreter, not a
  hand-computed expected value.

## Conventions

- `.mli` for every module with something to hide; the doc comment there is
  the spec. The closed variant types — `Dtype`, `Expr`, `Tensor`,
  `Kernel_ir`, `Pass`, `Backend` — have no `.mli` on purpose: the `.ml` *is*
  the type, and every pass matches on it exhaustively.
- Comments say *why*, not what. The existing ones set the register — they
  explain a constant, a bound, or a decision someone would otherwise undo.
- No trailing whitespace; the codebase is LF (`core.autocrlf` is on for
  Windows working copies, but blobs are LF).
- Names: `cudacaml` for anything the compiler, opam or a shell sees;
  `CUDACaml` in prose and URLs.

## Things that look like bugs and are not

- **saxpy is slower than the vanilla OCaml loop.** It is three flops per
  element, so the measurement is mostly the PCIe round trip. Expected, and
  documented in `bench/results/README.md`. Do not "fix" it.
- **`poly` in f64 is the same speed as f32 on the host side.** An OCaml float
  is a double either way. That is the point of the f64 column.
- **fp64 is ~1/64 of fp32 on GeForce parts and ~1/2 on A100-class parts.** A
  large f32/f64 gap on a laptop card is the hardware, not the compiler.
- **Benchmark times include upload and download.** They characterise the
  end-to-end pipeline, not peak arithmetic.

## Before you claim it works

```
make build && make unit          # always
make system                      # if you touched lower/, runtime/, backend_cuda/
```

Quote the last lines of whichever you ran. If a step was skipped, say so —
`0 failures` from a suite that skipped its GPU half is not evidence.
