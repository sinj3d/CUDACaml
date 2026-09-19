# Architecture

An embedded array DSL in OCaml that compiles to CUDA. One IR, two executors:
a CUDA JIT and a pure-OCaml reference interpreter, checked against each other.

The shape is borrowed from Hardcaml (typed DAG, small closed IR, a
language-neutral intermediate form, emitter/simulator duality, out-of-tree
backends behind one signature). The semantics are not: Hardcaml's graph is a
timeless circuit; ours lowers to loops, thread indices and a memory hierarchy.

## Data flow

```
user code ──Dsl──▶ Graph.t ──Pipeline──▶ Graph.t ──Lower──▶ Kernel_ir.program
                      │                                          │
                      │                                    Emit  │  (CUDA C++ text)
                      │                                          ▼
                      │                                  Runtime.Jit ──▶ module
                      │                                          │
                      ▼                                          ▼
              Backend_interp.run                          Executor.run (host plan)
                      │                                          │
                      └──────────── Differential.check ──────────┘
```

## Layers

Dependencies point strictly downward. Each layer is its own dune library so a
violation is a build error, not a code-review catch.

| # | Library | Modules | Owns |
|---|---|---|---|
| 1 | `ocaml_cuda.ir` | `Uid` `Dtype` `Shape` `Expr` `Tensor` `Value` `Graph` `Dsl` | The IR and the user-facing surface. Types are public variants; passes match on them exhaustively. |
| 2 | `ocaml_cuda.passes` | `Pass` `Layout` `Pipeline` | Graph → Graph rewrites. Pure. Semantics-preserving as judged by the interpreter. |
| 3 | `ocaml_cuda.lower` | `Fusion` `Schedule` `Kernel_ir` `Lower` | Graph → imperative kernels + host plan. `Fusion` is an *analysis* (which nodes get a buffer); `Lower` inlines everything else. **Every performance decision is made here** and is visible in `Kernel_ir`. |
| 4 | `ocaml_cuda.runtime` | `Device` `Buffer` `Jit` `Launch` | Bytes, pointers, PTX strings, kernel handles. IR-agnostic. Wraps `cudajit`. |
| 5 | `ocaml_cuda.backend` | `Backend` `Differential` | The executor signature and the correctness harness. |
| 5 | `ocaml_cuda.backend_interp` | (one module) | Reference evaluator. Runs the *unoptimised* graph. |
| 5 | `ocaml_cuda.backend_cuda` | `Emit` `Mangle` `Executor` + main | Pipeline → Lower → Emit → Jit; Executor per run. |
| — | `ocaml_cuda` | umbrella | Flat namespace for users. |

## The interfaces that matter

- **`Tensor.node`** (layer 1) is the contract between the front end and every
  pass. Eight variants. Adding one means updating `Tensor.deps`, `Fusion`,
  `Lower` and `Backend_interp`; the compiler will list them.
- **`Expr.fn1` / `fn2`** is how user closures enter the IR: applied once to
  `Arg` placeholders at construction time, never stored. The IR stays
  first-order and inspectable.
- **`Pass.S`** = `{ name; run : Graph.t -> Graph.t }`. Pipeline is a fold.
- **`Fusion.plan`** decides materialisation only: Params, `Reduce`/`Scan`,
  outputs, and anything with fan-out ≥ 2 get a buffer; every other
  element-wise node is inlined into its consumer's kernel by `Lower`. This
  gives map/map2/reduce fusion without an n-ary node in the IR.
- **`Kernel_ir.program`** = kernels + `host_op` plan. `Emit` prints the
  kernels; `Executor` walks the plan. Neither makes decisions.
- **`Backend.S`** = `{ compile : Graph.t -> compiled; run : compiled -> inputs -> outputs }`.
  Everything above the backends (CLI, tests, benchmarks) sees only this.
- **`Value.t`** is the only host tensor type: a Bigarray, so it never moves
  under the GC and uploads are pointer + byte count.

## Invariants worth defending

1. Parallelism is expressed only through `Tensor` combinators. No loop is
   ever proved parallel.
2. Shapes are static per `Graph.t`. Re-build the graph for a new size.
3. `Backend_interp` never runs passes. It is the oracle, so it must stay
   independent of what it judges.
4. `Emit` is a pretty-printer. A choice that has to be made there belongs in
   `Lower` or `Schedule`.
5. `runtime` never imports `Graph` or `Kernel_ir`.

## Deliberately out of scope (v1)

Nested parallelism / flattening, dynamic shapes, broadcasting, bool tensors,
autotuning, anything beyond `F32` in the fast path, and the `[%kernel]` ppx
(which would desugar to `Dsl` calls and changes nothing below layer 1).

## Suggested order of work

1. `runtime` against `cudajit`: hand-written string kernel compiles and
   launches end to end. This de-risks the whole project; do it first.
2. `Value`, `Graph.topological_order`, `Backend_interp`. Now `saxpy` runs
   on the CPU and the oracle exists.
3. `Lower` + `Emit` for `Map`/`Map2`/`Iota` with a grid-stride loop, then
   `Executor`. First GPU result; `Differential` goes green.
4. `Fusion`. The demo number.
5. `Reduce` (block tree reduction, `__shfl_down_sync`), then `Scan`, `Gather`.
6. ppx, if time remains.
