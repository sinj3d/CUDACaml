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
| 1 | `cudacaml.ir` | `Uid` `Dtype` `Shape` `Expr` `Tensor` `Value` `Graph` `Dsl` | The IR and the user-facing surface. Types are public variants; passes match on them exhaustively. |
| 2 | `cudacaml.passes` | `Pass` `Layout` `Pipeline` | Graph → Graph rewrites. Pure. Semantics-preserving as judged by the interpreter. |
| 3 | `cudacaml.lower` | `Fusion` `Schedule` `Kernel_ir` `Lower` | Graph → imperative kernels + host plan. `Fusion` is an *analysis* (which nodes get a buffer); `Lower` inlines everything else. **Every performance decision is made here** and is visible in `Kernel_ir`. |
| 4 | `cudacaml.runtime` | `Device` `Buffer` `Jit` `Launch` | Bytes, pointers, PTX strings, kernel handles. IR-agnostic. Wraps `cudajit`. |
| 5 | `cudacaml.backend` | `Backend` `Differential` | The executor signature and the correctness harness. |
| 5 | `cudacaml.backend_interp` | (one module) | Reference evaluator. Runs the *unoptimised* graph. |
| 5 | `cudacaml.backend_cuda` | `Emit` `Mangle` `Executor` + main | Pipeline → Lower → Emit → Jit; Executor per run. |
| — | `cudacaml` | umbrella | Flat namespace for users. |

## The interfaces that matter

- **`Tensor.node`** (layer 1) is the contract between the front end and every
  pass. Eleven variants. Adding one means updating `Tensor.deps`, `Fusion`,
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
6. Every dtype-dependent decision is made from the `Dtype.t` witness in
   `Lower` / `Emit` / `Interp`; no dtype is special-cased as "the fast one".

## Deliberately out of scope

Nested parallelism / flattening, dynamic shapes, boolean *tensors*
(comparisons and `select` exist at the expression level), autotuning, and the
`[%kernel]` ppx (which would desugar to `Dsl` calls and changes nothing below
layer 1). Kernel geometry is a fixed heuristic in `Schedule`, not a search.

## Where to start reading

The layers below are ordered so that each one only depends on the ones above
it, and that is also the order that makes them legible:

1. `lib/ir/tensor.ml` — the eleven node variants. Everything else is a fold
   over them, so this file bounds the whole system.
2. `lib/backend_interp/` — the oracle, and the shortest complete statement of
   what every node *means*.
3. `lib/lower/fusion.ml` then `lower.ml` — where the performance decisions
   live, and the only place they live.
4. `lib/backend_cuda/emit.ml` — a pretty-printer over `Kernel_ir`. If you are
   tempted to make a decision here, it belongs in step 3.

`cudacaml emit <example>` and `cudacaml dot <example>` both run without a GPU
and are the fastest way to see what a graph became.
