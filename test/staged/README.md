# Staged tests

Tests written ahead of the code they exercise. This directory has **no
`dune` file**, so nothing here is built. Each file is promoted to
`test/unit/` (and added to `test/unit/dune`) as the last step of the task
that owns it; the system suite is promoted by T28.

| File | Owner | Promoted when |
|---|---|---|
| `unit/test_info.ml` | T11 | `Runtime.Device.info` and `ocaml-cuda info` exist |
| `unit/test_broadcast.ml` | T13 | `Tensor.Broadcast`, `Dsl.broadcast`, `Dsl.full` |
| `unit/test_rows.ml` | T14 | last-axis `Reduce`/`Scan`, `Dsl.reduce_rows`/`scan_rows`/`transpose`, `Kernel_ir.Block_id` |
| `unit/test_expr_ops.ml` | T15 | bit ops, shifts, `Sin`/`Cos`/`Erf`/`Erfinv`, `Dsl.ne/gt/ge/and_/or_/not_` |
| `unit/test_rng.ml` | T16 | `Rng` library |
| `unit/test_deriv.ml` | T17 | `Deriv` (scalar derivatives, `apply1`/`apply2`, `simplify`) |
| `unit/test_grad.ml` | T18 | `Grad.grad` |
| `unit/test_scatter.ml` | T19 | `Tensor.Scatter_add`, `Kernel_ir.Atomic_add`, Gather adjoint |
| `unit/test_black_scholes.ml` | T20 | `examples/black_scholes.ml` |
| `unit/test_multikernel.ml` | T21 | grid-wide reduce, parallel scan |
| `unit/test_matmul.ml` | T22 | `Tensor.Matmul`, 2-D launch |
| `unit/test_lsm.ml` | T23 | `examples/lsm.ml` |
| `unit/test_pool.ml` | T24 | persistent `Executor.t`, `Buffer.live_count` |
| `unit/test_streams.ml` | T25 | `Runtime.Stream`/`Event`, `Backend_cuda.run_async`/resident |
| `unit/test_pinned.ml` | T26 | `Runtime.Pinned`, `Value.of_raw` |
| `unit/test_multigpu.ml` | T27 | `Backend_cuda.Multi` |
| `system/test_system.ml` | T28 | replaces `test/system/test_system.ml` |

Rules: a staged file is edited only by the task that owns it, and only to
fix a compile error against the interfaces its spec defines (rule 2 in
`tasks/README.md`). Assertions are never weakened.
