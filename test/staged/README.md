# Staged tests

Tests written ahead of the code they exercise. This directory has **no
`dune` file**, so nothing here is built. Each file is promoted to
`test/unit/` (and added to `test/unit/dune`) as the last step of the task
that owns it; the system suite is promoted by T28.

| File | Owner | Promoted when |
|---|---|---|
| `unit/test_pool.ml` | T24 | persistent `Executor.t`, `Buffer.live_count` |
| `unit/test_streams.ml` | T25 | `Runtime.Stream`/`Event`, `Backend_cuda.run_async`/resident |
| `unit/test_pinned.ml` | T26 | `Runtime.Pinned`, `Value.of_raw` |
| `unit/test_multigpu.ml` | T27 | `Backend_cuda.Multi` |
| `system/test_system.ml` | T28 | replaces `test/system/test_system.ml` |

Rules: a staged file is edited only by the task that owns it, and only to
fix a compile error against the interfaces its spec defines (rule 2 in
`tasks/README.md`). Assertions are never weakened.
