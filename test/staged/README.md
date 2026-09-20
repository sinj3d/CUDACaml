# Staged tests

Tests written ahead of the code they exercise. This directory has **no
`dune` file**, so nothing here is built. Each file is promoted to
`test/unit/` (and added to `test/unit/dune`) as the last step of the task
that owns it; the system suite is promoted by T28.

| File | Owner | Promoted when |
|---|---|---|

Nothing is staged: every v2 unit test has been promoted to `test/unit/`, and
`system/test_system.ml` was promoted by T28, replacing `test/system/test_system.ml`.

Rules: a staged file is edited only by the task that owns it, and only to
fix a compile error against the interfaces its spec defines (rule 2 in
`tasks/README.md`). Assertions are never weakened.
