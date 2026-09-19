# T28 — System suite v2 and the A100 campaign (GATED)

## Gate

Do **not** start until all of the following hold:

1. `make unit` exits 0 on the GPU machine with every T11–T27 test promoted
   and **nothing skipped**.
2. Every task T11–T27 has a report with a green verify block.
3. `dune exec ocaml-cuda -- check <ex>` is `ok` for every example in
   `ocaml-cuda list`, including `bs_mc`, `bs_paths`, `bs_greeks`.
4. `scripts/brev-setup.sh` has run on a brev.dev A100 instance and
   `bench/results/` contains that card's file (T12).

## Files you own

- `test/system/test_system.ml` ← replaced by `git mv test/staged/system/test_system.ml test/system/test_system.ml`
- `test/system/dune` (add `ocaml_cuda_examples` if missing; it is already there)
- `README.md` Status section and a new "Results" table; `bench/results/*.md`
- `test/staged/README.md` (empty the table; leave the rules)

## What runs

`make system 2>&1 | tee system.log` on **both** the RTX 5080 (WSL) and
the A100 (brev). Both logs go in the report.

| ID | What | Pass criterion |
|---|---|---|
| S1 | every example in `Programs.all`, interp vs cuda | Ok at 1e-5 |
| S2 | saxpy at n ∈ {0, 1, 255, 256, 257, 1009, 300000} | Ok |
| S3 | sum of 4 194 304 elements (now the two-kernel reduce) | Ok at 1e-3 |
| S4 | chain of five maps | 1 kernel and Ok |
| S5 | one compiled saxpy run 50× | correct sums, and `live_count` constant (T24) |
| S6 | i32 negative division | Ok, exact |
| S7 | timing saxpy 2^24 | informational |
| S8 | `Rng.u32` over 2^20 elements, interp vs cuda | Ok, **exact** (I32) |
| S9 | `Rng.normal F32` over 2^20: mean, var on device | mean within 0.01, var within 0.02 |
| S10 | `reduce_rows`/`scan_rows` over `[1024; 4096]` F32 | Ok at 1e-3 |
| S11 | `matmul` 256×192 · 192×128, F32 and F64 | Ok at 1e-4 (F32), 1e-9 (F64) |
| S12 | `scatter_add` histogram with heavy collisions, I32 and F32 | Ok exact (I32), 1e-4 (F32) |
| S13 | `bs_greeks` F64 n=65536, interp vs cuda | Ok at 1e-6 |
| S14 | `bs_greeks` F32 n=2^20 on device vs analytic | delta within 0.01, vega within 0.6, rho within 0.6 |
| S15 | `run_async` 16 jobs on 4 streams = sync results | exact |
| S16 | resident chain `chain → sum` = interpreter | Ok at 1e-5 |
| S17 | `Lsm.Make_cuda_resident` n=65536, 50 steps vs binomial | within 0.15 |
| S18 | pinned vs pageable round trip | exact; bandwidth printed |
| S19 | `Multi ~devices:[0;0]` = single device | exact I32 / 1e-6 F32; `[0;1]` when ≥ 2 devices |
| S20 | timing table: bs_mc F32 vs F64 at 2^22, greeks/price ratio, per-run ms with persistent executor | informational |

Expected last line: `N tests, 0 failures`, no SKIP lines on either machine
except S19's multi-device half on the 5080.

## README

Add a "Results" section with one table per card (from `bench/results/`),
columns: workload, f32 time, f64 time, f64/f32 ratio, speedup vs OCaml. Add
the S14 Greeks line and the S17 LSM line under "Verified". State the
gradient-to-forward cost ratio measured, not assumed. Update the Status
list to cover both cards with driver and toolkit versions from
`ocaml-cuda info` and `nvidia-smi`.

## Triage table

| Symptom | Owner |
|---|---|
| S8 mismatch | T15 (shift/xor emit) or T16 (mulhilo zero-extension) |
| S9 out of band but S8 exact | T16 `to_unit_interval` / `normal_inv_cdf`, or T15 `erfinvf` |
| S10 mismatch, S3 ok | T21 rows indexing in the partial kernel |
| S11 F32 fails at 1e-4 but F64 ok | T22 accumulation order or tile bounds; check with k a multiple of 16 |
| S12 F32 fails, I32 ok | tolerance: atomics reorder; 1e-4 is generous, so look at the zero-fill ordering (T19) |
| S13 mismatch | T18/T19 adjoint on device vs interp: bisect by output name; each gradient is its own output |
| S14 off but S13 ok | T20 seed handling or an f32 precision issue in `normal_inv_cdf` tails |
| S15 mismatch | T25 `wait` not waiting, or buffer sharing across streams |
| S16 mismatch | T25 resident pointer substitution |
| S17 off by > 0.15 | T23 regression on ITM paths; compare with `Make (Backend_interp)` |
| S19 `[0;1]` mismatch | T27 `with_device` context handling |
| S5 `live_count` drifts | T24 |
| NVRTC error | T15/T19/T22 emit; `ocaml-cuda emit <ex>` |

## Report

Both `system.log`s, both `ocaml-cuda info` outputs, `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
from both machines, the S20 tables, and the README diff.
