# ocaml-cuda

An embedded array DSL in OCaml that compiles to CUDA. One IR, two executors —
a CUDA JIT and a pure-OCaml reference interpreter — checked against each other.

```ocaml
let x = param "x" Dtype.F32 (vec n) in
let y = param "y" Dtype.F32 (vec n) in
let r = map2 add (map (fun v -> mul (const Dtype.F32 2.0) v) x) y in
Graph.create ~name:"saxpy" ~outputs:[ ("r", Tensor.P r); ("s", Tensor.P (reduce add ~init:zero r)) ]
```

`Dsl` builds a typed DAG, a fusion analysis decides which nodes need a buffer,
`Lower` turns the rest into imperative kernels, and `Emit` prints CUDA C++ that
NVRTC compiles at run time. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
layering and the invariants that hold it together.

## What fusion buys you

Five chained element-wise maps become **one** kernel with no intermediate
buffers — `dune exec ocaml-cuda -- emit chain`:

```cuda
extern "C" __global__ void k_18(float* p_x, float* t18) {
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x); i < 4096; i += (gridDim.x * blockDim.x)) {
    t18[i] = (((-((p_x[i] + 1.0f) * (p_x[i] + 1.0f))) - 1.0f) * 0.5f);
  }
}
```

and `dune exec ocaml-cuda -- check chain` proves it still agrees with the
interpreter.

## CLI

| Command | Does |
|---|---|
| `ocaml-cuda list` | names of the bundled example programs |
| `ocaml-cuda emit <ex>` | the generated CUDA C++ (pipe it to `nvcc -ptx` to debug) |
| `ocaml-cuda dot <ex>` | the graph as Graphviz |
| `ocaml-cuda run <ex>` | run on the GPU and print the outputs |
| `ocaml-cuda check <ex>` | differential-test the GPU against the interpreter |
| `ocaml-cuda info` | the device description (name, compute capability, SMs, memory) |

`emit` and `dot` need no GPU.

## Building

```
make build     # compile
make unit      # unit tests; GPU-dependent ones skip themselves without a device
make system    # gated end-to-end suite (needs a GPU)
```

### The CUDA toolchain

`lib/runtime` binds the CUDA **driver** API and NVRTC through
[`cudajit`](https://github.com/lukstafi/ocaml-cudajit). Two version constraints
are easy to trip over and neither fails in an obvious way:

- **Use CUDA 12.x, not 13.x.** In 13.x `cuCtxCreate` maps to `cuCtxCreate_v4`
  (4 arguments, was 3) and `nvrtcCompileProgram` / `cuStreamGetId` changed
  signatures, so cudajit 0.7.2 fails to build against its headers. On a
  Blackwell card (`sm_120`) you also need at least 12.8, which leaves 12.8–12.9.
- **GCC 14 rejects cudajit's generated stubs**, because it promotes
  `-Wincompatible-pointer-types` to an error. The two offending casts are benign
  const/typedef mismatches; build with that warning relaxed.

`libcuda.so.1` is resolved by the system loader, but `libnvrtc` ships with the
toolkit. If it is not on the default loader path, point `CUDA_PATH` at your
install — the `Makefile` picks it up and puts the right directory on
`LD_LIBRARY_PATH`. `CUDA_PATH` must be set at **run** time too, not just when
linking: `Nvrtc.compile_to_ptx` prepends `$CUDA_PATH/include`.

`ctypes-foreign` links `-lffi`, normally from `libffi-dev`. If you cannot
install it as root, stage it under a prefix and set `LOCAL_PREFIX` to point
there. Both variables are optional and are no-ops when the paths do not exist.

Run `make env` to see what the build resolved — the first thing to check when a
link fails on `-lffi` or `-lnvrtc`, or when a GPU test skips unexpectedly.

## Status

Verified on an **RTX 5080 Laptop GPU** — `sm_120`, 60 SMs, 16302 MiB, driver
592.82 (`nvidia-smi`), CUDA 12.9.86, OCaml 5.4.0, WSL2:

- unit suite: **229 tests, 0 failures** across 24 executables. One `SKIP`: the
  `devices >= 2` half of the multi-GPU suite, which this one-card machine
  cannot exercise.
- system suite (`make system`): **38 tests, 0 failures**, S1–S20.
- `ocaml-cuda check <ex>` is `ok` for all 13 bundled examples, `bs_mc`,
  `bs_paths` and `bs_greeks` among them.

An **A100** run is **pending**: `scripts/brev-setup.sh` has not yet been run on
a brev.dev instance, so there is no `bench/results/nvidia-a100-*.md` and no
A100 column below. Nothing here is extrapolated to that card.

### Verified

- saxpy at n = 2²⁴: interpreter 1.376 s, CUDA warm 0.100 s (**13.8×**) — S7.
- every example differential-tested against the interpreter, saxpy at
  n ∈ {0, 1, 255, 256, 257, 1009, 300000}, a 4 194 304-element two-kernel
  reduction, and 50 consecutive runs of one compiled program with a constant
  buffer `live_count` — S1–S5.
- `Rng.u32` is **bit-exact** against the interpreter over 2²⁰ draws; `Rng.normal`
  F32 over 2²⁰ has mean 0.0005 and variance 1.0009 on device — S8, S9.
- **Greeks**, `bs_greeks` F32 at 2²⁰ paths on device against Black–Scholes
  closed form: price 10.4741 (10.4506), delta 0.6375 (0.6368), vega 37.645
  (37.524), rho 53.281 (53.232). The F64 graph at 65 536 paths agrees with the
  interpreter to 1e-6 — S13, S14.
- **LSM**, `Lsm.Make_cuda_resident` American put, 65 536 paths × 50 steps:
  6.0176 against a 500-step binomial 6.0888, a gap of 0.071 in 0.459 s — S17.
- 16 async jobs over 4 streams reproduce the synchronous results exactly, and a
  resident `chain → sum` chain matches the interpreter — S15, S16.
- `Multi ~devices:[0;0]` reproduces the single-device result exactly — S19.

## Results

RTX 5080 Laptop GPU, commit `24fb96e`. Times are per run, upload and download
included, JIT excluded.

| workload | f32 | f64 | f64/f32 | speedup vs OCaml (f32 / f64) |
|---|---:|---:|---:|---|
| saxpy, n = 2²⁰ | 1.96 ms | 3.71 ms | 1.89× | 0.36× / 0.18× |
| poly degree 64, n = 2²⁰ | 1.36 ms | 2.77 ms | 2.03× | 34.3× / 16.9× |
| bs_mc price, n = 2²² | 0.4 ms | 2.0 ms | 5.7× | — |
| bs_greeks, n = 2²² | 10.0–10.2 ms | 13.3–16.2 ms | 1.3–1.6× | — |

The first two rows are `make record` (best of 5, n = 1 048 576); the full
section with GFLOP/s is in
[`bench/results/nvidia-geforce-rtx-5080-laptop-gpu.md`](bench/results/nvidia-geforce-rtx-5080-laptop-gpu.md).
Poly is the row worth comparing across cards: it keeps saxpy's bus traffic and
scales the arithmetic, so saxpy at three flops per element is mostly PCIe and
is *slower* than the vanilla OCaml loop. The last two rows are S20 of the
system suite (mean of 5 runs through the persistent executor, n = 4 194 304),
given as the range over two consecutive suite runs; the two `bs_` rows have no
hand-written OCaml counterpart, hence no speedup.

**Gradient-to-forward cost ratio, measured.** The reverse-mode adjoint of
`bs_mc` — four Greeks from one backward pass — costs, at n = 2²², **28.4–28.7×
the forward price** at f32 and **6.6–8.1×** at f64 (S20, mean of 5, two suite
runs). The same two graphs timed best-of-5 by `ocaml-cuda-bench` at the same n
and dtype give **2.7×**. Both are real measurements: the adjoint graph's 25
kernels churn the buffer pool, so its mean is an order of magnitude above its
best case, and a desk should budget for the mean. No ratio here is assumed
from the graph shape.

Host transfer, 64 MiB round trip (S18): pinned 10.6 and 3.2 GB/s H→D over two
runs against pageable 3.5 and 3.1 GB/s. Only the pinned figure moves; at its
best it is about 3× the pageable path, and the low reading is a cold staging
buffer, not a property of the bus.

### Precision

`F32` and `F64` go through the same pipeline — the same fusion, the same
lowering, the same emitted kernel shape — and both are differential-tested
against the interpreter; an f64 graph emits `double`. `I32` and `I64` are
supported for element-wise and reduction ops. The dtype is chosen per tensor
at construction, e.g. `param "x" Dtype.F64 (vec n)`.

What differs between `F32` and `F64` is not the compiler but the card. fp64
*throughput* is roughly 1/64 of fp32 on GeForce parts and about 1/2 on
A100/H100-class parts, so an f64 kernel that is correct on a laptop may still
be the wrong choice there.

Out of scope in v1: nested parallelism, dynamic shapes, broadcasting, bool
tensors, and autotuning.
