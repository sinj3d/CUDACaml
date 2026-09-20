# CUDACaml

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
layering and the invariants that hold it together, and
[SKILLS.md](SKILLS.md) if you are pointing a coding agent at this repo.

Every graph runs two ways. The CUDA path is the product; the interpreter is the
oracle. `Differential.check` runs both and compares, which is how every claim on
this page was produced.

## What fusion buys you

Five chained element-wise maps become **one** kernel with no intermediate
buffers — `dune exec cudacaml -- emit chain`:

```cuda
extern "C" __global__ void k_18(float* p_x, float* t18) {
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x); i < 4096; i += (gridDim.x * blockDim.x)) {
    t18[i] = (((-((p_x[i] + 1.0f) * (p_x[i] + 1.0f))) - 1.0f) * 0.5f);
  }
}
```

and `dune exec cudacaml -- check chain` proves it still agrees with the
interpreter.

## What is in the library

| Area | What you get |
|---|---|
| **Tensor ops** | `map`, `map2`, `reduce`, `scan`, `reduce_rows`, `scan_rows`, `gather`, `scatter_add`, `reshape`, `transpose`, `broadcast`, `matmul` |
| **Element ops** | arithmetic, `min`/`max`, `sqrt`/`exp`/`log`, `sin`/`cos`, `erf`/`erfinv`, bit ops and shifts, comparisons, `select`, `cast` |
| **Reverse-mode AD** | `Grad.grad : Graph.t -> output:string -> wrt:string list -> Graph.t`. A graph-to-graph pass: the adjoint is an ordinary graph, so it fuses, lowers and runs like any other |
| **RNG** | Philox-4x32-10 as a graph — `Rng.u32`, `uniform`, `normal`. Counter-based, so it is reproducible and needs no state |
| **Dtypes** | `F32`, `F64`, `I32`, `I64` through the same pipeline; an f64 graph emits `double` |
| **Runtime** | persistent executor with a buffer pool, streams and events, async runs, device-resident values, pinned host memory, per-device contexts |
| **Multi-GPU** | `Multi` — data-parallel execution of one graph across devices, with `concat` / `mean_scalar` to recombine |
| **Examples** | Black–Scholes Monte Carlo with pathwise Greeks, Longstaff–Schwartz American pricing, and ten smaller programs |

## Building

```
make build     # compile
make unit      # unit tests; GPU-dependent ones skip themselves without a device
make system    # gated end-to-end suite (needs a GPU)
make bench     # hand-written OCaml against the CUDA backend
make record    # bench both precisions, append to bench/results/<card>.md
```

You need OCaml >= 5.1, `dune` >= 3.0, and [`cudajit`](https://github.com/lukstafi/ocaml-cudajit)
(`opam install dune cudajit`). `scripts/brev-setup.sh` takes a fresh Ubuntu GPU
box from empty to a green `make unit` in one command, if you want a cloud GPU
to try it on.

### The CUDA toolchain

`lib/runtime` binds the CUDA **driver** API and NVRTC through `cudajit`. Two
version constraints are easy to trip over and neither fails in an obvious way:

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

## CLI

| Command | Does |
|---|---|
| `cudacaml list` | names of the bundled example programs |
| `cudacaml emit <ex>` | the generated CUDA C++ (pipe it to `nvcc -ptx` to debug) |
| `cudacaml dot <ex>` | the graph as Graphviz |
| `cudacaml run <ex>` | run on the GPU and print the outputs |
| `cudacaml check <ex>` | differential-test the GPU against the interpreter |
| `cudacaml info` | the device description (name, compute capability, SMs, memory) |

`emit` and `dot` need no GPU.

## Verified

Run on two cards. `S1`–`S20` below are the test IDs in
[`test/system/test_system.ml`](test/system/test_system.ml); `make system` prints
them.

| | RTX 5080 Laptop GPU | A100-SXM4-40GB ×2 |
|---|---|---|
| Compute | `sm_120`, 60 SMs, 16302 MiB | `sm_80`, 108 SMs, 40441 MiB |
| Driver / CUDA | 592.82 / 12.9.86 | 595.91.07 / 12.9 |
| Host | OCaml 5.4.0, WSL2 | OCaml 5.4.0, Ubuntu 22.04 |
| Unit suite | 229 tests, 0 failures, 24 executables | 231 tests, 0 failures, 24 executables |
| System suite | **38 tests, 0 failures** | **38 tests, 0 failures** |
| Skips | the `devices >= 2` multi-GPU half | **none** |

The two unit counts differ by exactly the two multi-GPU tests a one-card
machine cannot run.

- every bundled example differential-tested against the interpreter; saxpy at
  n ∈ {0, 1, 255, 256, 257, 1009, 300000}; a 4 194 304-element two-kernel
  reduction; 50 consecutive runs of one compiled program with a constant buffer
  `live_count` — S1–S5.
- `Rng.u32` is **bit-exact** against the interpreter over 2²⁰ draws; `Rng.normal`
  F32 over 2²⁰ has mean 0.0005 and variance 1.0009 on device — S8, S9.
- **Greeks**, `bs_greeks` F32 at 2²⁰ paths on device against Black–Scholes
  closed form: price 10.4741 (10.4506), delta 0.6375 (0.6368), vega 37.645
  (37.524), rho 53.281 (53.232). The F64 graph at 65 536 paths agrees with the
  interpreter to 1e-6 — S13, S14.
- **LSM**, `Lsm.Make_cuda_resident` American put, 65 536 paths × 50 steps:
  6.0176 against a 500-step binomial 6.0888, a gap of 0.071 — S17.
- 16 async jobs over 4 streams reproduce the synchronous results exactly, and a
  resident `chain → sum` chain matches the interpreter — S15, S16.
- `Multi ~devices:[0;0]` reproduces the single-device result exactly, and on the
  two-card box `~devices:[0;1]` does too — S19.

## Results

### Where the datacentre card earns its keep

The reverse-mode adjoint of `bs_mc` — four Greeks from one backward pass, 25
kernels — at n = 2²², f32, through the persistent executor (S20):

| | forward price | Greeks (adjoint) | gradient / forward |
|---|---:|---:|---:|
| RTX 5080 Laptop | 0.4 ms | 10.0–10.2 ms | **~28×** |
| A100-SXM4-40GB | 0.2 ms | 0.7 ms | **~4×** |

The forward price is about twice as fast on the A100; the gradient is about
**fourteen** times faster. The adjoint graph churns the buffer pool, and that is
where the two memory systems stop looking alike. If you are pricing with AAD,
this row is the one that matters — and it is measured, not inferred from the
graph shape.

### Throughput

`make record` at n = 2²⁴, best of 5, commit `311b283`. `poly` is a degree-64
Horner chain; it keeps saxpy's bus traffic and scales the arithmetic, so it is
the row worth comparing across cards.

| card | poly f32 | poly f64 | f64/f32 |
|---|---:|---:|---:|
| RTX 5080 Laptop | 52.86 GFLOP/s | 13.57 GFLOP/s | **3.9×** |
| A100-SXM4-40GB | 27.26 GFLOP/s | 14.73 GFLOP/s | **1.85×** |

The f64 column is the honest one. fp64 throughput is roughly 1/64 of fp32 on
GeForce parts and about 1/2 on A100-class parts, and that is exactly what the
ratios show: the laptop card loses 3.9× going to double precision, the A100
1.85×. On f32 the laptop card's clocks win; on f64 they do not.

Two caveats, because they are easy to misread:

- **These are whole runs — upload, launch, download.** At n = 2²⁴ roughly 40% of
  the measured time is the PCIe round trip at ~4.8 GB/s pageable, so this
  characterises the end-to-end pipeline, not the card's peak arithmetic. That is
  deliberate: it is what a caller actually experiences.
- **Speedup-vs-OCaml columns in `bench/results/` do not compare across cards.**
  The baseline is the host CPU, and the two machines have different ones.

Full per-card history, with saxpy rows and both shapes, is in
[`bench/results/`](bench/results/). Host transfer, 64 MiB round trip (S18):
pinned 12.3 GB/s against pageable 4.8 GB/s on the A100.

### Precision

`F32` and `F64` go through the same pipeline — the same fusion, the same
lowering, the same emitted kernel shape — and both are differential-tested
against the interpreter. `I32` and `I64` are supported for element-wise and
reduction ops. The dtype is chosen per tensor at construction, e.g.
`param "x" Dtype.F64 (vec n)`.

What differs between `F32` and `F64` is not the compiler but the card, which is
what the table above measures.

## Limitations

Not supported: nested parallelism, dynamic shapes, boolean *tensors*
(comparisons and `select` exist at the expression level), and autotuning. Kernel
geometry is chosen by a fixed heuristic, not searched.

## Repository layout

```
lib/ir          typed DAG, dtypes, shapes, the expression language
lib/passes      graph-to-graph passes and the pipeline that orders them
lib/lower       fusion, scheduling, lowering to Kernel_ir
lib/backend     the backend signature and Differential (the two-executor check)
lib/backend_cuda   CUDA C++ emission, the executor, multi-GPU
lib/backend_interp pure-OCaml reference interpreter
lib/runtime     driver API and NVRTC bindings: jit, launch, streams, buffers
lib/ad          reverse-mode AD over the graph
lib/rng         Philox counter-based RNG
examples/       Black-Scholes, Longstaff-Schwartz, and smaller programs
test/unit       one executable per module
test/system     the gated end-to-end suite (S1-S20)
bench/          benchmark driver and per-card recorded results
```

## License

[MIT](LICENSE).
