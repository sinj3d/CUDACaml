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
linking: `Nvrtc.compile_to_ptx` prepends `$CUDA_PATH/include`. Run `make env`
to see what the build resolved.

## Status

Verified on an RTX 5080 Laptop (sm_120, driver 592.82), OCaml 5.4.0, CUDA 12.9.86:

- unit suite: **89 tests, 0 failures** across eight executables, none skipped
- system suite: **22 tests, 0 failures** — every example differential-tested
  against the interpreter, saxpy at n ∈ {0, 1, 255, 256, 257, 1009, 300000},
  a 4 194 304-element reduction, and 50 consecutive runs of one compiled program
- saxpy at n = 2²⁴: interpreter 1.713 s, CUDA warm 0.171 s (**10×**)

Out of scope in v1: nested parallelism, dynamic shapes, broadcasting, bool
tensors, autotuning, and anything beyond `F32` in the fast path.
