# Benchmark results

One Markdown file per card, named after the device: the name `ocaml-cuda info`
prints, lower-cased, with every run of non-alphanumerics replaced by a dash.

```
NVIDIA A100-SXM4-80GB           ->  nvidia-a100-sxm4-80gb.md
NVIDIA GeForce RTX 5080 Laptop  ->  nvidia-geforce-rtx-5080-laptop.md
```

Write them with `make record` (or `bash scripts/bench-record.sh N REPS DEGREE`),
which appends a dated section per run and never overwrites one. A file is
therefore a history: one section per commit worth recording, oldest first.

Each section carries the date, the commit, the shape of the run, the four
lines of `ocaml-cuda info`, and a two-row table — one row per precision.

## The columns

| Column | Meaning |
|---|---|
| `dtype` | precision of the **device** graphs, `f32` or `f64` |
| `n` | elements per input vector |
| `saxpy_cuda_s` | best-of-REPS wall time of one `Backend_cuda.run` of saxpy |
| `poly_cuda_s` | the same for the degree-64 Horner chain |
| `poly_gflops` | `n * (2*degree - 1) / poly_cuda_s`, so GPU GFLOP/s at that dtype |
| `saxpy_speedup` | vanilla OCaml time / CUDA time |
| `poly_speedup` | the same for poly |

The CUDA times are whole runs — upload, launch, download — so saxpy, at three
flops per element, mostly measures the PCIe bus and is often *slower* than the
vanilla loop. Poly keeps the same traffic and scales the arithmetic, which is
the number worth comparing across cards. JIT compilation is excluded; it is
amortised over runs and is printed separately by the benchmark.

## Reading the two rows

The vanilla loops are identical in both rows: an OCaml float is a double, so
the host side does exactly the same work at `f32` and at `f64`. Everything the
two rows differ by is the card. That ratio is the point of the table: fp64
throughput is roughly 1/64 of fp32 on GeForce parts and about 1/2 on
A100/H100-class parts, so the `f64` row is where a laptop and a datacentre
card stop looking alike.
