# NVIDIA A100-SXM4-40GB

Benchmark results for this card, newest section last. Appended by
`scripts/bench-record.sh`; see `bench/results/README.md` for what
the columns mean.

## 2026-09-20 11:12 UTC

commit `311b283`, n = 1048576, reps = 5 (best of), poly degree = 64

```
device: NVIDIA A100-SXM4-40GB
compute: sm_80
multiprocessors: 108
memory_mib: 40441
```

| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 1048576 | 0.003539 | 0.002825 | 47.14 | 0.5761 | 51.41 |
| f64 | 1048576 | 0.006306 | 0.004746 | 28.06 | 0.3195 | 30.64 |

## 2026-09-20 11:17 UTC

commit `311b283`, n = 16777216, reps = 5 (best of), poly degree = 64

```
device: NVIDIA A100-SXM4-40GB
compute: sm_80
multiprocessors: 108
memory_mib: 40441
```

| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 16777216 | 0.09306 | 0.07816 | 27.26 | 0.4203 | 29.72 |
| f64 | 16777216 | 0.1714 | 0.1446 | 14.73 | 0.2242 | 16.08 |
