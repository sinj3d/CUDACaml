# NVIDIA GeForce RTX 5080 Laptop GPU

Benchmark results for this card, newest section last. Appended by
`scripts/bench-record.sh`; see `bench/results/README.md` for what
the columns mean.

## 2026-09-19 21:27 UTC

commit `3e85a08`, n = 1048576, reps = 3 (best of), poly degree = 64

```
device: NVIDIA GeForce RTX 5080 Laptop GPU
compute: sm_120
multiprocessors: 60
memory_mib: 16302
```

| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 1048576 | 0.006629 | 0.00505 | 26.37 | 0.1127 | 9.554 |
| f64 | 1048576 | 0.01019 | 0.006828 | 19.5 | 0.0696 | 6.977 |

## 2026-09-20 05:46 UTC

commit `24fb96e`, n = 1048576, reps = 5 (best of), poly degree = 64

```
device: NVIDIA GeForce RTX 5080 Laptop GPU
compute: sm_120
multiprocessors: 60
memory_mib: 16302
```

| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 1048576 | 0.001959 | 0.001362 | 97.79 | 0.3573 | 34.3 |
| f64 | 1048576 | 0.003711 | 0.002765 | 48.16 | 0.1822 | 16.85 |

## 2026-09-20 11:26 UTC

commit `311b283`, n = 16777216, reps = 5 (best of), poly degree = 64

```
device: NVIDIA GeForce RTX 5080 Laptop GPU
compute: sm_120
multiprocessors: 60
memory_mib: 16302
```

| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 16777216 | 0.04904 | 0.04031 | 52.86 | 0.2849 | 18.61 |
| f64 | 16777216 | 0.09576 | 0.157 | 13.57 | 0.1491 | 4.741 |
