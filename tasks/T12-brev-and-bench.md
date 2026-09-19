# T12 — brev.dev bootstrap, benchmark dtype flag, results capture

## Goal

Make a fresh A100 box on brev.dev go from empty to `make test && make bench`
with one script, and make the benchmark produce a row for a results table
that distinguishes cards and precisions. No compiler changes.

Depends on: T11 (for `ocaml-cuda info`). Phase 0.

## Files you own

- `scripts/brev-setup.sh` (new)
- `scripts/bench-record.sh` (new)
- `bench/bench.ml`, `bench/dune`
- `bench/results/README.md` (new; the directory holds one Markdown file per card)
- `Makefile` (only the `bench` target and a new `record` target)

## `scripts/brev-setup.sh`

Idempotent bash, `set -euo pipefail`, runs as the brev user with sudo.
Ubuntu 22.04 or 24.04.

1. `apt-get install -y build-essential git m4 pkg-config libffi-dev opam`
   (plus `unzip bubblewrap` which opam wants). Skip if present.
2. Detect the CUDA toolkit: `nvcc --version` or `/usr/local/cuda/version.json`.
   If the major version is 13, install 12.9 side by side with the runfile
   (`--silent --toolkit --toolkitpath=$HOME/cuda-12.9 --no-drm --override`)
   and print `export CUDA_PATH=$HOME/cuda-12.9`. The reasons are in
   README.md ("Use CUDA 12.x, not 13.x"). The A100 is sm_80, so any 12.x
   works; the 12.8 floor is a Blackwell fact and does not apply.
3. `opam init --disable-sandboxing -y` if `~/.opam` is absent;
   `opam switch create ocaml-cuda 5.4.0` if the switch is absent;
   `opam install -y dune cudajit`.
4. Clone or pull `https://github.com/<user>/ocaml-cuda` into `~/ocaml-cuda`.
   Take the URL from `$OCAML_CUDA_REPO` with that default; a user's fork
   should not require editing the script.
5. `make env && make build && make unit`. Print the next steps:
   `make system`, `make record`.

Every step prints what it is about to do and what it found (`found opam
2.1.5`, `toolkit 12.9 at /usr/local/cuda`). If GCC ≥ 14 is detected, print
the README's note about cudajit's stubs and how to relax the warning.

## `bench/bench.ml`

Add a fourth optional argument, `dtype`, `f32` (default) or `f64`. Both
workloads build their graphs at that dtype and the vanilla loop is
unchanged (OCaml floats are doubles either way). Print the dtype in the
header line and per-workload lines. Also print, after both workloads, one
machine-readable line:

```
RESULT card="<Device.name>" dtype=f64 n=16777216 saxpy_cuda_s=0.171 poly_cuda_s=0.412 poly_gflops=... saxpy_speedup=... poly_speedup=...
```

Numbers with `%.4g`. The `RESULT` line is what `bench-record.sh` greps.

Keep `agree` and the exit-1-on-mismatch behaviour.

## `scripts/bench-record.sh`

Runs `dune exec ocaml-cuda -- info`, then `dune exec ocaml-cuda-bench --
$N $REPS $DEGREE f32` and the same with `f64`, and appends a section to
`bench/results/<slug>.md` where slug is the device name lower-cased with
non-alphanumerics replaced by `-` (`nvidia-a100-sxm4-80gb`). The section
has the date, the four `info` lines, and a two-row table (f32, f64) with
the fields of the `RESULT` line. Never overwrite a file; append.

`Makefile`:

```
record: build
	scripts/bench-record.sh $(BENCH_ARGS)
```

## Failure modes to avoid

- A script that assumes `/usr/local/cuda` exists.
- Hard-coding the repo URL or the switch name in more than one place.
- Widening the `agree` tolerance in the f64 path: f64 sums agree to far
  better than 1e-3, so the same check passes.
- Vanilla poly in f64 is the same speed as f32: that is expected and is the
  point of the fp64 column. Do not "fix" it.

## Verify

```
bash -n scripts/brev-setup.sh && bash -n scripts/bench-record.sh
dune build 2>&1 && make unit
dune exec ocaml-cuda-bench -- 1048576 3 64 f64 | grep '^RESULT'   # GPU machine
```

## Tests

No unit test: nothing here is a library. The verify block is the test.
Run the setup script on a brev A100 once and paste its output in the
report; T28 needs the resulting `bench/results/` file.
