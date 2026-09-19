#!/usr/bin/env bash
#
# Run the benchmark at both precisions and append one dated section to
# bench/results/<card slug>.md, where the slug is the device name lower-cased
# with every run of non-alphanumerics replaced by a dash:
#
#     NVIDIA A100-SXM4-80GB  ->  bench/results/nvidia-a100-sxm4-80gb.md
#
# Existing files are never overwritten, only appended to: the point of the
# directory is the history of what this card did across commits.
#
#     make record                               # defaults
#     make record BENCH_ARGS="1048576 3 64"     # n, reps, poly degree
#     bash scripts/bench-record.sh 1048576 3 64
#
# Run it through `make record` unless CUDA_PATH and LD_LIBRARY_PATH are
# already right: the Makefile is what points the loader at libnvrtc.
#
set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-16777216}"
REPS="${2:-5}"
DEGREE="${3:-64}"

# --- device -----------------------------------------------------------------

if ! info="$(dune exec ocaml-cuda -- info)"; then
  echo "bench-record: 'ocaml-cuda info' failed; is there a device, and is" >&2
  echo "CUDA_PATH set? Run 'make env' to see what the build resolved." >&2
  exit 1
fi

card="$(printf '%s\n' "$info" | sed -n 's/^device: //p')"
if [ -z "$card" ]; then
  echo "bench-record: no 'device:' line in the output of 'ocaml-cuda info'" >&2
  exit 1
fi

slug="$(printf '%s' "$card" |
  tr '[:upper:]' '[:lower:]' |
  tr -c 'a-z0-9' '-' |
  tr -s '-' |
  sed -e 's/^-//' -e 's/-$//')"

out="bench/results/$slug.md"

# --- benchmark --------------------------------------------------------------

# One RESULT line per dtype. The benchmark exits non-zero when the two paths
# disagree, and `set -e` makes that abort before anything is written.
declare -A result
for dtype in f32 f64; do
  echo "==> ocaml-cuda-bench $N $REPS $DEGREE $dtype"
  log="$(dune exec ocaml-cuda-bench -- "$N" "$REPS" "$DEGREE" "$dtype" | tee /dev/stderr)"
  line="$(printf '%s\n' "$log" | grep '^RESULT' || true)"
  if [ -z "$line" ]; then
    echo "bench-record: no RESULT line from the $dtype run" >&2
    exit 1
  fi
  result[$dtype]="$line"
done

# key=value out of a RESULT line; values never contain a space, and the one
# field that does (card="...") is taken from `info` instead.
field() {
  printf '%s\n' "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p"
}

row() {
  local dtype="$1" line="${result[$1]}"
  printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
    "$dtype" \
    "$(field "$line" n)" \
    "$(field "$line" saxpy_cuda_s)" \
    "$(field "$line" poly_cuda_s)" \
    "$(field "$line" poly_gflops)" \
    "$(field "$line" saxpy_speedup)" \
    "$(field "$line" poly_speedup)"
}

# --- append -----------------------------------------------------------------

mkdir -p bench/results
if [ ! -e "$out" ]; then
  {
    printf '# %s\n\n' "$card"
    printf 'Benchmark results for this card, newest section last. Appended by\n'
    printf '`scripts/bench-record.sh`; see `bench/results/README.md` for what\n'
    printf 'the columns mean.\n'
  } >> "$out"
fi

{
  printf '\n## %s\n\n' "$(date -u '+%Y-%m-%d %H:%M UTC')"
  printf 'commit `%s`, n = %s, reps = %s (best of), poly degree = %s\n\n' \
    "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" "$N" "$REPS" "$DEGREE"
  printf '```\n%s\n```\n\n' "$info"
  printf '| dtype | n | saxpy_cuda_s | poly_cuda_s | poly_gflops | saxpy_speedup | poly_speedup |\n'
  printf '|---|---:|---:|---:|---:|---:|---:|\n'
  row f32
  row f64
} >> "$out"

echo "==> appended a section to $out"
