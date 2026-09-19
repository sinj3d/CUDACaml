.PHONY: build unit system test bench record clean env

# ---------------------------------------------------------------------------
# CUDA runtime discovery.
#
# lib/runtime links against cudajit, which links libnvrtc. libcuda.so.1 is
# resolved by the system loader (on WSL it ships in /usr/lib/wsl/lib, which is
# already registered), but libnvrtc lives with the toolkit, so a toolkit that
# is not in the default loader path has to be added here or every GPU test
# dies with "libnvrtc.so.12: cannot open shared object file".
#
# CUDA_PATH is honoured if set; otherwise we look for a local, no-sudo install.
# All of this is a no-op when the toolkit is installed system-wide, and a no-op
# when there is no toolkit at all (the GPU tests then SKIP themselves).
# ---------------------------------------------------------------------------
CUDA_PATH ?= $(firstword $(wildcard $(HOME)/cuda-12.9 /usr/local/cuda))

ifneq ($(CUDA_PATH),)
CUDA_LIB := $(firstword $(wildcard $(CUDA_PATH)/lib64 $(CUDA_PATH)/lib))
export LD_LIBRARY_PATH := $(CUDA_LIB):$(LD_LIBRARY_PATH)
export LIBRARY_PATH    := $(CUDA_LIB):$(LIBRARY_PATH)
export CPATH           := $(CUDA_PATH)/include:$(CPATH)
export CUDA_PATH
endif

# ---------------------------------------------------------------------------
# Optional user prefix.
#
# ctypes-foreign (via cudajit) links -lffi. Normally libffi-dev supplies it
# system-wide; where it cannot be installed as root, staging it under a user
# prefix and pointing LOCAL_PREFIX here keeps the link working. Purely a
# no-op when the directory does not exist.
# ---------------------------------------------------------------------------
LOCAL_PREFIX ?= $(firstword $(wildcard $(HOME)/local))

ifneq ($(LOCAL_PREFIX),)
export LD_LIBRARY_PATH := $(LOCAL_PREFIX)/lib:$(LD_LIBRARY_PATH)
export LIBRARY_PATH    := $(LOCAL_PREFIX)/lib:$(LIBRARY_PATH)
export CPATH           := $(LOCAL_PREFIX)/include:$(CPATH)
export PKG_CONFIG_PATH := $(LOCAL_PREFIX)/lib/pkgconfig:$(PKG_CONFIG_PATH)
endif

# Print what the build will use. Handy when a GPU test skips unexpectedly,
# or when the link fails looking for -lffi or -lnvrtc.
env:
	@echo "CUDA_PATH       = $(CUDA_PATH)"
	@echo "CUDA_LIB        = $(CUDA_LIB)"
	@echo "LOCAL_PREFIX    = $(LOCAL_PREFIX)"
	@echo "LD_LIBRARY_PATH = $(LD_LIBRARY_PATH)"
	@echo "LIBRARY_PATH    = $(LIBRARY_PATH)"

build:
	dune build 2>&1

# Unit tests: one executable per task, no GPU needed except test_runtime /
# test_executor, which SKIP themselves when no device is present.
unit: build
	dune test test/unit

# System tests run ONLY if `unit` succeeded (make stops on the first failing
# prerequisite) and only with the env var set.
system: unit
	OCAML_CUDA_SYSTEM=1 dune test test/system --force

# Hand-written OCaml against the CUDA backend on the same two workloads.
# Needs a device; exits 77 without one. Override the shape with e.g.
#   make bench BENCH_ARGS="1048576 5 256 f64"  (n, reps, poly degree, dtype)
# The dtype is f32 or f64 and defaults to f32.
BENCH_ARGS ?=
bench: build
	dune exec ocaml-cuda-bench -- $(BENCH_ARGS)

# Both precisions, then a dated section appended to bench/results/<card>.md.
# BENCH_ARGS here is "n reps degree" only: the dtype is what it varies.
record: build
	bash scripts/bench-record.sh $(BENCH_ARGS)

test: system

clean:
	dune clean
