.PHONY: build unit system test clean env

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

# Print what the build will use. Handy when a GPU test skips unexpectedly.
env:
	@echo "CUDA_PATH       = $(CUDA_PATH)"
	@echo "CUDA_LIB        = $(CUDA_LIB)"
	@echo "LD_LIBRARY_PATH = $(LD_LIBRARY_PATH)"

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

test: system

clean:
	dune clean
