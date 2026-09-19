.PHONY: build unit system test clean

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
