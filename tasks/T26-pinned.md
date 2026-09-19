# T26 — Pinned (page-locked) host memory

## Goal

Async copies are only asynchronous, and only run at full PCIe rate, from
page-locked host memory. cudajit 0.7 does not bind `cuMemHostAlloc`, so
bind it here with ctypes (already a transitive dependency through cudajit;
declare it in `lib/runtime/dune`) and give `Value` a way to wrap foreign
memory.

Depends on: T25. Phase 5.

## Files you own

- `lib/runtime/pinned.ml`, `lib/runtime/pinned.mli` (new), `lib/runtime/dune`
- `lib/ir/value.ml`, `lib/ir/value.mli` (you may edit)
- `bench/bench.ml` (use pinned inputs and outputs when `--pinned` is given; print which)
- `test/staged/unit/test_pinned.ml` → promote (`test/unit/dune` already lists `unix` after T25; the test times copies)

New dune libraries for `ocaml_cuda_runtime`: `ctypes ctypes.foreign`.
No new opam packages: both are installed by cudajit.

## Interfaces

```ocaml
(* value.mli addition *)
(** Wrap an existing C-layout Bigarray. [Invalid_argument] if its kind does
    not match [dtype] (F32 → Float32, F64 → Float64, I32 → Int32, I64 → Int64;
    Bool is rejected) or its length is not [Shape.numel shape]. *)
val of_raw : 'a Dtype.t -> Shape.t -> 'a raw -> 'a t

(* pinned.mli *)
(** A host [Value] in page-locked memory obtained from cuMemHostAlloc with
    CU_MEMHOSTALLOC_PORTABLE. Freed by a finaliser through cuMemFreeHost;
    the Bigarray keeps the allocation alive. *)
val alloc : 'a Dtype.t -> Shape.t -> 'a Value.t

(** Copy of an ordinary Value into pinned memory. *)
val of_value : 'a Value.t -> 'a Value.t

(** True if the Value was produced by [alloc]/[of_value] (tracked by
    physical address in a weak table). *)
val is_pinned : _ Value.t -> bool
```

## Implementation notes

- `Foreign.foreign ~from:(Dl.dlopen ~filename:"libcuda.so.1" ~flags:[Dl.RTLD_NOW])
  "cuMemHostAlloc" (ptr (ptr void) @-> size_t @-> uint @-> returning int)`;
  `cuMemFreeHost (ptr void @-> returning int)`. Check the return code is 0
  and raise `Failure` with the code otherwise. `Device.init ()` first.
- `Ctypes.bigarray_of_ptr Ctypes.array1 n kind (from_voidp typ p)` gives
  the Array1; wrap with `Value.of_raw`. Attach the finaliser to the
  Bigarray (`Gc.finalise`), not to the `Value`.
- Kind matching in `of_raw`: match on `Bigarray.Array1.kind a` against
  `Bigarray.Float32` etc. together with the dtype witness.
- Windows: `libcuda.so.1` does not exist; this module is built only for a
  Linux/WSL toolchain, as is everything under `runtime`. Fail at call
  time, not at load time, when the library is missing (lazy `dlopen`).

## Failure modes to avoid

- Freeing pinned memory while a copy is in flight: the job closure of T25
  captures the Value, so the finaliser cannot run before `wait`. Do not
  weaken that.
- `bigarray_of_ptr` with a length in bytes instead of elements.
- Letting `of_raw` accept a `Float64` Bigarray for `F32` because both have
  element type `float`: that is exactly the mismatch the kind check exists for.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_pinned.exe
```

## Tests (already written: `test/staged/unit/test_pinned.ml`; GPU, skips without one)

- `Pinned.alloc F32 [1000]` is writable, readable, `is_pinned`, numel 1000; an ordinary `Value.create` is not pinned
- `of_raw` rejects a Float64 Bigarray for `F32` and a wrong length; accepts a matching one and shares storage (write through one, read through the other)
- pinned input → upload → download into pinned output round-trips exactly through `Backend_cuda.run` on saxpy
- `run_async` (T25) with pinned inputs and outputs gives the same sums as sync
- informational: H→D bandwidth pinned vs pageable at 64 MiB, printed in GB/s; no assertion
- allocate and drop 50 pinned 1 MiB values, `Gc.full_major`, then allocate again: no failure (finalisers free)
