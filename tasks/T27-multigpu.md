# T27 — Multi-GPU: one context per device, data-parallel runs

## Goal

Monte Carlo is embarrassingly parallel: split the paths across devices,
run the same program on each, combine on the host. This task makes the
runtime device-aware (today it hard-codes ordinal 0) and adds a small
data-parallel driver. It is plumbing; it can only be exercised on a
multi-GPU box (brev.dev, T28), so the unit tests use `devices:[0; 0]` to
check the splitting logic on one card and skip the true multi-device
assertions unless `Device.count () >= 2`.

Depends on: T25. Phase 5.

## Files you own

- `lib/runtime/device.ml`, `lib/runtime/device.mli`
- `lib/backend_cuda/ocaml_cuda_backend_cuda.ml`, `lib/backend_cuda/ocaml_cuda_backend_cuda.mli`
- `lib/backend_cuda/multi.ml`, `lib/backend_cuda/multi.mli` (new)
- `test/staged/unit/test_multigpu.ml` → promote

## Interfaces

```ocaml
(* device.mli additions *)
val count : unit -> int
(** Make device [ordinal]'s primary context current on this thread for the
    call. Nested calls restore the previous device. [init ()] is device 0
    and remains the default. *)
val with_device : int -> (unit -> 'a) -> 'a
val current : unit -> int
val info_of : int -> info

(* ocaml_cuda_backend_cuda.mli additions *)
(** Compile for and run on a specific device. [run] on such a [compiled]
    switches to its device for the duration of the call. *)
val compile_on : device:int -> ?streams:int -> Graph.t -> compiled
val device_of : compiled -> int

(* multi.mli *)
open Ocaml_cuda_ir
type t
(** [create ~devices ~n build]: [build ~n:(n / d)] is compiled on each
    device in [devices] (d = List.length devices; [n] must be divisible by
    d). The leading dimension [n] is the split axis. *)
val create : devices:int list -> n:int -> (n:int -> Graph.t) -> t

(** Inputs whose leading dimension is [n] are split into d contiguous
    slices; inputs with any other shape are passed whole to every device.
    Returns one output list per device, in [devices] order. All devices
    are launched before any is waited on. *)
val run : t -> inputs:(string * Value.packed) list -> (string * Value.packed) list list

(** Concatenate per-device outputs along the leading axis (for outputs
    whose leading dimension is n / d). *)
val concat : Value.packed list -> Value.packed

(** Mean of per-device scalar outputs (for Monte Carlo means). *)
val mean_scalar : Value.packed list -> float

val release : t -> unit
```

## Implementation notes

- `Device`: keep a table `ordinal -> (Cuda.Device.t * Cuda.Context.t)`;
  `with_device` pushes the target context (`Cuda.Context.set_current`),
  runs, restores the previous one in a `Fun.protect`.
- `Jit.compile` must run under the target device's context (the module
  belongs to a context). `compile_on` wraps the whole compile in
  `with_device`; `run` wraps the run.
- `Multi.run`: for device i, slice inputs (`Value` sub-copy: there is no
  view type, so copy `n/d` elements into a fresh Value of the sliced
  shape), then `run_async` (T25) under `with_device`; collect all jobs;
  `wait` each (again under its device). Launching all before waiting is
  what gives the parallelism.
- Seeds: the same `seed` on every device would give identical paths. Do
  not solve this in `Multi`: document that a caller who wants independent
  streams passes a per-device input via `run` on each device themselves,
  and provide `Multi.run_per_device : t -> inputs:(int -> (string * Value.packed) list) -> ...`
  where the function receives the device index. `run` is `run_per_device`
  over the sliced inputs.

## Failure modes to avoid

- Calling a driver function on a `Buffer` from another device's context: every
  device-touching call in `Multi` goes through `with_device`.
- `with_device` that does not restore: subsequent single-device code would
  silently run on the wrong card.
- Splitting an input whose leading dim happens to equal `n` but is a
  parameter table, not paths. Document and accept: the split axis rule is
  by shape, and a caller can reshape to avoid it.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_multigpu.exe
```

## Tests (already written: `test/staged/unit/test_multigpu.ml`; GPU, skips without one)

- `Device.count () >= 1`; `with_device 0` returns its value and `current ()` is 0 after
- `Multi.create ~devices:[0; 0] ~n:8192 saxpy`: `run` returns 2 output lists; `concat` of `r` equals the single-device `r`; `mean_scalar` of `s` equals single-device `s / 2` (each half sums half) — the test computes the expected value from the slices
- non-split input: `bs_mc` (T20) with `seed` per device via `run_per_device`; each device's price is within 1.0 of the analytic price and the two differ (different seeds)
- `release` returns `live_count` to the baseline
- only when `count () >= 2`: `compile_on ~device:1` runs saxpy correctly; `Multi.create ~devices:[0; 1]` results match `[0; 0]` results exactly for I32 and within 1e-6 for F32; `info_of 1` has a non-empty name
