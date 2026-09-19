# T11 — `Device.info`, `ocaml-cuda info`, and the f64 correction

## Goal

Two small things that every later benchmark and README table depends on:

1. A device-description record in the runtime layer and an `info`
   subcommand that prints it, so a results table is self-describing.
2. Correct the documentation. `F64` already works end to end (an f64 graph
   emits `double` and passes `Differential.check`); the README and
   ARCHITECTURE.md say otherwise. The real f64 caveat is hardware
   throughput, and it belongs in the docs as a hardware fact.

Depends on: nothing in v2. Phase 0.

## Files you own

- `lib/runtime/device.ml`, `lib/runtime/device.mli` (you may edit this `.mli`)
- `bin/main.ml`
- `README.md`, `ARCHITECTURE.md`
- `test/staged/unit/test_info.ml` → promote to `test/unit/` (see tasks/README.md)

## Interfaces

Add to `lib/runtime/device.mli`:

```ocaml
type info = {
  name : string;
  compute_capability : int * int;   (** (major, minor); sm_120 is (12, 0) *)
  multiprocessors : int;
  total_memory_bytes : int;
}

(** Device 0. Raises when no device is present; call [available] first. *)
val info : unit -> info

(** One line per field, in this order and with these exact keys, so a
    shell script can grep it:
      device: <name>
      compute: sm_<major><minor>
      multiprocessors: <n>
      memory_mib: <total / 1048576>
    The [sm_] spelling concatenates major and minor without a separator,
    which is how nvcc names architectures ([sm_80], [sm_120]). *)
val info_to_string : info -> string
```

cudajit gives you `(Cuda.Device.get_attributes dev)` with fields `name`,
`compute_capability_major`, `compute_capability_minor`,
`multiprocessor_count`, and `Cuda.Device.get_free_and_total_mem ()` for
memory (the second component is total). cudajit 0.7 exposes **no** driver
or NVRTC version query; do not invent one.

`bin/main.ml`: add `info` to the command match and to `usage`:

```
ocaml-cuda info      print the device description (exit 3 with a message when no device)
```

## Documentation changes

`README.md`:

- Remove "anything beyond `F32` in the fast path" from the out-of-scope
  sentence. Replace with a short "Precision" paragraph in the Status
  section stating: `F32` and `F64` go through the same pipeline and both
  are differential-tested; `I32`/`I64` are supported for element-wise and
  reduction ops; fp64 *throughput* is a property of the card, roughly 1/64
  of fp32 on GeForce parts and about 1/2 on A100/H100-class parts. Say
  that the dtype is chosen per tensor with `Dtype.F64`.
- Add `info` to the CLI table.

`ARCHITECTURE.md`: same removal from "Deliberately out of scope (v1)".
Add one line under "Invariants worth defending": every dtype-dependent
decision is made from the `Dtype.t` witness in `Lower`/`Emit`/`Interp`; no
dtype is special-cased as "the fast one".

Do not touch the measured numbers in the Status section.

## Failure modes to avoid

- Printing the info in `Device.name ()`: `name` is used by tests that
  compare exact strings. Add, do not change.
- Making `info ()` call `Cuda.init` without `init ()`: go through the
  existing `init`.
- A `memory_mib` computed with `/ 1_000_000`.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_info.exe
grep -c "beyond \`F32\`" README.md ARCHITECTURE.md     # both 0
dune exec ocaml-cuda -- info                          # on a GPU machine
```

## Tests (already written: `test/staged/unit/test_info.ml`)

- without a device: `info` raises and the test skips
- `compute_capability` major ≥ 5, name non-empty, multiprocessors ≥ 1,
  memory ≥ 1 GiB
- `info_to_string` has exactly the four keys in order and `sm_` is
  major concatenated with minor
- `Device.name ()` is unchanged and equals `(info ()).name`
