# T25 — Streams, events, async copies, and device-resident values

## Goal

Two things a nightly batch needs and a demo does not:

1. **Overlap.** Independent runs of one compiled program on a small ring of
   streams, each with its own buffer set, so the upload of job k+1 overlaps
   the kernels of job k. Exposed as `run_async` / `wait`.
2. **Resident values.** Tensors that stay on the device between runs, so a
   driver loop like Longstaff–Schwartz (T23) does not round-trip its path
   matrix through the host every timestep.

cudajit 0.7 already exposes `Cuda.Stream` (create, `memcpy_H_to_D_async`,
`memcpy_D_to_H_async`, `synchronize`, `launch_kernel` on a stream) and
`Cuda.Event` (create, record, wait, synchronize, `elapsed_time`). This
task wraps them in the runtime layer and uses them in the backend.

Depends on: T24. Phase 5.

## Files you own

- `lib/runtime/stream.ml`, `lib/runtime/stream.mli`, `lib/runtime/event.ml`, `lib/runtime/event.mli` (new)
- `lib/runtime/buffer.ml`, `lib/runtime/buffer.mli`, `lib/runtime/launch.ml`, `lib/runtime/launch.mli` (you may edit)
- `lib/backend_cuda/executor.ml`, `lib/backend_cuda/executor.mli`
- `lib/backend_cuda/ocaml_cuda_backend_cuda.ml`, `lib/backend_cuda/ocaml_cuda_backend_cuda.mli`
- `examples/lsm.ml` — add `Make_cuda_resident` (see below) without changing `Make`
- `test/unit/dune` — add `unix` to `libraries` (the staged test times runs with `Unix.gettimeofday`)
- `test/staged/unit/test_streams.ml` → promote

## Interfaces

```ocaml
(* stream.mli *)
type t
val default : t                       (* the NULL stream *)
val create : unit -> t
val synchronize : t -> unit

(* event.mli *)
type t
val record : Stream.t -> t            (* creates a timing-enabled event and records it *)
val wait : Stream.t -> t -> unit      (* the stream waits for the event *)
val synchronize : t -> unit
val elapsed_ms : start:t -> stop:t -> float

(* buffer.mli additions *)
val upload_async : Value.packed -> t -> stream:Stream.t -> unit
val download_async : t -> Value.packed -> stream:Stream.t -> unit
(* The host Value must stay reachable until the stream is synchronised;
   callers hold it. *)

(* launch.mli *)
val run : ?stream:Stream.t -> Jit.kernel -> grid:int -> ?grid_y:int -> block:int -> ?block_y:int -> shared_bytes:int -> Buffer.t list -> unit

(* executor.mli additions *)
val create_on : stream:Stream.t -> Kernel_ir.program -> Jit.module_ -> t
val stream : t -> Stream.t
(** Like [run] but returns before the downloads complete; [finish] waits
    and returns the outputs. The returned closure owns the host Values. *)
val run_async : t -> inputs:(string * Value.packed) list -> (unit -> (string * Value.packed) list)

(** Resident: run with device inputs and device outputs. Inputs are
    [Buffer.t]s of the right byte size (checked); outputs are freshly
    allocated buffers the caller owns. No host traffic. *)
val run_resident : t -> inputs:(string * Buffer.t) list -> (string * Buffer.t) list

(* ocaml_cuda_backend_cuda.mli additions *)
type resident = { buf : Runtime.Buffer.t; dtype : Dtype.packed; shape : Shape.t }
val upload : Value.packed -> resident
val download : resident -> Value.packed
val free_resident : resident -> unit
val run_resident : compiled -> inputs:(string * resident) list -> (string * resident) list
(** Output residents carry the dtype and shape from the plan's [Download] ops. *)

(** [streams] executors on [streams] streams, jobs assigned round-robin. *)
val compile_with : streams:int -> Graph.t -> compiled
type job
val run_async : compiled -> inputs:(string * Value.packed) list -> job
val wait : job -> (string * Value.packed) list
```

`compile` = `compile_with ~streams:1`. `run` on a multi-stream `compiled`
uses executor 0 and synchronises, so `Backend.S` semantics are unchanged.

## Implementation notes

- `run_async` on the executor: `upload_async` each input on its stream,
  launch on its stream, allocate fresh output Values, `download_async`
  into them, record an event, return a closure that synchronises the event
  and returns the outputs. The closure captures the input Values too, so
  they outlive the copies.
- A job on executor `i` must not start until the previous job on executor
  `i` finished (the buffers are shared). Streams are FIFO, so issuing in
  order is enough; but the *host Values* of the previous job's outputs are
  what its `wait` returns, and they are already downloaded into distinct
  Values. No extra sync needed.
- Without pinned host memory (T26), the driver may make async copies
  synchronous with respect to the host. Overlap of copy and compute then
  comes only from the launch being async. The API is still correct; T26
  makes it fast, and the test here is a correctness test with an
  informational timing.
- `run_resident`: bind the plan's `Upload` buffers to the given inputs by
  *copying* device-to-device (`Cuda.Deviceptr.memcpy_D_to_D`) into the
  executor's own param buffers, or, better, substitute pointers: launch
  with the caller's buffer in place of the param buffer. Substitution is
  zero-copy and is the intended design; the plan's `Launch` args are
  looked up by name, so a per-run override table on top of the executor's
  table is a few lines.
- `Make_cuda_resident` in `examples/lsm.ml`: same algorithm as `Make`,
  specialised to `Backend_cuda`, keeping `S` and `cf` resident across the
  time loop and downloading only `xtx`/`xty` (18 doubles) per step.

## Failure modes to avoid

- Returning outputs from `run_async` before the event: the classic
  silent-garbage bug. `wait` must block.
- Letting a host `Value` be collected during an in-flight copy. Capture it.
- Creating a stream per job: streams are created at `compile_with` and reused.
- Substituting a resident buffer whose byte size differs from the param's: check and raise.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_streams.exe
```

## Tests (already written: `test/staged/unit/test_streams.ml`; GPU, skips without one)

- `Stream.create`, `Event.record`/`synchronize`/`elapsed_ms` ≥ 0 on the default stream
- `compile_with ~streams:3` saxpy; 8 `run_async` jobs with different inputs; `wait` each; every sum equals the synchronous `run` result
- `wait` twice on one job returns equal values
- resident: `upload` then `download` round-trips a `[1000]` F32 value exactly
- `run_resident` of saxpy with resident `x`,`y` returns residents whose `download` equals the host run; output shape and dtype are right
- chaining: output of `run_resident (chain)` fed as input to `run_resident (sum)` equals the interpreter's composition
- `live_count` after freeing all residents and releasing equals the starting value
- informational: time 16 sync runs vs 16 async runs on 4 streams at n = 2^22 and print both
- `Lsm.Make_cuda_resident` price agrees with `Lsm.Make (Backend_cuda)` on the same seed to 1e-9 (n_paths 4096, n_steps 20)
