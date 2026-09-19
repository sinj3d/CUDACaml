# T07 — `Runtime`: device, buffers, NVRTC JIT, launch (over cudajit)

## Goal

Implement the four modules in `lib/runtime/` on top of the `cudajit` opam
package (OCaml bindings to the CUDA **driver** API and NVRTC). This is the
only layer that touches the GPU and the only one that knows nothing about
the IR. It must build and load on a machine with no GPU; every function
except `Device.available` may raise there.

**Do this task on the machine with the NVIDIA GPU.** It de-risks the whole
project: if a hand-written kernel string compiles and launches from OCaml,
everything above is pure OCaml.

Depends on: T01 (`Value.raw`). Independent of T02–T06.

## Files you own

- `lib/runtime/device.ml`, `buffer.ml`, `jit.ml`, `launch.ml`
- `lib/runtime/dune` — add `cudajit` to `(libraries ...)`
- `dune-project` — uncomment the `cudajit` dependency line

Do not edit the `.mli`s.

## Setup

```
opam install cudajit          # needs CUDA toolkit + driver installed; libcuda.so / nvcuda.dll on the path
ls $(opam var lib)/cudajit/   # read cuda.mli and nvrtc.mli — they are the authoritative API
```

The API below was checked against the cudajit README; **argument labels
may differ slightly by version. Trust the installed `.mli` over this
document** and note any deviation in your report.

```ocaml
module Cu = Cudajit.Cuda   (* or `Cudajit.Cu`; check *)
module Nvrtc = Cudajit.Nvrtc

Cu.init ()
let dev = Cu.Device.get ~ordinal:0
let ctx = Cu.Context.create ~flags:0 dev          (* keep it alive for the process *)
Cu.Context.synchronize ()

let prog = Nvrtc.compile_to_ptx ~cu_src:source ~name ~options:[] ~with_debug:false
(* prog carries the PTX text and the compile log; on failure it raises with the log *)
let module_ = Cu.Module.load_data_ex prog []
let func = Cu.Module.get_function module_ ~name

let dptr = Cu.Deviceptr.mem_alloc ~size_in_bytes
Cu.Deviceptr.memcpy_H_to_D ~dst:dptr ~src:(bigarray) ()
Cu.Deviceptr.memcpy_D_to_H ~dst:(bigarray) ~src:dptr ()
Cu.Deviceptr.mem_free dptr

Cu.Stream.launch_kernel func ~grid_dim_x ~block_dim_x ~shared_mem_bytes Cu.Stream.no_stream
  [ Cu.Stream.Tensor dptr1; Cu.Stream.Tensor dptr2; ... ]
```

`memcpy_*` take a `Bigarray.Genarray.t`; convert an `Array1` with
`Bigarray.genarray_of_array1`.

## Interfaces (verbatim `.mli`s)

```ocaml
(* device.mli *)
val available : unit -> bool
val init : unit -> unit
val synchronize : unit -> unit
val name : unit -> string

(* buffer.mli *)
type t
val alloc : bytes:int -> t
val free : t -> unit
val byte_size : t -> int
val upload : Value.packed -> t -> unit
val download : t -> Value.packed -> unit

(* jit.mli *)
type module_
type kernel
val compile : name:string -> source:string -> module_
val ptx : module_ -> string
val get_kernel : module_ -> string -> kernel
val unload : module_ -> unit

(* launch.mli *)
val run : Jit.kernel -> grid:int -> block:int -> shared_bytes:int -> Buffer.t list -> unit
```

`Value.raw : 'a Value.t -> 'a raw` with
`type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw`;
`Value.byte_size : _ Value.t -> int`.

## Implementation

**Device.** Module-level `let state = ref None` holding the context.
- `init ()`: if `!state = None` then `Cu.init (); get device 0; create
  context; store`. Idempotent. Raises on failure.
- `available ()`: `match init () with () -> true | exception _ -> false`,
  **memoised** in a `bool option ref` so it is probed once.
- `synchronize ()`: `init (); Cu.Context.synchronize ()`.
- `name ()`: `init ()`; device name via `Cu.Device.get_name` or the
  attributes API (check `.mli`); fall back to `"cuda device 0"` if no
  such call exists.

**Buffer.** `type t = { ptr : Cu.Deviceptr.t; bytes : int }`.
- `alloc ~bytes`: `Device.init ()`; `mem_alloc ~size_in_bytes:(max 1 bytes)`
  (CUDA rejects 0-byte allocations; a 0-element tensor still needs a valid
  pointer).
- `upload (Value.P v) t`: `if Value.byte_size v <> t.bytes then invalid_arg`.
  Then `match Value.raw v with Value.Raw a -> memcpy_H_to_D ~dst:t.ptr ~src:(genarray_of_array1 a) ()`.
  Skip the memcpy when bytes = 0.
- `download t (Value.P v)`: symmetric.
- `free t`: `mem_free t.ptr`.

**Jit.** `type module_ = { m : Cu.Module.t; ptx : string }`, `type kernel = Cu.Module.func` (whatever `get_function` returns).
- `compile ~name ~source`: `Device.init ()`; `Nvrtc.compile_to_ptx ~cu_src:source ~name:(name ^ ".cu") ~options:["--gpu-architecture=compute_XX"?]`
  — start with `~options:[]`; only add an arch flag if NVRTC complains.
  On exception re-raise as `Failure ("nvrtc: " ^ log)` so the compiler
  log is visible. Then `load_data_ex`.
- `ptx m = m.ptx` (obtain the PTX string from the compile result; the
  README shows it is a field/accessor on the compiled program).
- `get_kernel m name`: `get_function` — raises if the name is missing;
  keep that behaviour (test expects an exception for `"nope"`).
- `unload`: `Cu.Module.unload` if available, else no-op.

**Launch.** `run k ~grid ~block ~shared_bytes bufs`:
`launch_kernel k ~grid_dim_x:grid ~block_dim_x:block ~shared_mem_bytes:shared_bytes no_stream (List.map (fun b -> Tensor (Buffer.unsafe_ptr b)) bufs)`.

`launch.ml` needs the device pointer hidden inside `Buffer.t`. Add exactly
one line to `lib/runtime/buffer.mli`:

```ocaml
val unsafe_ptr : t -> Cudajit.Cuda.Deviceptr.t   (* adjust the module path to the installed cudajit *)
```

This is the only `.mli` edit this task is allowed. Add `buffer.mli` to
your "files changed" report.

## Invariants

- `Device.available ()` never raises and never prints.
- All copies are synchronous. `Launch.run` is asynchronous (returns before
  the kernel finishes); callers call `Device.synchronize ()` before reading
  results. Do not synchronize inside `run`.
- One context for the process lifetime. Never create a context per call.

## Failure modes to avoid

- Passing an OCaml `float array` or `Bytes` to memcpy — only Bigarrays
  are GC-stable; that is why `Value` exists.
- 0-byte `mem_alloc` (fails with `CUDA_ERROR_INVALID_VALUE`).
- Forgetting `extern "C"` in the test kernel is not your problem (the
  test source has it), but `get_function` failing with a mangled name is
  the symptom if someone drops it.
- Swallowing NVRTC's log; the first real bug in T06's output will be
  undiagnosable without it.
- Windows: cudajit may not build. This task is expected to run on Linux.

## Verify

On the GPU machine:

```
dune build 2>&1 && dune exec test/unit/test_runtime.exe
```

Expected: `8 tests, 0 failures` and the `.entry saxpy` PTX check passing.
On a machine with no device the same command must print `SKIP no CUDA device`
and `0 tests, 0 failures` (exit 0) — check this too if you can.

## Tests (already written: `test/unit/test_runtime.ml`)

init idempotent; device name non-empty; alloc/free/byte_size; I32
upload→download roundtrip; mismatched-size upload raises; compile yields
PTX containing `.entry saxpy` and `get_kernel` finds it / raises on a bad
name; compile error raises; **hand-written saxpy kernel end to end on
1000 elements with grid 4 × block 256 equals the expected floats**.
