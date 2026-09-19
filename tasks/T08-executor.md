# T08 — `Differential` + `Executor` (+ wiring `Backend_cuda`)

## Goal

Two pieces:

1. `lib/backend/differential.ml` — run a graph on two backends and compare
   outputs. Pure OCaml; needed by every test from here on.
2. `lib/backend_cuda/executor.ml` — walk a `Kernel_ir.host_op` plan against
   a loaded JIT module using the runtime layer. With this,
   `Backend_cuda.compile`/`run` (already written in
   `ocaml_cuda_backend_cuda.ml`) work end to end.

Depends on: T03, T05, T06, T07. The GPU half of the tests skips without a
device; the Differential half runs anywhere.

## Files you own

- `lib/backend/differential.ml`
- `lib/backend_cuda/executor.ml`

Do not edit `.mli`s, `ocaml_cuda_backend_cuda.ml`, or anything in `runtime/`.

## Interfaces

```ocaml
(* differential.mli *)
val check :
  ?tolerance:float ->
  reference:(module Backend.S) ->
  candidate:(module Backend.S) ->
  Graph.t ->
  inputs:(string * Value.packed) list ->
  (unit, string) result

(* executor.mli *)
val run :
  Kernel_ir.program -> Jit.module_ ->
  inputs:(string * Value.packed) list -> (string * Value.packed) list

(* Backend.S, for reference *)
module type S = sig
  val name : string
  type compiled
  val compile : Graph.t -> compiled
  val run : compiled -> inputs:(string * Value.packed) list -> (string * Value.packed) list
end
```

Runtime you call (T07): `Device.init/synchronize`, `Buffer.alloc/free/upload/download`,
`Jit.get_kernel`, `Launch.run kernel ~grid ~block ~shared_bytes bufs`.

Plan you execute (`Kernel_ir`):
```ocaml
type host_op =
  | Alloc of buffer
  | Upload of { param : string; into : buffer }
  | Launch of { kernel : string; args : buffer list }
  | Download of { from : buffer; output : string; shape : Shape.t }
  | Free of buffer
type kernel = { name; params; shared; body; launch : Schedule.launch }  (* launch = {grid; block; shared_bytes} *)
type buffer = { name : string; dtype : Dtype.packed; memspace; numel : int }
```
`Dtype.size_in_bytes`, `Dtype.equal`, `Dtype.name`, `Value.create/get/numel/dtype/shape`, `Shape.equal/to_string`.

## Implementation — Differential

```ocaml
let check ?(tolerance = 1e-5) ~reference ~candidate graph ~inputs =
  let module R = (val reference : Backend.S) in
  let module C = (val candidate : Backend.S) in
  match R.run (R.compile graph) ~inputs with
  | exception e -> Error ("reference raised: " ^ Printexc.to_string e)
  | expected ->
  match C.run (C.compile graph) ~inputs with
  | exception e -> Error (C.name ^ " raised: " ^ Printexc.to_string e)
  | got -> compare_all ~tolerance expected got
```

`compare_all`: for each `(name, Value.P e)` in `expected`, find `name` in
`got` by **name** (order is irrelevant); missing → `Error "missing output r"`.
Then, with `Value.P g`:
- `Dtype.equal (Value.dtype e) (Value.dtype g)` must be `Some Equal`; use the
  witness to compare typed elements. Else `Error "output r: dtype ..."`.
- shapes `Shape.equal`, else `Error`.
- elementwise, for floats: `abs (g - e) <= tolerance * (1.0 +. abs e)`
  (relative-plus-absolute). Two NaNs count as equal. For ints: exact.
  On the first mismatch return
  `Error (Printf.sprintf "output %s[%d]: expected %g, got %g (tol %g)" ...)`.
- Extra outputs in `got` are ignored.
Return `Ok ()` if everything matches.

Write the elementwise comparison with a `type a.` function over
`a Dtype.t` so each dtype branch compares its own OCaml type.

## Implementation — Executor

```ocaml
let run (program : Kernel_ir.program) module_ ~inputs =
  Device.init ();
  let bufs : (string, Buffer.t) Hashtbl.t = Hashtbl.create 16 in
  let kernels = List.map (fun (k : Kernel_ir.kernel) -> (k.name, k)) program.kernels in
  let outputs = ref [] in
  let synced = ref false in
  List.iter (function
    | Kernel_ir.Alloc b ->
        Hashtbl.replace bufs b.name (Buffer.alloc ~bytes:(b.numel * size_of b.dtype))
    | Upload { param; into } ->
        let v = match List.assoc_opt param inputs with Some v -> v | None -> invalid_arg ("missing input " ^ param) in
        Buffer.upload v (find into)          (* Buffer.upload itself checks byte size *)
    | Launch { kernel; args } ->
        let k = List.assoc kernel kernels in
        Launch.run (Jit.get_kernel module_ kernel) ~grid:k.launch.grid ~block:k.launch.block
          ~shared_bytes:k.launch.shared_bytes (List.map find args);
        synced := false
    | Download { from; output; shape } ->
        if not !synced then (Device.synchronize (); synced := true);
        let v = fresh_value from.dtype shape in        (* Value.P (Value.create d shape) via a `type a.` helper *)
        Buffer.download (find from) v;
        outputs := (output, v) :: !outputs
    | Free b -> Buffer.free (find b.name); Hashtbl.remove bufs b.name)
    program.plan;
  List.rev !outputs
```

`find b = Hashtbl.find bufs b.name` (raise a clear `Failure` if absent —
that would be a T05 bug). `fresh_value (Dtype.P d) shape = Value.P (Value.create d shape)`.

**Exception safety:** wrap the loop in
`Fun.protect ~finally:(fun () -> Hashtbl.iter (fun _ b -> try Buffer.free b with _ -> ()) bufs)`
so a launch failure does not leak device memory. In the `Free` case do
`Buffer.free (find b.name); Hashtbl.remove bufs b.name` — on the success
path the table is empty by the end and `finally` frees nothing, so
nothing is ever freed twice.

Also validate up front that every `Upload` param name exists in `inputs`
**before** allocating anything, so the "missing input" error is raised
without touching the device.

## Invariants

- Output list order = order of `Download` ops = `Graph.outputs` order.
- No device memory leaks on any path.
- `Executor.run` may be called many times on the same compiled program
  with different inputs (S5 system test does 50 runs).
- Differential never raises; every failure is an `Error string`.

## Failure modes to avoid

- Reading results before `synchronize` (kernel launches are async).
- Comparing `Value.packed` values with `=`.
- Absolute-only tolerance: a 4M-element float sum needs relative tolerance.
- Matching outputs positionally instead of by name.
- Allocating the output `Value` with the buffer's `numel` as a 1-D shape:
  use the `shape` carried by `Download`.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_executor.exe
```

Expected without a GPU: `6 tests, 0 failures` plus a SKIP line.
With a GPU: `9 tests, 0 failures`.

## Tests (already written: `test/unit/test_executor.ml`)

CPU: interp agrees with itself on every example; a 0.1 % corruption of
element 1 fails at default tolerance and the message names the output; the
same passes at tolerance 1e-2; reversed output order passes; missing output
is an error; a raising candidate gives `Error` containing `boom`.
GPU: saxpy n=1000 on `Backend_cuda` matches interp; missing input raises;
a compiled `sum` program runs twice with different inputs giving 100 and 200.
