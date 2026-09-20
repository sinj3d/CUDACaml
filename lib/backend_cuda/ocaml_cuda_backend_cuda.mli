(** The CUDA backend: [Pipeline] -> [Lower] -> [Emit] -> [Jit], then
    [Executor] per run. *)

open Ocaml_cuda_ir

include Ocaml_cuda_backend.Backend.S

(** The generated CUDA C++ for a graph, without compiling it. What the CLI
    prints and what golden tests snapshot. *)
val source : Graph.t -> string

(** Frees the device buffers a [compile] allocated. Idempotent; [run]
    after it raises. A dropped [compiled] is released by a finaliser, but
    do not rely on that. *)
val release : compiled -> unit

(** The persistent executor behind a [compiled]. For tests and tooling.
    On a multi-stream [compiled] this is lane 0, the one [run] uses. *)
val executor : compiled -> Executor.t

(** {1 Overlap}

    [compile_with ~streams:n] JITs once and builds [n] executors, one per
    stream, each with its own buffer pool. Jobs go round-robin, so the
    uploads of job [k+1] can overlap the kernels of job [k].
    [compile = compile_with ~streams:1], and [run] always uses lane 0 and
    waits, so {!Ocaml_cuda_backend.Backend.S} semantics are unchanged. *)
val compile_with : streams:int -> Graph.t -> compiled

(** {1 Device selection}

    A CUDA module belongs to a context and a context belongs to a device,
    so a [compiled] belongs to one device for its whole life. Compile for
    and run on a specific device: [compile_on] does all of the JIT and the
    pool allocation under device [device], and [run], [run_async]/[wait],
    [run_resident] and [release] switch to it for the duration of the
    call and switch back. [compile] is [compile_on ~device:0], which is
    what the runtime has always done. *)
val compile_on : device:int -> ?streams:int -> Graph.t -> compiled

val device_of : compiled -> int

type job

(** Issues a run and returns immediately. The job owns its input host
    [Value]s, so they outlive the copies that read them. *)
val run_async : compiled -> inputs:(string * Value.packed) list -> job

(** Blocks on the event recorded after the job's downloads, then returns
    its outputs. Waiting twice is harmless and yields the same values. *)
val wait : job -> (string * Value.packed) list

(** {1 Device-resident values}

    A [resident] is a device buffer plus the dtype and shape needed to
    interpret it. It survives across runs, so a driver loop can keep a
    big intermediate on the device instead of round-tripping it. The
    caller owns every [resident] it is handed and must [free_resident]
    it. *)

type resident = {
  buf : Ocaml_cuda_runtime.Buffer.t;
  dtype : Dtype.packed;
  shape : Shape.t;
}

val upload : Value.packed -> resident
val download : resident -> Value.packed
val free_resident : resident -> unit

(** Device in, device out: no host traffic. Inputs bind to the graph's
    params by name and must match their byte size. Output residents are
    freshly allocated and carry the dtype and shape from the plan's
    [Download] ops. Uses lane 0 and synchronises before returning. *)
val run_resident : compiled -> inputs:(string * resident) list -> (string * resident) list

(** Internals, exposed for tests and tooling. *)
module Emit = Emit

module Mangle = Mangle
module Executor = Executor

(** Data-parallel runs across several devices. *)
module Multi = Multi
