open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime
module Emit = Emit
module Mangle = Mangle
module Executor = Executor
module Multi = Multi

let name = "cuda"

(* A compiled program owns its device buffers (see [Executor]); [release]
   gives them back. With [compile_with ~streams:n] it owns [n] of them,
   one per stream, each with its own buffer pool, and hands jobs out
   round-robin. The [Jit.module_] is shared: the code is identical, only
   the memory and the ordering differ. *)
type compiled = {
  program : Kernel_ir.program;
  execs : Executor.t array;
  mutable next : int;
  device : int;  (** the ordinal every buffer, module and stream below belongs to *)
}

let source graph = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program |> Emit.program

(* A CUDA module is loaded into one context and an executor's buffers and
   streams belong to one context, so the whole build happens under the
   target device and so does every later use of it. Device 0 is the
   default, which is exactly what v1 did: it was the only device the
   runtime could name. *)
let compile_on ~device ?(streams = 1) graph =
  if streams < 1 then invalid_arg "Backend_cuda.compile_on: streams must be >= 1";
  Device.with_device device (fun () ->
      let program = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program in
      let module_ = Jit.compile ~name:program.name ~source:(Emit.program program) in
      (* Executor 0 stays on the NULL stream so a single-stream [compiled] is
         bit-for-bit the v1 arrangement; the rest get non-blocking streams,
         which is what keeps the NULL stream from serialising them. *)
      let execs =
        Array.init streams (fun i ->
            let stream = if i = 0 then Stream.default else Stream.create () in
            Executor.create_on ~stream program module_)
      in
      { program; execs; next = 0; device })

let compile_with ~streams graph = compile_on ~device:0 ~streams graph
let compile graph = compile_on ~device:0 graph
let device_of c = c.device

(* [Backend.S.run] is synchronous by contract, so it uses executor 0 and
   waits. On a multi-stream [compiled] that is one lane of the ring;
   jobs already queued on lane 0 are ahead of it in the stream, so it
   neither races them nor sees their buffers half written. *)
let run c ~inputs = Device.with_device c.device (fun () -> Executor.run c.execs.(0) ~inputs)
let release c = Device.with_device c.device (fun () -> Array.iter Executor.release c.execs)
let executor c = c.execs.(0)

(* A job carries its device: the wait does a [cuEventSynchronize] and a
   stream query, both of which need the context that recorded the event. *)
type job = { job_device : int; finish : unit -> (string * Value.packed) list }

let run_async c ~inputs =
  let i = c.next in
  c.next <- (c.next + 1) mod Array.length c.execs;
  {
    job_device = c.device;
    finish = Device.with_device c.device (fun () -> Executor.run_async c.execs.(i) ~inputs);
  }

let wait (j : job) = Device.with_device j.job_device j.finish

(* ------------------------------------------------------------------ *)
(* Device-resident values                                              *)
(* ------------------------------------------------------------------ *)

type resident = { buf : Buffer.t; dtype : Dtype.packed; shape : Shape.t }

let upload (Value.P v as p) =
  Device.init ();
  let buf = Buffer.alloc ~bytes:(Value.byte_size v) in
  Buffer.upload p buf;
  { buf; dtype = Dtype.P (Value.dtype v); shape = Value.shape v }

let download r =
  let v = match r.dtype with Dtype.P d -> Value.P (Value.create d r.shape) in
  Buffer.download r.buf v;
  v

let free_resident r = Buffer.free r.buf

(* The executor speaks in buffers only; the dtype and shape of an output
   live in the plan's [Download] op, which is where the host [Value]s of
   an ordinary [run] get theirs too. *)
let output_specs (program : Kernel_ir.program) =
  List.filter_map
    (function
      | Kernel_ir.Download { from; output; shape } -> Some (output, (from.dtype, shape))
      | Kernel_ir.Alloc _ | Kernel_ir.Upload _ | Kernel_ir.Launch _ | Kernel_ir.Free _ -> None)
    program.plan

let run_resident c ~inputs =
  let specs = output_specs c.program in
  let bufs =
    Device.with_device c.device (fun () ->
        Executor.run_resident c.execs.(0) ~inputs:(List.map (fun (n, r) -> (n, r.buf)) inputs))
  in
  List.map
    (fun (output, buf) ->
      match List.assoc_opt output specs with
      | Some (dtype, shape) -> (output, { buf; dtype; shape })
      | None -> failwith ("Backend_cuda.run_resident: no plan entry for output " ^ output))
    bufs
