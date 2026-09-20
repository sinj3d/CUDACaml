open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime
module Emit = Emit
module Mangle = Mangle
module Executor = Executor

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
}

let source graph = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program |> Emit.program

let compile_with ~streams graph =
  if streams < 1 then invalid_arg "Backend_cuda.compile_with: streams must be >= 1";
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
  { program; execs; next = 0 }

let compile graph = compile_with ~streams:1 graph

(* [Backend.S.run] is synchronous by contract, so it uses executor 0 and
   waits. On a multi-stream [compiled] that is one lane of the ring;
   jobs already queued on lane 0 are ahead of it in the stream, so it
   neither races them nor sees their buffers half written. *)
let run c ~inputs = Executor.run c.execs.(0) ~inputs
let release c = Array.iter Executor.release c.execs
let executor c = c.execs.(0)

type job = unit -> (string * Value.packed) list

let run_async c ~inputs =
  let i = c.next in
  c.next <- (c.next + 1) mod Array.length c.execs;
  Executor.run_async c.execs.(i) ~inputs

let wait (j : job) = j ()

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
    Executor.run_resident c.execs.(0)
      ~inputs:(List.map (fun (n, r) -> (n, r.buf)) inputs)
  in
  List.map
    (fun (output, buf) ->
      match List.assoc_opt output specs with
      | Some (dtype, shape) -> (output, { buf; dtype; shape })
      | None -> failwith ("Backend_cuda.run_resident: no plan entry for output " ^ output))
    bufs
