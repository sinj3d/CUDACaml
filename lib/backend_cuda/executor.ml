open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime

let buffer_bytes (b : Kernel_ir.buffer) =
  match b.dtype with Dtype.P d -> b.numel * Dtype.size_in_bytes d

(* [Download] carries the real shape; the buffer only knows its numel. *)
let fresh_value dtype shape =
  match dtype with Dtype.P d -> Value.P (Value.create d shape)

(* Every [Upload] param must be present before a single byte of device
   memory is touched, so "missing input" costs nothing to report. *)
let check_inputs (plan : Kernel_ir.host_op list) ~inputs =
  List.iter
    (function
      | Kernel_ir.Upload { param; into = _ } ->
          if not (List.mem_assoc param inputs) then
            invalid_arg ("Executor.run: missing input " ^ param)
      | Kernel_ir.Alloc _ | Kernel_ir.Launch _ | Kernel_ir.Download _ | Kernel_ir.Free _ -> ())
    plan

(* A compiled program owns its device memory for its whole lifetime: the
   plan's [Alloc]s happen once in [create], the plan's [Free]s happen once
   in [release], and [run] does nothing but move data and launch.

   It also owns a stream. Every operation of every run is issued on that
   stream, so two executors of the same program can be driven
   concurrently and one executor's successive jobs stay ordered. *)
type t = {
  program : Kernel_ir.program;
  module_ : Jit.module_;
  stream : Stream.t;
  bufs : (string, Buffer.t) Hashtbl.t;
  kernels : (string * Kernel_ir.kernel) list;
  mutable released : bool;
}

(* The plan is validated, not trusted: a buffer allocated but never freed
   would leak for the executor's lifetime, and a free with no alloc would
   mean [release] frees a pointer it does not own. Validation happens
   before any allocation, so a bad plan costs no device memory. *)
let validate (plan : Kernel_ir.host_op list) =
  let allocated = Hashtbl.create 16 and freed = Hashtbl.create 16 in
  List.iter
    (function
      | Kernel_ir.Alloc b ->
          if Hashtbl.mem allocated b.name then
            failwith ("Executor.create: buffer " ^ b.name ^ " is allocated twice");
          Hashtbl.replace allocated b.name b
      | Kernel_ir.Free b ->
          if not (Hashtbl.mem allocated b.name) then
            failwith ("Executor.create: buffer " ^ b.name ^ " is freed but never allocated");
          if Hashtbl.mem freed b.name then
            failwith ("Executor.create: buffer " ^ b.name ^ " is freed twice");
          Hashtbl.replace freed b.name ()
      | Kernel_ir.Upload _ | Kernel_ir.Launch _ | Kernel_ir.Download _ -> ())
    plan;
  Hashtbl.iter
    (fun name _ ->
      if not (Hashtbl.mem freed name) then
        failwith ("Executor.create: buffer " ^ name ^ " is allocated but never freed"))
    allocated

let release t =
  if not t.released then begin
    t.released <- true;
    Hashtbl.iter (fun _ b -> Buffer.free b) t.bufs;
    Hashtbl.reset t.bufs
  end

let create_on ~stream (program : Kernel_ir.program) module_ =
  validate program.plan;
  Device.init ();
  let bufs : (string, Buffer.t) Hashtbl.t = Hashtbl.create 16 in
  (* Each executor has its own table, so two programs whose buffer names
     coincide still get two distinct device allocations. *)
  List.iter
    (function
      | Kernel_ir.Alloc b -> Hashtbl.replace bufs b.name (Buffer.alloc ~bytes:(buffer_bytes b))
      | Kernel_ir.Upload _ | Kernel_ir.Launch _ | Kernel_ir.Download _ | Kernel_ir.Free _ -> ())
    program.plan;
  let t =
    {
      program;
      module_;
      stream;
      bufs;
      kernels = List.map (fun (k : Kernel_ir.kernel) -> (k.name, k)) program.kernels;
      released = false;
    }
  in
  (* A dropped executor must not leak device memory. Tests call [release]
     explicitly; this is only the safety net, and it must never raise from
     inside the collector. *)
  Gc.finalise (fun t -> try release t with _ -> ()) t;
  t

let create program module_ = create_on ~stream:Stream.default program module_
let stream t = t.stream
let buffer_count t = Hashtbl.length t.bufs

(* Every op of a run is issued on the executor's own stream, so the plan
   is ordered by the stream's FIFO discipline rather than by the host: a
   launch cannot start before the uploads it reads, and a download cannot
   start before the launch that wrote its buffer. That is also what makes
   two successive jobs on one executor safe -- they share buffers, but
   the second job's uploads queue behind the first job's downloads. *)
let run_async t ~inputs =
  if t.released then failwith "Executor.run: this executor has been released";
  check_inputs t.program.plan ~inputs;
  let find (b : Kernel_ir.buffer) =
    match Hashtbl.find_opt t.bufs b.name with
    | Some d -> d
    | None -> failwith ("Executor.run: buffer " ^ b.name ^ " was never allocated")
  in
  let outputs = ref [] in
  (* Buffers outlive a run, so a [Download] of a buffer that this run never
     wrote would hand back the previous run's data. [Lower] never emits
     such a plan; check it rather than trust it. *)
  let written : (string, unit) Hashtbl.t = Hashtbl.create 16 in
  (* No [Fun.protect]: a failed run leaves the pool owned and intact.
     Freeing here would destroy buffers the caller still owns. *)
  List.iter
    (function
      | Kernel_ir.Alloc _ | Kernel_ir.Free _ -> ()
      | Kernel_ir.Upload { param; into } -> (
          match List.assoc_opt param inputs with
          | Some v ->
              Buffer.upload_async v (find into) ~stream:t.stream;
              Hashtbl.replace written into.name ()
          | None -> invalid_arg ("Executor.run: missing input " ^ param))
      | Kernel_ir.Launch { kernel; args } ->
          let k =
            match List.assoc_opt kernel t.kernels with
            | Some k -> k
            | None -> failwith ("Executor.run: unknown kernel " ^ kernel)
          in
          let { Schedule.grid; grid_y; block; block_y; shared_bytes } = k.launch in
          Launch.run ~stream:t.stream (Jit.get_kernel t.module_ kernel) ~grid ~grid_y ~block
            ~block_y ~shared_bytes (List.map find args);
          List.iter (fun (b : Kernel_ir.buffer) -> Hashtbl.replace written b.name ()) args
      | Kernel_ir.Download { from; output; shape } ->
          if not (Hashtbl.mem written from.name) then
            failwith
              ("Executor.run: buffer " ^ from.name
             ^ " is downloaded but was neither uploaded nor launched into on this run");
          (* Fresh every run: callers keep the outputs they are given. *)
          let v = fresh_value from.dtype shape in
          Buffer.download_async (find from) v ~stream:t.stream;
          outputs := (output, v) :: !outputs)
    t.program.plan;
  let finished = Event.record t.stream in
  let outputs = List.rev !outputs in
  fun () ->
    Event.synchronize finished;
    (* The input Values are the source of host-to-device copies that may
       still be in flight, and nothing else roots them; this closure
       does. [opaque_identity] stops the optimiser deciding otherwise. *)
    ignore (Sys.opaque_identity inputs);
    (* Opportunistic: when the whole stream happens to be drained this
       hands cudajit's retained launch arguments back. Never blocks, and
       waiting on the event alone would let them accumulate. *)
    ignore (Stream.is_idle t.stream);
    outputs

(* Synchronous by contract. The extra [Stream.synchronize] costs nothing
   -- the event has already fired -- and keeps v1's property that a run
   leaves no host-side launch state behind. *)
let run t ~inputs =
  let outputs = (run_async t ~inputs) () in
  Stream.synchronize t.stream;
  outputs

(* Resident runs substitute the caller's device buffers for the plan's own
   by name, which is zero-copy: the kernels read and write the caller's
   memory directly and not one byte crosses the bus. Substitution is
   sound because [Lower] gives every graph node its own buffer, so a
   param buffer is only ever written by its [Upload] and an output buffer
   is only ever written by the kernel that produced it. *)
let run_resident t ~inputs =
  if t.released then failwith "Executor.run_resident: this executor has been released";
  let overrides : (string, Buffer.t) Hashtbl.t = Hashtbl.create 16 in
  let find (b : Kernel_ir.buffer) =
    match Hashtbl.find_opt overrides b.name with
    | Some d -> d
    | None -> (
        match Hashtbl.find_opt t.bufs b.name with
        | Some d -> d
        | None -> failwith ("Executor.run_resident: buffer " ^ b.name ^ " was never allocated"))
  in
  (* Bind the inputs before anything is issued, so a size mismatch or a
     missing input costs no device work and leaks no allocation. *)
  List.iter
    (function
      | Kernel_ir.Upload { param; into } -> (
          match List.assoc_opt param inputs with
          | Some (b : Buffer.t) ->
              let want = buffer_bytes into and got = Buffer.byte_size b in
              if want <> got then
                invalid_arg
                  (Printf.sprintf
                     "Executor.run_resident: input %s is %d bytes, parameter %s needs %d" param got
                     into.name want);
              Hashtbl.replace overrides into.name b
          | None -> invalid_arg ("Executor.run_resident: missing input " ^ param))
      | Kernel_ir.Alloc _ | Kernel_ir.Launch _ | Kernel_ir.Download _ | Kernel_ir.Free _ -> ())
    t.program.plan;
  (* Outputs: one fresh buffer per [Download], substituted for the plan's
     own so the producing kernel writes straight into it. If the name is
     already bound -- a graph whose output *is* one of its params, or two
     downloads of one buffer -- fall back to a device-to-device copy. *)
  let copies : (string * Buffer.t) list ref = ref [] and outputs = ref [] in
  List.iter
    (function
      | Kernel_ir.Download { from; output; shape = _ } ->
          let b = Buffer.alloc ~bytes:(buffer_bytes from) in
          if Hashtbl.mem overrides from.name then copies := (from.name, b) :: !copies
          else Hashtbl.replace overrides from.name b;
          outputs := (output, b) :: !outputs
      | Kernel_ir.Alloc _ | Kernel_ir.Upload _ | Kernel_ir.Launch _ | Kernel_ir.Free _ -> ())
    t.program.plan;
  List.iter
    (function
      | Kernel_ir.Alloc _ | Kernel_ir.Free _ | Kernel_ir.Upload _ -> ()
      | Kernel_ir.Launch { kernel; args } ->
          let k =
            match List.assoc_opt kernel t.kernels with
            | Some k -> k
            | None -> failwith ("Executor.run_resident: unknown kernel " ^ kernel)
          in
          let { Schedule.grid; grid_y; block; block_y; shared_bytes } = k.launch in
          Launch.run ~stream:t.stream (Jit.get_kernel t.module_ kernel) ~grid ~grid_y ~block
            ~block_y ~shared_bytes (List.map find args)
      | Kernel_ir.Download { from; output = _; shape = _ } -> (
          match List.assoc_opt from.name !copies with
          | Some dst -> Buffer.copy_device ~dst ~src:(find from) ~stream:t.stream
          | None -> ()))
    t.program.plan;
  (* The caller gets device buffers, not a promise: they may hand them
     straight to another executor on another stream, or download them
     synchronously. One synchronise here is what makes either safe. *)
  Stream.synchronize t.stream;
  List.rev !outputs
