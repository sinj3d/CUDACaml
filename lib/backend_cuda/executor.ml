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
   in [release], and [run] does nothing but move data and launch. *)
type t = {
  program : Kernel_ir.program;
  module_ : Jit.module_;
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

let create (program : Kernel_ir.program) module_ =
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

let buffer_count t = Hashtbl.length t.bufs

let run t ~inputs =
  if t.released then failwith "Executor.run: this executor has been released";
  check_inputs t.program.plan ~inputs;
  let find (b : Kernel_ir.buffer) =
    match Hashtbl.find_opt t.bufs b.name with
    | Some d -> d
    | None -> failwith ("Executor.run: buffer " ^ b.name ^ " was never allocated")
  in
  let outputs = ref [] in
  (* Launches are asynchronous: nothing may be read back before a sync. *)
  let synced = ref false in
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
              Buffer.upload v (find into);
              Hashtbl.replace written into.name ()
          | None -> invalid_arg ("Executor.run: missing input " ^ param))
      | Kernel_ir.Launch { kernel; args } ->
          let k =
            match List.assoc_opt kernel t.kernels with
            | Some k -> k
            | None -> failwith ("Executor.run: unknown kernel " ^ kernel)
          in
          let { Schedule.grid; grid_y; block; block_y; shared_bytes } = k.launch in
          Launch.run (Jit.get_kernel t.module_ kernel) ~grid ~grid_y ~block ~block_y ~shared_bytes
            (List.map find args);
          List.iter (fun (b : Kernel_ir.buffer) -> Hashtbl.replace written b.name ()) args;
          synced := false
      | Kernel_ir.Download { from; output; shape } ->
          if not (Hashtbl.mem written from.name) then
            failwith
              ("Executor.run: buffer " ^ from.name
             ^ " is downloaded but was neither uploaded nor launched into on this run");
          if not !synced then begin
            Device.synchronize ();
            synced := true
          end;
          (* Fresh every run: callers keep the outputs they are given. *)
          let v = fresh_value from.dtype shape in
          Buffer.download (find from) v;
          outputs := (output, v) :: !outputs)
    t.program.plan;
  List.rev !outputs
