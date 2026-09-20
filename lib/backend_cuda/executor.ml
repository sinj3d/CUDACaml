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

let run (program : Kernel_ir.program) module_ ~inputs =
  check_inputs program.plan ~inputs;
  Device.init ();
  let bufs : (string, Buffer.t) Hashtbl.t = Hashtbl.create 16 in
  let kernels = List.map (fun (k : Kernel_ir.kernel) -> (k.name, k)) program.kernels in
  let find (b : Kernel_ir.buffer) =
    match Hashtbl.find_opt bufs b.name with
    | Some d -> d
    | None -> failwith ("Executor.run: buffer " ^ b.name ^ " was never allocated")
  in
  let outputs = ref [] in
  (* Launches are asynchronous: nothing may be read back before a sync. *)
  let synced = ref false in
  Fun.protect
    ~finally:(fun () -> Hashtbl.iter (fun _ b -> try Buffer.free b with _ -> ()) bufs)
    (fun () ->
      List.iter
        (function
          | Kernel_ir.Alloc b -> Hashtbl.replace bufs b.name (Buffer.alloc ~bytes:(buffer_bytes b))
          | Kernel_ir.Upload { param; into } -> (
              match List.assoc_opt param inputs with
              | Some v -> Buffer.upload v (find into)
              | None -> invalid_arg ("Executor.run: missing input " ^ param))
          | Kernel_ir.Launch { kernel; args } ->
              let k =
                match List.assoc_opt kernel kernels with
                | Some k -> k
                | None -> failwith ("Executor.run: unknown kernel " ^ kernel)
              in
              let { Schedule.grid; grid_y; block; block_y; shared_bytes } = k.launch in
              Launch.run (Jit.get_kernel module_ kernel) ~grid ~grid_y ~block ~block_y
                ~shared_bytes (List.map find args);
              synced := false
          | Kernel_ir.Download { from; output; shape } ->
              if not !synced then begin
                Device.synchronize ();
                synced := true
              end;
              let v = fresh_value from.dtype shape in
              Buffer.download (find from) v;
              outputs := (output, v) :: !outputs
          | Kernel_ir.Free b ->
              Buffer.free (find b);
              Hashtbl.remove bufs b.name)
        program.plan;
      List.rev !outputs)
