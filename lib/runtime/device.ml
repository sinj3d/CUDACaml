(* Device discovery and the one context that lives for the whole process.

   cudajit exposes the driver API as a top-level [Cuda] module (library
   [cudajit.cuda]), not as [Cudajit.Cuda]. *)

(* The device and its primary context, created once by [init]. *)
let state : (Cuda.Device.t * Cuda.Context.t) option ref = ref None

let init () =
  match !state with
  | Some _ -> ()
  | None ->
      Cuda.init ();
      let dev = Cuda.Device.get ~ordinal:0 in
      let ctx = Cuda.Context.get_primary dev in
      (* Binding the primary context to this thread is mandatory: without it
         every later driver call fails with CUDA_ERROR_INVALID_CONTEXT. *)
      Cuda.Context.set_current ctx;
      state := Some (dev, ctx)

(* Probed once; [available] must never raise and never print. *)
let probed : bool option ref = ref None

let available () =
  match !probed with
  | Some b -> b
  | None ->
      let b = match init () with () -> true | exception _ -> false in
      probed := Some b;
      b

let synchronize () =
  init ();
  Cuda.Context.synchronize ()

let name () =
  init ();
  match !state with
  | Some (dev, _) -> (Cuda.Device.get_attributes dev).name
  | None -> "cuda device 0"

type info = {
  name : string;
  compute_capability : int * int;
  multiprocessors : int;
  total_memory_bytes : int;
}

let info () =
  init ();
  match !state with
  | None -> failwith "Device.info: no CUDA device"
  | Some (dev, _) ->
      let a = Cuda.Device.get_attributes dev in
      let _free, total = Cuda.Device.get_free_and_total_mem () in
      {
        name = a.name;
        compute_capability = (a.compute_capability_major, a.compute_capability_minor);
        multiprocessors = a.multiprocessor_count;
        total_memory_bytes = total;
      }

let info_to_string i =
  let major, minor = i.compute_capability in
  Printf.sprintf "device: %s\ncompute: sm_%d%d\nmultiprocessors: %d\nmemory_mib: %d\n" i.name major
    minor i.multiprocessors
    (i.total_memory_bytes / 1_048_576)
