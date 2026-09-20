(* Device discovery and the primary contexts that live for the whole
   process.

   cudajit exposes the driver API as a top-level [Cuda] module (library
   [cudajit.cuda]), not as [Cudajit.Cuda].

   One entry per device the caller has touched. A device's primary context
   is retained the first time it is asked for and never dropped: contexts
   are process-lifetime here, and a [Buffer] handed out under one must
   stay valid until its owner frees it. *)
let contexts : (int, Cuda.Device.t * Cuda.Context.t) Hashtbl.t = Hashtbl.create 4

(* Device 0 and its context, created once by [init]; also the flag that
   makes [init] idempotent. *)
let state : (Cuda.Device.t * Cuda.Context.t) option ref = ref None

(* The ordinal whose context is current on this thread. [init] binds
   device 0, and only [with_device] ever changes it -- always restoring
   what it found. *)
let cur = ref 0

(* Must not be called before [Cuda.init]. *)
let open_device ordinal =
  match Hashtbl.find_opt contexts ordinal with
  | Some dc -> dc
  | None ->
      let dev = Cuda.Device.get ~ordinal in
      let ctx = Cuda.Context.get_primary dev in
      Hashtbl.replace contexts ordinal (dev, ctx);
      (dev, ctx)

let init () =
  match !state with
  | Some _ -> ()
  | None ->
      Cuda.init ();
      let ((_, ctx) as dc) = open_device 0 in
      (* Binding the primary context to this thread is mandatory: without it
         every later driver call fails with CUDA_ERROR_INVALID_CONTEXT. *)
      Cuda.Context.set_current ctx;
      cur := 0;
      state := Some dc

(* Probed once; [available] must never raise and never print. *)
let probed : bool option ref = ref None

let available () =
  match !probed with
  | Some b -> b
  | None ->
      let b = match init () with () -> true | exception _ -> false in
      probed := Some b;
      b

let count () =
  init ();
  Cuda.Device.get_count ()

(* Restoring by ordinal rather than by [Cuda.Context.get_current] is
   deliberate: [cur] is the only thing that ever moves the binding, so it
   is the truth about what this thread was on, and looking the context up
   again cannot fail on a thread where no context is bound yet.

   [Fun.protect] is what makes the restore unconditional. A [with_device]
   that leaked its device on an exception would leave every later
   single-device call -- allocation, launch, free -- pointed at the wrong
   card, and it would not fail loudly. *)
let with_device ordinal f =
  init ();
  let _, ctx = open_device ordinal in
  let previous = !cur in
  Cuda.Context.set_current ctx;
  cur := ordinal;
  Fun.protect
    ~finally:(fun () ->
      let _, prev_ctx = open_device previous in
      Cuda.Context.set_current prev_ctx;
      cur := previous)
    f

let current () =
  init ();
  !cur

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

(* [get_free_and_total_mem] reads the *current* context, not a device
   handle, so the whole read happens under [with_device]. *)
let info_of ordinal =
  with_device ordinal (fun () ->
      let dev, _ = open_device ordinal in
      let a = Cuda.Device.get_attributes dev in
      let _free, total = Cuda.Device.get_free_and_total_mem () in
      {
        name = a.name;
        compute_capability = (a.compute_capability_major, a.compute_capability_minor);
        multiprocessors = a.multiprocessor_count;
        total_memory_bytes = total;
      })

let info () = info_of 0

let info_to_string i =
  let major, minor = i.compute_capability in
  Printf.sprintf "device: %s\ncompute: sm_%d%d\nmultiprocessors: %d\nmemory_mib: %d\n" i.name major
    minor i.multiprocessors
    (i.total_memory_bytes / 1_048_576)
