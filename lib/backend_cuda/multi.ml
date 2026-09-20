(* Data-parallel runs across several devices.

   The whole point of this module is discipline about *which context is
   current*. A CUDA context is per-device and bound to a thread; a buffer
   allocated under one is not a valid pointer under another, a module is
   loaded into one context, and a stream belongs to one. So every call
   below that can reach the driver -- the JIT compile, the executor's
   allocations, the stream creation, issuing a run, waiting on it, and
   freeing -- is wrapped in [Device.with_device], and nothing in between
   touches the driver. *)

open Ocaml_cuda_ir
open Ocaml_cuda_lower
open Ocaml_cuda_runtime

(* One device's share of the work: the ordinal it runs on and the
   executor that owns its buffers, its module and its stream. All three
   belong to that ordinal's context and may only be used under it. *)
type slot = { ordinal : int; exec : Executor.t }

type t = { slots : slot array; n : int; per : int; mutable released : bool }

(* ------------------------------------------------------------------ *)
(* Host-side slicing and combining                                     *)
(* ------------------------------------------------------------------ *)

(* Elements per index along the leading axis. *)
let row_stride dims = match dims with _ :: rest -> List.fold_left ( * ) 1 rest | [] -> 1

(* There is no view type in [Value], so a slice is a copy into a fresh
   value of the sliced shape. It is a host-side copy of n/d elements,
   which is small next to the upload it feeds. *)
let slice ~index ~per (Value.P v) =
  let dims = Shape.dims (Value.shape v) in
  let rest = match dims with _ :: r -> r | [] -> [] in
  let stride = row_stride dims in
  let count = per * stride in
  let offset = index * count in
  let dst = Value.create (Value.dtype v) (Shape.of_dims (per :: rest)) in
  for k = 0 to count - 1 do
    Value.set dst k (Value.get v (offset + k))
  done;
  Value.P dst

(* The split axis rule, by shape alone: an input is sliced exactly when
   its leading dimension is the total [n]. A rank-0 input (every scalar
   parameter, including an RNG key) is passed whole, as is anything whose
   leading dimension is something else. *)
let leading_dim (Value.P v) = match Shape.dims (Value.shape v) with d :: _ -> Some d | [] -> None

let concat (vs : Value.packed list) : Value.packed =
  match vs with
  | [] -> invalid_arg "Multi.concat: no values to concatenate"
  | Value.P first :: _ ->
      let dt = Value.dtype first in
      let rest =
        match Shape.dims (Value.shape first) with
        | _ :: r -> r
        | [] -> invalid_arg "Multi.concat: a rank-0 value has no leading axis"
      in
      let lead =
        List.fold_left
          (fun acc (Value.P v) ->
            match Shape.dims (Value.shape v) with
            | l :: r when r = rest -> acc + l
            | _ ->
                invalid_arg
                  (Printf.sprintf "Multi.concat: shape %s does not match %s outside the leading axis"
                     (Shape.to_string (Value.shape v))
                     (Shape.to_string (Value.shape first))))
          0 vs
      in
      let dst = Value.create dt (Shape.of_dims (lead :: rest)) in
      let pos = ref 0 in
      List.iter
        (fun (Value.P v) ->
          (* The witness is what lets us read elements of a value whose
             element type is existential: without it [Value.get v] has no
             type in common with [Value.set dst]. *)
          match Dtype.equal (Value.dtype v) dt with
          | None ->
              invalid_arg
                (Printf.sprintf "Multi.concat: dtype %s does not match %s"
                   (Dtype.name (Value.dtype v)) (Dtype.name dt))
          | Some Dtype.Equal ->
              let k = Value.numel v in
              for i = 0 to k - 1 do
                Value.set dst (!pos + i) (Value.get v i)
              done;
              pos := !pos + k)
        vs;
      Value.P dst

let scalar_float (Value.P v) : float =
  if Value.numel v <> 1 then
    invalid_arg
      (Printf.sprintf "Multi.mean_scalar: %s is not a scalar" (Shape.to_string (Value.shape v)));
  match Value.dtype v with
  | Dtype.F32 -> Value.get v 0
  | Dtype.F64 -> Value.get v 0
  | Dtype.I32 -> Int32.to_float (Value.get v 0)
  | Dtype.I64 -> Int64.to_float (Value.get v 0)
  | Dtype.Bool -> invalid_arg "Multi.mean_scalar: a bool output has no mean"

let mean_scalar vs =
  match vs with
  | [] -> invalid_arg "Multi.mean_scalar: no values to average"
  | _ ->
      List.fold_left (fun acc v -> acc +. scalar_float v) 0.0 vs /. float_of_int (List.length vs)

(* ------------------------------------------------------------------ *)
(* Compilation and running                                             *)
(* ------------------------------------------------------------------ *)

let release t =
  if not t.released then begin
    t.released <- true;
    Array.iter
      (fun s -> Device.with_device s.ordinal (fun () -> Executor.release s.exec))
      t.slots
  end

let create ~devices ~n build =
  let d = List.length devices in
  if d = 0 then invalid_arg "Multi.create: devices must not be empty";
  if n mod d <> 0 then
    invalid_arg (Printf.sprintf "Multi.create: n = %d is not divisible by %d devices" n d);
  let per = n / d in
  let slots =
    Array.map
      (fun ordinal ->
        (* Passes, lowering and emission are pure host work, but the JIT
           loads the module into the current context and the executor
           allocates its pool there, so the whole build runs under the
           target device. Each slot gets its own non-blocking stream: on
           distinct cards that is what lets the launches overlap, and on a
           repeated ordinal it keeps the two lanes from serialising on the
           NULL stream. *)
        Device.with_device ordinal (fun () ->
            let program = build ~n:per |> Ocaml_cuda_passes.Pipeline.run |> Lower.program in
            let module_ = Jit.compile ~name:program.name ~source:(Emit.program program) in
            { ordinal; exec = Executor.create_on ~stream:(Stream.create ()) program module_ }))
      (Array.of_list devices)
  in
  let t = { slots; n; per; released = false } in
  (* Safety net only; callers [release]. It matters more here than for a
     single-device executor: [Executor]'s own finaliser would free device
     1's pointers under whatever context happened to be current, whereas
     this one restores the right card first. [t] holds the executors, so
     it is finalised before they are. *)
  Gc.finalise (fun t -> try release t with _ -> ()) t;
  t

let run_per_device t ~inputs =
  if t.released then failwith "Multi.run: this Multi.t has been released";
  (* Two passes, and the split matters: every device is launched before
     any is waited on, so the cards run at the same time instead of one
     after another. *)
  let jobs =
    Array.mapi
      (fun i s ->
        let ins = inputs i in
        Device.with_device s.ordinal (fun () -> Executor.run_async s.exec ~inputs:ins))
      t.slots
  in
  Array.to_list
    (Array.mapi (fun i s -> Device.with_device s.ordinal (fun () -> jobs.(i) ())) t.slots)

let run t ~inputs =
  run_per_device t ~inputs:(fun i ->
      List.map
        (fun (name, v) ->
          if leading_dim v = Some t.n then (name, slice ~index:i ~per:t.per v) else (name, v))
        inputs)
