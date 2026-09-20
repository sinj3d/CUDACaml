(* Device memory. Only Bigarrays ever reach a memcpy: they live outside the
   OCaml heap, so the GC cannot move them under the driver's feet. *)

open Ocaml_cuda_ir

type t = { ptr : Cuda.Deviceptr.t; bytes : int }

(* Allocations minus frees. Every [alloc] is +1 and every [free] is -1,
   including the [max 1 bytes] dummy that a 0-element tensor gets: the
   count is of device allocations, not of bytes. *)
let live = ref 0
let live_count () = !live

let alloc ~bytes =
  Device.init ();
  (* CUDA rejects a 0-byte allocation, but a 0-element tensor still needs a
     valid pointer to hand to a kernel. *)
  let t = { ptr = Cuda.Deviceptr.mem_alloc ~size_in_bytes:(max 1 bytes); bytes } in
  incr live;
  t

let free t =
  Cuda.Deviceptr.mem_free t.ptr;
  decr live

let byte_size t = t.bytes
let unsafe_ptr t = t.ptr

let check_size ~who ~host ~device =
  if host <> device then
    invalid_arg
      (Printf.sprintf "Buffer.%s: host value is %d bytes, buffer is %d bytes" who host device)

let upload (Value.P v) t =
  check_size ~who:"upload" ~host:(Value.byte_size v) ~device:t.bytes;
  if t.bytes > 0 then
    match Value.raw v with
    | Value.Raw a ->
        Cuda.Deviceptr.memcpy_H_to_D ~dst:t.ptr ~src:(Bigarray.genarray_of_array1 a) ()

let download t (Value.P v) =
  check_size ~who:"download" ~host:(Value.byte_size v) ~device:t.bytes;
  if t.bytes > 0 then
    match Value.raw v with
    | Value.Raw a ->
        Cuda.Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 a) ~src:t.ptr ()

(* Asynchronous copies. These return as soon as the copy is *issued*, so
   the host Bigarray behind [v] must stay reachable until the stream is
   synchronised: nothing here roots it. Callers hold it. *)
let upload_async (Value.P v) t ~stream =
  check_size ~who:"upload_async" ~host:(Value.byte_size v) ~device:t.bytes;
  if t.bytes > 0 then
    match Value.raw v with
    | Value.Raw a ->
        Cuda.Stream.memcpy_H_to_D ~dst:t.ptr
          ~src:(Bigarray.genarray_of_array1 a)
          (Stream.unsafe_stream stream)

let download_async t (Value.P v) ~stream =
  check_size ~who:"download_async" ~host:(Value.byte_size v) ~device:t.bytes;
  if t.bytes > 0 then
    match Value.raw v with
    | Value.Raw a ->
        Cuda.Stream.memcpy_D_to_H
          ~dst:(Bigarray.genarray_of_array1 a)
          ~src:t.ptr
          (Stream.unsafe_stream stream)

(* Device to device, stream-ordered. Both buffers must be the same size:
   a partial copy here would be a silently wrong tensor. *)
let copy_device ~dst ~src ~stream =
  if dst.bytes <> src.bytes then
    invalid_arg
      (Printf.sprintf "Buffer.copy_device: destination is %d bytes, source is %d bytes" dst.bytes
         src.bytes);
  if dst.bytes > 0 then
    Cuda.Stream.memcpy_D_to_D ~size_in_bytes:dst.bytes ~dst:dst.ptr ~src:src.ptr
      (Stream.unsafe_stream stream)
