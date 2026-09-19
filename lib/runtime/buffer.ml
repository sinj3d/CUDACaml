(* Device memory. Only Bigarrays ever reach a memcpy: they live outside the
   OCaml heap, so the GC cannot move them under the driver's feet. *)

open Ocaml_cuda_ir

type t = { ptr : Cuda.Deviceptr.t; bytes : int }

let alloc ~bytes =
  Device.init ();
  (* CUDA rejects a 0-byte allocation, but a 0-element tensor still needs a
     valid pointer to hand to a kernel. *)
  { ptr = Cuda.Deviceptr.mem_alloc ~size_in_bytes:(max 1 bytes); bytes }

let free t = Cuda.Deviceptr.mem_free t.ptr
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
