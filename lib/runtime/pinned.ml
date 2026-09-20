(* Page-locked host memory.

   cudajit 0.7 binds cuMemAlloc but not cuMemHostAlloc, so this is the one
   place in the project that reaches for the driver by hand. Everything
   about that reach is [lazy]: a machine with no libcuda.so.1 -- Windows,
   a CPU-only box -- must still be able to *load* this module, and only
   fail if something actually asks it for pinned memory. *)

open Cudacaml_ir

(* CU_MEMHOSTALLOC_PORTABLE: page-locked for every context in the process,
   not only for the one that happened to be current at allocation. *)
let portable = 0x01
let driver = lazy (Dl.dlopen ~filename:"libcuda.so.1" ~flags:[ Dl.RTLD_NOW ])

let host_alloc =
  lazy
    (Foreign.foreign ~from:(Lazy.force driver) "cuMemHostAlloc"
       Ctypes.(ptr (ptr void) @-> size_t @-> uint @-> returning int))

let host_free =
  lazy
    (Foreign.foreign ~from:(Lazy.force driver) "cuMemFreeHost"
       Ctypes.(ptr void @-> returning int))

(* Which host addresses are ours. Keyed by the address of the allocation
   and not by the Bigarray itself: [Hashtbl.hash] of a Bigarray hashes its
   *contents*, which the owner is free to overwrite between a write and a
   lookup, so a Bigarray cannot be a hash key at all. The finaliser drops
   the entry in the same breath as it frees the block, so the table never
   holds an address the driver has taken back, and never grows without
   bound. *)
let owned : (nativeint, unit) Hashtbl.t = Hashtbl.create 64

let address_of_raw : type a. a Value.raw -> nativeint =
 fun (Value.Raw a) ->
  Ctypes.raw_address_of_ptr (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.array1 a))

let is_pinned v = Hashtbl.mem owned (address_of_raw (Value.raw v))

let alloc_bytes n =
  Device.init ();
  let out = Ctypes.(allocate (ptr void) null) in
  (* CUDA rejects a 0-byte allocation, but a 0-element tensor still needs
     an address to hand to a copy. *)
  let rc =
    (Lazy.force host_alloc) out
      (Unsigned.Size_t.of_int (max 1 n))
      (Unsigned.UInt.of_int portable)
  in
  if rc <> 0 then failwith (Printf.sprintf "Pinned.alloc: cuMemHostAlloc failed with %d" rc);
  Ctypes.( !@ ) out

(* One finaliser per allocation, attached to the Bigarray -- the only
   thing that points at the block. While any [Value] can reach that
   Bigarray, and so while any copy in flight whose job closure holds the
   [Value] can reach it, the block cannot be collected and cannot be
   freed; when the last reference goes, the driver gets the memory back
   exactly once. Nothing else in the project calls cuMemFreeHost.

   The closure captures the pointer and the address, never [ba]: a
   finaliser that captured the value it is attached to would keep it
   alive for ever. *)
let own p ba =
  let addr = Ctypes.raw_address_of_ptr p in
  Hashtbl.replace owned addr ();
  Gc.finalise
    (fun _ ->
      Hashtbl.remove owned addr;
      ignore ((Lazy.force host_free) p : int))
    ba

(* Locally abstract, like [Value.create]: each dtype branch builds a
   Bigarray of a different element kind over the same block, and the Bool
   branch allocates nothing at all. *)
let alloc : type a. a Dtype.t -> Shape.t -> a Value.t =
 fun dtype shape ->
  let n = Shape.numel shape in
  let bytes = n * Dtype.size_in_bytes dtype in
  let raw : a Value.raw =
    match dtype with
    | Dtype.F32 ->
        let p = alloc_bytes bytes in
        let a = Ctypes.(bigarray_of_ptr array1 n Bigarray.Float32 (from_voidp float p)) in
        Bigarray.Array1.fill a 0.0;
        own p a;
        Value.Raw a
    | Dtype.F64 ->
        let p = alloc_bytes bytes in
        let a = Ctypes.(bigarray_of_ptr array1 n Bigarray.Float64 (from_voidp double p)) in
        Bigarray.Array1.fill a 0.0;
        own p a;
        Value.Raw a
    | Dtype.I32 ->
        let p = alloc_bytes bytes in
        let a = Ctypes.(bigarray_of_ptr array1 n Bigarray.Int32 (from_voidp int32_t p)) in
        Bigarray.Array1.fill a 0l;
        own p a;
        Value.Raw a
    | Dtype.I64 ->
        let p = alloc_bytes bytes in
        let a = Ctypes.(bigarray_of_ptr array1 n Bigarray.Int64 (from_voidp int64_t p)) in
        Bigarray.Array1.fill a 0L;
        own p a;
        Value.Raw a
    | Dtype.Bool -> invalid_arg "Pinned.alloc: Bool tensors are not storable"
  in
  Value.of_raw dtype shape raw

(* Element by element: the two Bigarrays' element kinds are existential
   inside [Value.raw], so nothing here can prove to [Bigarray.Array1.blit]
   that they are the same kind, even though the equal dtypes guarantee it. *)
let of_value v =
  let dst = alloc (Value.dtype v) (Value.shape v) in
  for i = 0 to Value.numel v - 1 do
    Value.set dst i (Value.get v i)
  done;
  dst
