(** Host-side tensor data.

    Always a [Bigarray]: allocated outside the OCaml heap, never moved by the
    GC, contiguous C layout, so a device transfer is a pointer plus a byte
    count. Nothing else ever crosses the FFI. *)

type 'a t
type packed = P : _ t -> packed

val create : 'a Dtype.t -> Shape.t -> 'a t
val of_list : 'a Dtype.t -> Shape.t -> 'a list -> 'a t
val dtype : 'a t -> 'a Dtype.t
val shape : 'a t -> Shape.t
val numel : _ t -> int
val byte_size : _ t -> int
val get : 'a t -> int -> 'a
val set : 'a t -> int -> 'a -> unit
val to_list : 'a t -> 'a list

(** Underlying storage, for the runtime upload/download path only. The
    element kind is existential because it depends on the dtype. *)
type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw

val raw : 'a t -> 'a raw

(** Wrap an existing C-layout Bigarray, sharing its storage: writes through
    one are visible through the other. [Invalid_argument] if its kind does
    not match [dtype] (F32 -> Float32, F64 -> Float64, I32 -> Int32,
    I64 -> Int64; Bool is rejected) or its length is not
    [Shape.numel shape]. *)
val of_raw : 'a Dtype.t -> Shape.t -> 'a raw -> 'a t
