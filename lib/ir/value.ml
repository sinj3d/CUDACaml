type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw
type 'a t = { dtype : 'a Dtype.t; shape : Shape.t; data : 'a raw }
type packed = P : _ t -> packed

let todo what = failwith ("Value." ^ what ^ ": TODO")
let create _ _ = todo "create"
let of_list _ _ _ = todo "of_list"
let dtype t = t.dtype
let shape t = t.shape
let numel t = Shape.numel t.shape
let byte_size t = numel t * Dtype.size_in_bytes t.dtype
let get _ _ = todo "get"
let set _ _ _ = todo "set"
let to_list _ = todo "to_list"
let raw t = t.data
