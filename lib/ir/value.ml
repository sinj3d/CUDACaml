type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw
type 'a t = { dtype : 'a Dtype.t; shape : Shape.t; data : 'a raw }
type packed = P : _ t -> packed

let dtype t = t.dtype
let shape t = t.shape
let numel t = Shape.numel t.shape
let byte_size t = numel t * Dtype.size_in_bytes t.dtype
let raw t = t.data

(* Locally abstract type: each dtype branch builds a Bigarray with a different
   element kind, all carrying the same element type [a]. Or-patterns would not
   refine [a], so the branches stay separate. *)
let create : type a. a Dtype.t -> Shape.t -> a t =
 fun dtype shape ->
  let n = Shape.numel shape in
  let data : a raw =
    match dtype with
    | Dtype.F32 ->
        let a = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout n in
        Bigarray.Array1.fill a 0.0;
        Raw a
    | Dtype.F64 ->
        let a = Bigarray.Array1.create Bigarray.Float64 Bigarray.C_layout n in
        Bigarray.Array1.fill a 0.0;
        Raw a
    | Dtype.I32 ->
        let a = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout n in
        Bigarray.Array1.fill a 0l;
        Raw a
    | Dtype.I64 ->
        let a = Bigarray.Array1.create Bigarray.Int64 Bigarray.C_layout n in
        Bigarray.Array1.fill a 0L;
        Raw a
    | Dtype.Bool -> invalid_arg "Value.create: Bool tensors are not storable"
  in
  { dtype; shape; data }

let get t i = match t.data with Raw a -> Bigarray.Array1.get a i
let set t i v = match t.data with Raw a -> Bigarray.Array1.set a i v
let to_list t = List.init (numel t) (get t)

let of_list dtype shape l =
  let n = Shape.numel shape in
  let got = List.length l in
  if got <> n then
    invalid_arg
      (Printf.sprintf "Value.of_list: shape %s holds %d elements, got %d"
         (Shape.to_string shape) n got);
  let t = create dtype shape in
  List.iteri (fun i v -> set t i v) l;
  t

(* Adopt a Bigarray that something else allocated: page-locked memory from
   [Runtime.Pinned], an mmap, a buffer a C library owns. The kind check is
   the whole point of the function. [F32] and [F64] are both
   [float Dtype.t], so the OCaml type of an element cannot tell a Float32
   Bigarray from a Float64 one, and adopting the wrong one would hand the
   driver half, or twice, the bytes it is told to copy. *)
let of_raw : type a. a Dtype.t -> Shape.t -> a raw -> a t =
 fun dtype shape data ->
  (match dtype with
  | Dtype.Bool -> invalid_arg "Value.of_raw: Bool tensors are not storable"
  | Dtype.F32 | Dtype.F64 | Dtype.I32 | Dtype.I64 -> ());
  (match data with
  | Raw a ->
      let n = Shape.numel shape and got = Bigarray.Array1.dim a in
      if got <> n then
        invalid_arg
          (Printf.sprintf "Value.of_raw: shape %s holds %d elements, the Bigarray holds %d"
             (Shape.to_string shape) n got);
      let kind_matches =
        match (dtype, Bigarray.Array1.kind a) with
        | Dtype.F32, Bigarray.Float32 -> true
        | Dtype.F64, Bigarray.Float64 -> true
        | Dtype.I32, Bigarray.Int32 -> true
        | Dtype.I64, Bigarray.Int64 -> true
        | _ -> false
      in
      if not kind_matches then
        invalid_arg
          (Printf.sprintf "Value.of_raw: dtype %s does not match the Bigarray's element kind"
             (Dtype.name dtype)));
  { dtype; shape; data }
