(** Scalar element types that may live in device memory.

    The type index ties a device dtype to the OCaml type of one element on
    the host, so [Expr] and [Tensor] nodes are well-typed by construction.

    [Bool] is an expression-only type: comparisons produce it and [Select]
    consumes it, but v1 has no bool tensors (Bigarray has no bool kind). *)

type _ t =
  | F32 : float t
  | F64 : float t
  | I32 : int32 t
  | I64 : int64 t
  | Bool : bool t

type packed = P : _ t -> packed

let size_in_bytes : type a. a t -> int = function
  | F32 | I32 -> 4
  | F64 | I64 -> 8
  | Bool -> 1

let name : type a. a t -> string = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I32 -> "i32"
  | I64 -> "i64"
  | Bool -> "bool"

let packed_name (P d) = name d

(** Runtime type-equality witness. Lets code that holds a [packed] dtype and
    a value of unknown element type recover a typed view safely. *)
type (_, _) eq = Equal : ('a, 'a) eq

let equal : type a b. a t -> b t -> (a, b) eq option =
 fun a b ->
  match (a, b) with
  | F32, F32 -> Some Equal
  | F64, F64 -> Some Equal
  | I32, I32 -> Some Equal
  | I64, I64 -> Some Equal
  | Bool, Bool -> Some Equal
  | _ -> None

let is_float : type a. a t -> bool = function F32 | F64 -> true | I32 | I64 | Bool -> false
