open Cudacaml_ir

(* Two NaNs count as equal; otherwise a relative-plus-absolute band, so a
   four-million-element float sum is judged on its magnitude rather than on
   an absolute epsilon it can never meet. *)
let float_close ~tolerance e g =
  if Float.is_nan e && Float.is_nan g then true
  else Float.abs (g -. e) <= tolerance *. (1.0 +. Float.abs e)

(* One dtype branch per OCaml element type: the witness from [Dtype.equal]
   is what lets us reach the typed elements at all. *)
let elem_equal : type a. a Dtype.t -> tolerance:float -> a -> a -> bool =
 fun d ~tolerance ->
  match d with
  | Dtype.F32 -> fun e g -> float_close ~tolerance e g
  | Dtype.F64 -> fun e g -> float_close ~tolerance e g
  | Dtype.I32 -> fun e g -> Int32.equal e g
  | Dtype.I64 -> fun e g -> Int64.equal e g
  | Dtype.Bool -> fun e g -> Bool.equal e g

(* Only ever used to render a mismatch message. *)
let elem_to_float : type a. a Dtype.t -> a -> float =
 fun d x ->
  match d with
  | Dtype.F32 -> x
  | Dtype.F64 -> x
  | Dtype.I32 -> Int32.to_float x
  | Dtype.I64 -> Int64.to_float x
  | Dtype.Bool -> if x then 1.0 else 0.0

let compare_elements : type a.
    tolerance:float -> string -> a Dtype.t -> a Value.t -> a Value.t -> (unit, string) result =
 fun ~tolerance output d e g ->
  let eq = elem_equal d ~tolerance in
  let n = Value.numel e in
  let rec loop i =
    if i >= n then Ok ()
    else
      let ev = Value.get e i and gv = Value.get g i in
      if eq ev gv then loop (i + 1)
      else
        Error
          (Printf.sprintf "output %s[%d]: expected %g, got %g (tol %g)" output i (elem_to_float d ev)
             (elem_to_float d gv) tolerance)
  in
  loop 0

let compare_one : type a b.
    tolerance:float -> string -> a Value.t -> b Value.t -> (unit, string) result =
 fun ~tolerance output e g ->
  let de = Value.dtype e and dg = Value.dtype g in
  match Dtype.equal de dg with
  | None ->
      Error
        (Printf.sprintf "output %s: dtype mismatch: expected %s, got %s" output (Dtype.name de)
           (Dtype.name dg))
  | Some Dtype.Equal ->
      let se = Value.shape e and sg = Value.shape g in
      if not (Shape.equal se sg) then
        Error
          (Printf.sprintf "output %s: shape mismatch: expected %s, got %s" output
             (Shape.to_string se) (Shape.to_string sg))
      else compare_elements ~tolerance output de e g

(* By name, never by position: a backend is free to emit its outputs in any
   order, and extra ones are not our business. *)
let compare_all ~tolerance expected got =
  let rec loop = function
    | [] -> Ok ()
    | (output, Value.P e) :: rest -> (
        match List.assoc_opt output got with
        | None -> Error (Printf.sprintf "missing output %s" output)
        | Some (Value.P g) -> (
            match compare_one ~tolerance output e g with
            | Ok () -> loop rest
            | Error _ as err -> err))
  in
  loop expected

let check ?(tolerance = 1e-5) ~reference ~candidate graph ~inputs =
  let module R = (val reference : Backend.S) in
  let module C = (val candidate : Backend.S) in
  match R.run (R.compile graph) ~inputs with
  | exception e -> Error (R.name ^ " raised: " ^ Printexc.to_string e)
  | expected -> (
      match C.run (C.compile graph) ~inputs with
      | exception e -> Error (C.name ^ " raised: " ^ Printexc.to_string e)
      | got -> compare_all ~tolerance expected got)
