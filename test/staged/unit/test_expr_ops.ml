(* T15: bit ops, shifts, Sin/Cos/Erf/Erfinv, and the missing comparison /
   logic surface. Integer results are exact; float results in F64 are
   checked tightly. No GPU: emit checks look at the generated source. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let i32 l = Value.P (Value.of_list Dtype.I32 (vec (List.length l)) l)
let i64 l = Value.P (Value.of_list Dtype.I64 (vec (List.length l)) l)
let f64 l = Value.P (Value.of_list Dtype.F64 (vec (List.length l)) l)
let f32 l = Value.P (Value.of_list Dtype.F32 (vec (List.length l)) l)
let one name t = [ (name, Tensor.P t) ]
let h = Int32.of_string

let ints o name : int32 list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.I32 -> Value.to_list v | _ -> failwith "not i32")

let longs o name : int64 list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.I64 -> Value.to_list v | _ -> failwith "not i64")

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> (
      match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> failwith "not float")

(* r = f x y on I32 vectors *)
let binop32 f xs ys =
  let x = param "x" Dtype.I32 (vec (List.length xs)) and y = param "y" Dtype.I32 (vec (List.length ys)) in
  let g = Graph.create ~name:"g" ~outputs:(one "r" (map2 f x y)) in
  ints (run g [ ("x", i32 xs); ("y", i32 ys) ]) "r"

let binop64 f xs ys =
  let x = param "x" Dtype.I64 (vec (List.length xs)) and y = param "y" Dtype.I64 (vec (List.length ys)) in
  let g = Graph.create ~name:"g" ~outputs:(one "r" (map2 f x y)) in
  longs (run g [ ("x", i64 xs); ("y", i64 ys) ]) "r"

let unop_f64 f xs =
  let x = param "x" Dtype.F64 (vec (List.length xs)) in
  let g = Graph.create ~name:"g" ~outputs:(one "r" (map f x)) in
  floats (run g [ ("x", f64 xs) ]) "r"

let has_sub ~sub s =
  let n = String.length sub and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = sub || go (i + 1)) in
  go 0

let ints_equal ~expect got =
  if List.length expect <> List.length got then C.fail "length";
  List.iteri (fun i (e, g) -> if not (Int32.equal e g) then C.fail "index %d: expected %ld, got %ld" i e g) (List.combine expect got)

let () =
  C.test "i32 xor / and / or on masks" (fun () ->
      let a = [ h "0x0F0F0F0F" ] and b = [ h "0x00FF00FF" ] in
      ints_equal ~expect:[ h "0x0FF00FF0" ] (binop32 bit_xor a b);
      ints_equal ~expect:[ h "0x000F000F" ] (binop32 bit_and a b);
      ints_equal ~expect:[ h "0x0FFF0FFF" ] (binop32 bit_or a b));
  C.test "i32 bit ops on negatives" (fun () ->
      ints_equal ~expect:[ -6l ] (binop32 bit_xor [ -1l ] [ 5l ]);
      ints_equal ~expect:[ 248l ] (binop32 bit_and [ -8l ] [ 255l ]);
      ints_equal ~expect:[ -5l ] (binop32 bit_or [ -8l ] [ 3l ]));
  C.test "i32 shifts: logical shr, masked counts" (fun () ->
      ints_equal ~expect:[ Int32.min_int ] (binop32 shl [ 1l ] [ 31l ]);
      ints_equal ~expect:[ 15l ] (binop32 shr [ -1l ] [ 28l ]);
      ints_equal ~expect:[ 12345l ] (binop32 shr [ 12345l ] [ 32l ]);
      ints_equal ~expect:[ 6l ] (binop32 shl [ 3l ] [ 33l ]);
      ints_equal ~expect:[ 1l ] (binop32 shr [ Int32.min_int ] [ 31l ]));
  C.test "i64 shifts" (fun () ->
      (match binop64 shr [ -1L ] [ 60L ] with [ v ] -> if not (Int64.equal v 15L) then C.fail "shr" | _ -> C.fail "len");
      match binop64 shl [ 1L ] [ 63L ] with
      | [ v ] -> if not (Int64.equal v Int64.min_int) then C.fail "shl"
      | _ -> C.fail "len");
  C.test "bit ops on floats are rejected by the interpreter" (fun () ->
      let x = param "x" Dtype.F32 (vec 1) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map2 bit_and x x)) in
      C.raises (fun () -> ignore (run g [ ("x", f32 [ 1.0 ]) ])));
  C.test "sin, cos, erf in F64" (fun () ->
      let pi2 = Float.pi /. 2.0 in
      C.floats ~tol:1e-12 ~expect:[ 0.; 1. ] (unop_f64 sin [ 0.; pi2 ]);
      C.floats ~tol:1e-12 ~expect:[ 1.; 0. ] (unop_f64 cos [ 0.; pi2 ]);
      C.floats ~tol:1e-15 ~expect:[ 0.; 0.8427007929497149 ] (unop_f64 erf [ 0.; 1. ]));
  C.test "erfinv inverts erf to 1e-12 in F64" (fun () ->
      let xs = [ -2.; -0.5; 0.; 0.3; 1.5; 2.5 ] in
      C.floats ~tol:1e-12 ~expect:xs (unop_f64 (fun e -> erfinv (erf e)) xs));
  C.test "erfinv at the edges" (fun () ->
      match unop_f64 erfinv [ 1.0; 0.0; -1.0 ] with
      | [ p; z; m ] ->
          C.bool ~expect:true (p = Float.infinity);
          C.float ~tol:0.0 ~expect:0.0 z;
          C.bool ~expect:true (m = Float.neg_infinity)
      | _ -> C.fail "length");
  C.test "erfinv in F32" (fun () ->
      let x = param "x" Dtype.F32 (vec 1) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map erfinv x)) in
      C.floats ~tol:1e-6 ~expect:[ 0.4769362762 ] (floats (run g [ ("x", f32 [ 0.5 ]) ]) "r"));
  C.test "ne / gt / ge / and_ / or_ / not_ through select" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) and y = param "y" Dtype.F32 (vec 3) in
      let b c = select c (const Dtype.F32 1.0) (const Dtype.F32 0.0) in
      let k v = const Dtype.F32 v in
      let outs =
        [
          ("ne", Tensor.P (map2 (fun a c -> b (ne a c)) x y));
          ("gt", Tensor.P (map2 (fun a c -> b (gt a c)) x y));
          ("ge", Tensor.P (map2 (fun a c -> b (ge a c)) x y));
          ("and", Tensor.P (map (fun a -> b (and_ (gt a (k 1.0)) (lt a (k 3.0)))) x));
          ("or", Tensor.P (map (fun a -> b (or_ (lt a (k 2.0)) (gt a (k 2.0)))) x));
          ("not", Tensor.P (map2 (fun a c -> b (not_ (eq a c))) x y));
        ]
      in
      let o = run (Graph.create ~name:"g" ~outputs:outs) [ ("x", f32 [ 1.; 2.; 3. ]); ("y", f32 [ 2.; 2.; 2. ]) ] in
      C.floats ~expect:[ 1.; 0.; 1. ] (floats o "ne");
      C.floats ~expect:[ 0.; 0.; 1. ] (floats o "gt");
      C.floats ~expect:[ 0.; 1.; 1. ] (floats o "ge");
      C.floats ~expect:[ 0.; 1.; 0. ] (floats o "and");
      C.floats ~expect:[ 1.; 0.; 1. ] (floats o "or");
      C.floats ~expect:[ 1.; 0.; 1. ] (floats o "not"));
  C.test "emit: shifts go through unsigned with a masked count" (fun () ->
      let x = param "x" Dtype.I32 (vec 4) in
      let src = Backend_cuda.source (Graph.create ~name:"g" ~outputs:(one "r" (map (fun e -> shr e (const Dtype.I32 3l)) x))) in
      C.contains ~sub:"unsigned int" src;
      C.contains ~sub:"& 31" src;
      let y = param "y" Dtype.I64 (vec 4) in
      let src64 = Backend_cuda.source (Graph.create ~name:"g" ~outputs:(one "r" (map (fun e -> shl e (const Dtype.I64 3L)) y))) in
      C.contains ~sub:"unsigned long long" src64;
      C.contains ~sub:"& 63" src64);
  C.test "emit: f32 and f64 intrinsic spellings" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let s32 = Backend_cuda.source (Graph.create ~name:"g" ~outputs:(one "r" (map (fun e -> sin (erfinv e)) x))) in
      C.contains ~sub:"erfinvf(" s32;
      C.contains ~sub:"sinf(" s32;
      let y = param "y" Dtype.F64 (vec 4) in
      let s64 = Backend_cuda.source (Graph.create ~name:"g" ~outputs:(one "r" (map (fun e -> cos (erfinv e)) y))) in
      C.contains ~sub:"erfinv(" s64;
      C.contains ~sub:"cos(" s64;
      C.bool ~expect:false (has_sub ~sub:"erfinvf(" s64));
  C.run ()
