(** Registry of example programs with sample inputs. Used by the CLI, the
    interpreter tests and the gated system tests. Every program here must
    stay within the v1 subset (static shapes, no broadcasting). *)

open Ocaml_cuda
open Dsl

type t = {
  name : string;
  graph : unit -> Graph.t;
  inputs : unit -> (string * Value.packed) list;
}

let f32s dims f =
  let n = Shape.numel (Shape.of_dims dims) in
  Value.P (Value.of_list Dtype.F32 (Shape.of_dims dims) (List.init n f))

let vec n = Shape.of_dims [ n ]
let x_in n = ("x", f32s [ n ] (fun i -> float_of_int (i mod 17) -. 8.0))
let y_in n = ("y", f32s [ n ] (fun i -> float_of_int (n - i) /. 4.0))
let out name t = (name, Tensor.P t)

(* r = 2x + y ; s = sum r.  One fused map kernel + one reduce kernel. *)
let saxpy n =
  { name = "saxpy"; graph = (fun () -> Saxpy.program ~n ~a:2.0); inputs = (fun () -> [ x_in n; y_in n ]) }

(* s = sum x, with n not a multiple of the block size. *)
let sum n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    Graph.create ~name:"sum" ~outputs:[ out "s" (reduce add ~init:(const Dtype.F32 0.0) x) ]
  in
  { name = "sum"; graph; inputs = (fun () -> [ x_in n ]) }

(* m = max x, using a finite identity so no infinity literal is needed. *)
let maxval n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    Graph.create ~name:"maxval" ~outputs:[ out "m" (reduce max ~init:(const Dtype.F32 (-1e30)) x) ]
  in
  { name = "maxval"; graph; inputs = (fun () -> [ x_in n ]) }

(* p = inclusive prefix sum of x. *)
let prefix n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    Graph.create ~name:"prefix" ~outputs:[ out "p" (scan add ~init:(const Dtype.F32 0.0) x) ]
  in
  { name = "prefix"; graph; inputs = (fun () -> [ x_in n ]) }

(* r[i] = x[n-1-i], via an index tensor built from iota. *)
let reverse n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    let idx = map (fun i -> sub (const Dtype.I32 (Int32.of_int (n - 1))) i) (iota (vec n)) in
    Graph.create ~name:"reverse" ~outputs:[ out "r" (gather idx x) ]
  in
  { name = "reverse"; graph; inputs = (fun () -> [ x_in n ]) }

(* r = clamp x to [0, 1]: two nested selects, exercises divergence. *)
let clamp n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    let zero = const Dtype.F32 0.0 and one = const Dtype.F32 1.0 in
    let r = map (fun e -> select (lt e zero) zero (select (lt one e) one e)) x in
    Graph.create ~name:"clamp" ~outputs:[ out "r" r ]
  in
  { name = "clamp"; graph; inputs = (fun () -> [ x_in n ]) }

(* r[i] = float(i)^2 : no inputs at all; iota + cast inlined into one kernel. *)
let squares n =
  let graph () =
    let r = map (fun i -> let f = cast Dtype.F32 i in mul f f) (iota (vec n)) in
    Graph.create ~name:"squares" ~outputs:[ out "r" r ]
  in
  { name = "squares"; graph; inputs = (fun () -> []) }

(* Five chained maps: must lower to exactly ONE kernel. *)
let chain n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    let one = const Dtype.F32 1.0 in
    let r = x |> map (fun e -> add e one) |> map (fun e -> mul e e) |> map neg
            |> map (fun e -> sub e one) |> map (fun e -> mul e (const Dtype.F32 0.5)) in
    Graph.create ~name:"chain" ~outputs:[ out "r" r ]
  in
  { name = "chain"; graph; inputs = (fun () -> [ x_in n ]) }

(* s = x*x is used three times: it must be materialised exactly once. *)
let fanout n =
  let graph () =
    let x = param "x" Dtype.F32 (vec n) in
    let s = map (fun e -> mul e e) x in
    Graph.create ~name:"fanout"
      ~outputs:[ out "a" (map neg s); out "b" (map sqrt s); out "c" (reduce add ~init:(const Dtype.F32 0.0) s) ]
  in
  { name = "fanout"; graph; inputs = (fun () -> [ x_in n ]) }

(* 2-D input flattened then negated; reshape is metadata only. *)
let reshape_flat =
  let graph () =
    let x = param "x" Dtype.F32 (Shape.of_dims [ 4; 8 ]) in
    Graph.create ~name:"reshape_flat" ~outputs:[ out "r" (map neg (reshape (vec 32) x)) ]
  in
  { name = "reshape_flat"; graph; inputs = (fun () -> [ ("x", f32s [ 4; 8 ] float_of_int) ]) }

let all =
  [ saxpy 1024; sum 1000; maxval 257; prefix 300; reverse 100; clamp 512; squares 64;
    chain 4096; fanout 1000; reshape_flat ]

let find name = List.find_opt (fun e -> String.equal e.name name) all
