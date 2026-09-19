(** saxpy: r = a*x + y, then s = sum r.

    Exercises the two things v1 must do well: the [map] into [map2] fuses
    into one kernel, and the [reduce] is a second kernel with its own
    schedule. *)

open Ocaml_cuda
open Dsl

let program ~n ~a =
  let shape = Shape.of_dims [ n ] in
  let x = param "x" Dtype.F32 shape in
  let y = param "y" Dtype.F32 shape in
  let ax = map (fun xi -> mul (const Dtype.F32 a) xi) x in
  let r = map2 add ax y in
  let s = reduce add ~init:(const Dtype.F32 0.0) r in
  Graph.create ~name:"saxpy" ~outputs:[ ("r", Tensor.P r); ("s", Tensor.P s) ]
