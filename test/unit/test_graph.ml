(* Graph + Dsl validation. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check

let shape = Shape.of_dims [ 8 ]
let uid t = Uid.to_int (Tensor.uid (Tensor.P t))
let uids g = List.map (fun p -> Uid.to_int (Tensor.uid p)) (Graph.topological_order g)

let pos order u =
  let rec go i = function [] -> -1 | x :: xs -> if x = u then i else go (i + 1) xs in
  go 0 order

let () =
  C.test "params discovered in first-use order, deduped" (fun () ->
      let x = param "x" Dtype.F32 shape and y = param "y" Dtype.F32 shape in
      let r = map2 add (map (fun e -> mul e e) x) y in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r) ] in
      C.bool ~expect:true (List.map fst (Graph.params g) = [ "x"; "y" ]));
  C.test "topological order puts deps before dependents, no duplicates" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let a = map neg x in
      let b = map neg a in
      let g = Graph.create ~name:"g" ~outputs:[ ("b", Tensor.P b) ] in
      let o = uids g in
      C.int ~expect:3 (List.length o);
      C.bool ~expect:true (pos o (uid x) < pos o (uid a) && pos o (uid a) < pos o (uid b)));
  C.test "shared node appears once; fan_out counts uses" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let s = map (fun e -> mul e e) x in
      let a = map neg s and b = map sqrt s in
      let g = Graph.create ~name:"g" ~outputs:[ ("a", Tensor.P a); ("b", Tensor.P b) ] in
      C.int ~expect:4 (List.length (uids g));
      C.int ~expect:2 (Graph.fan_out g (Tensor.P s));
      C.int ~expect:1 (Graph.fan_out g (Tensor.P x));
      C.int ~expect:0 (Graph.fan_out g (Tensor.P a)));
  C.test "same node used twice by one consumer has fan_out 2" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let s = map neg x in
      let r = map2 add s s in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r) ] in
      C.int ~expect:2 (Graph.fan_out g (Tensor.P s)));
  C.test "outputs preserved in order and by name" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let g = Graph.create ~name:"g" ~outputs:[ ("b", Tensor.P (map neg x)); ("a", Tensor.P x) ] in
      C.bool ~expect:true (List.map fst (Graph.outputs g) = [ "b"; "a" ]);
      C.string ~expect:"g" (Graph.name g));
  C.test "two distinct Param nodes with the same name are rejected" (fun () ->
      let x1 = param "x" Dtype.F32 shape and x2 = param "x" Dtype.F32 shape in
      C.raises (fun () -> ignore (Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P (map2 add x1 x2)) ])));
  C.test "duplicate output names are rejected" (fun () ->
      let x = param "x" Dtype.F32 shape in
      C.raises (fun () -> ignore (Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P x); ("r", Tensor.P x) ])));
  C.test "empty outputs are rejected" (fun () ->
      C.raises (fun () -> ignore (Graph.create ~name:"g" ~outputs:[])));
  C.test "map2 with mismatched shapes is rejected at construction" (fun () ->
      let x = param "x" Dtype.F32 shape and y = param "y" Dtype.F32 (Shape.of_dims [ 9 ]) in
      C.raises (fun () -> ignore (map2 add x y)));
  C.test "reshape with different numel is rejected" (fun () ->
      let x = param "x" Dtype.F32 shape in
      C.raises (fun () -> ignore (reshape (Shape.of_dims [ 3; 3 ]) x));
      ignore (reshape (Shape.of_dims [ 2; 4 ]) x));
  C.test "gather output takes the index tensor's shape" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let idx = iota (Shape.of_dims [ 3 ]) in
      C.bool ~expect:true (Shape.equal (Tensor.shape (Tensor.P (gather idx x))) (Shape.of_dims [ 3 ])));
  C.test "reduce output is scalar-shaped" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let s = reduce add ~init:(const Dtype.F32 0.0) x in
      C.int ~expect:1 (Shape.numel (Tensor.shape (Tensor.P s))));
  C.test "to_dot names the graph and every node" (fun () ->
      let x = param "x" Dtype.F32 shape in
      let r = map neg x in
      let g = Graph.create ~name:"mygraph" ~outputs:[ ("r", Tensor.P r) ] in
      let dot = Graph.to_dot g in
      C.contains ~sub:"digraph" dot;
      C.contains ~sub:"mygraph" dot;
      C.contains ~sub:(Uid.to_string (Tensor.uid (Tensor.P x))) dot;
      C.contains ~sub:(Uid.to_string (Tensor.uid (Tensor.P r))) dot;
      C.contains ~sub:"->" dot);
  C.run ()
