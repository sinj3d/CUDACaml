(* T04: Fusion analysis. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check

let vec n = Shape.of_dims [ n ]
let mat plan t = Fusion.is_materialized plan (Tensor.P t)
let root_uids plan = List.map (fun p -> Uid.to_int (Tensor.uid p)) (Fusion.kernel_roots plan)
let uid t = Uid.to_int (Tensor.uid (Tensor.P t))

let () =
  C.test "chain of maps: only the last is a root" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let a = map neg x in
      let b = map neg a in
      let c = map neg b in
      let g = Graph.create ~name:"g" ~outputs:[ ("c", Tensor.P c) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p x);
      C.bool ~expect:false (mat p a);
      C.bool ~expect:false (mat p b);
      C.bool ~expect:true (mat p c);
      C.bool ~expect:true (root_uids p = [ uid c ]));
  C.test "params are materialised but never roots" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let g = Graph.create ~name:"g" ~outputs:[ ("x", Tensor.P x) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p x);
      C.bool ~expect:true (root_uids p = []));
  C.test "map feeding a reduce is inlined; reduce is a root" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let a = map neg x in
      let s = reduce add ~init:(const Dtype.F32 0.0) a in
      let g = Graph.create ~name:"g" ~outputs:[ ("s", Tensor.P s) ] in
      let p = Fusion.plan g in
      C.bool ~expect:false (mat p a);
      C.bool ~expect:true (root_uids p = [ uid s ]));
  C.test "fan_out >= 2 forces materialisation, roots in topological order" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let s = map (fun e -> mul e e) x in
      let a = map neg s and b = map sqrt s in
      let g = Graph.create ~name:"g" ~outputs:[ ("a", Tensor.P a); ("b", Tensor.P b) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p s);
      C.bool ~expect:true (root_uids p = [ uid s; uid a; uid b ]));
  C.test "an intermediate that is also an output is materialised" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let r = map neg x in
      let s = reduce add ~init:(const Dtype.F32 0.0) r in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r); ("s", Tensor.P s) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p r);
      C.bool ~expect:true (root_uids p = [ uid r; uid s ]));
  C.test "scan is a barrier; map over it is a separate root" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let p1 = scan add ~init:(const Dtype.F32 0.0) x in
      let r = map neg p1 in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p p1);
      C.bool ~expect:true (root_uids p = [ uid p1; uid r ]));
  C.test "iota, gather and reshape are inlineable" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let idx = map (fun i -> sub (const Dtype.I32 3l) i) (iota (vec 4)) in
      let r = map neg (reshape (Shape.of_dims [ 2; 2 ]) (gather idx x)) in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (root_uids p = [ uid r ]));
  C.test "same node used twice by one consumer is materialised" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let s = map neg x in
      let r = map2 add s s in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r) ] in
      let p = Fusion.plan g in
      C.bool ~expect:true (mat p s);
      C.bool ~expect:true (root_uids p = [ uid s; uid r ]));
  C.run ()
