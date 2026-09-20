(* Backend_interp, the oracle. Every expected value here is computed by
   hand; do NOT replace any of them with a value produced by the code under
   test. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let f32 l = Value.P (Value.of_list Dtype.F32 (vec (List.length l)) l)
let i32 l = Value.P (Value.of_list Dtype.I32 (vec (List.length l)) l)

let floats outputs name : float list =
  match List.assoc name outputs with
  | Value.P v -> (
      match Value.dtype v with
      | Dtype.F32 -> Value.to_list v
      | Dtype.F64 -> Value.to_list v
      | _ -> failwith "not a float tensor")

let ints outputs name : int32 list =
  match List.assoc name outputs with
  | Value.P v -> (
      match Value.dtype v with Dtype.I32 -> Value.to_list v | _ -> failwith "not an i32 tensor")

let one name t = [ (name, Tensor.P t) ]

let () =
  C.test "map" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map (fun e -> mul e e) x)) in
      C.floats ~expect:[ 1.; 4.; 9. ] (floats (run g [ ("x", f32 [ 1.; 2.; 3. ]) ]) "r"));
  C.test "map2 and map fused chain" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) and y = param "y" Dtype.F32 (vec 3) in
      let r = map2 add (map (fun e -> mul (const Dtype.F32 2.0) e) x) y in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      let o = run g [ ("x", f32 [ 1.; 2.; 3. ]); ("y", f32 [ 10.; 20.; 30. ]) ] in
      C.floats ~expect:[ 12.; 24.; 36. ] (floats o "r"));
  C.test "iota and cast" (fun () ->
      let r = map (fun i -> cast Dtype.F32 i) (iota (vec 4)) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      C.floats ~expect:[ 0.; 1.; 2.; 3. ] (floats (run g []) "r"));
  C.test "index inside a map" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) in
      let r = map (fun e -> add e (cast Dtype.F32 (index ()))) x in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      C.floats ~expect:[ 10.; 11.; 12. ] (floats (run g [ ("x", f32 [ 10.; 10.; 10. ]) ]) "r"));
  C.test "reduce sum is a sequential left fold from init" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:(const Dtype.F32 100.0) x)) in
      C.floats ~expect:[ 110. ] (floats (run g [ ("x", f32 [ 1.; 2.; 3.; 4. ]) ]) "s"));
  C.test "reduce max" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) in
      let g = Graph.create ~name:"g" ~outputs:(one "m" (reduce max ~init:(const Dtype.F32 (-1e30)) x)) in
      C.floats ~expect:[ 7. ] (floats (run g [ ("x", f32 [ -3.; 7.; 2. ]) ]) "m"));
  C.test "reduce of empty tensor is init" (fun () ->
      let x = param "x" Dtype.F32 (vec 0) in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:(const Dtype.F32 5.0) x)) in
      C.floats ~expect:[ 5. ] (floats (run g [ ("x", f32 []) ]) "s"));
  C.test "scan is inclusive" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let g = Graph.create ~name:"g" ~outputs:(one "p" (scan add ~init:(const Dtype.F32 0.0) x)) in
      C.floats ~expect:[ 1.; 3.; 6.; 10. ] (floats (run g [ ("x", f32 [ 1.; 2.; 3.; 4. ]) ]) "p"));
  C.test "gather" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) and i = param "i" Dtype.I32 (vec 4) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (gather i x)) in
      let o = run g [ ("x", f32 [ 10.; 20.; 30. ]); ("i", i32 [ 2l; 0l; 2l; 1l ]) ] in
      C.floats ~expect:[ 30.; 10.; 30.; 20. ] (floats o "r"));
  C.test "gather out of bounds raises" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) and i = param "i" Dtype.I32 (vec 1) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (gather i x)) in
      C.raises (fun () -> ignore (run g [ ("x", f32 [ 1.; 2.; 3. ]); ("i", i32 [ 3l ]) ])));
  C.test "reshape keeps data, changes shape" (fun () ->
      let x = param "x" Dtype.F32 (Shape.of_dims [ 2; 2 ]) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (reshape (vec 4) x)) in
      let o = run g [ ("x", Value.P (Value.of_list Dtype.F32 (Shape.of_dims [ 2; 2 ]) [ 1.; 2.; 3.; 4. ])) ] in
      C.floats ~expect:[ 1.; 2.; 3.; 4. ] (floats o "r");
      match List.assoc "r" o with Value.P v -> C.bool ~expect:true (Shape.equal (Value.shape v) (vec 4)));
  C.test "select and comparisons" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let zero = const Dtype.F32 0.0 in
      let r = map (fun e -> select (lt e zero) zero e) x in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      C.floats ~expect:[ 0.; 2.; 0.; 4. ] (floats (run g [ ("x", f32 [ -1.; 2.; -3.; 4. ]) ]) "r"));
  C.test "i32 division truncates toward zero like C" (fun () ->
      let x = param "x" Dtype.I32 (vec 2) in
      let r = map (fun e -> div e (const Dtype.I32 2l)) x in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      C.bool ~expect:true (ints (run g [ ("x", i32 [ -7l; 7l ]) ]) "r" = [ -3l; 3l ]));
  C.test "f32 arithmetic is rounded to single precision" (fun () ->
      let x = param "x" Dtype.F32 (vec 1) in
      let r = map (fun e -> add e (const Dtype.F32 0.2)) x in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      let v = List.hd (floats (run g [ ("x", f32 [ 0.1 ]) ]) "r") in
      C.float ~tol:1e-9 ~expect:0.300000011920929 v);
  C.test "shared intermediate evaluated once, outputs in order" (fun () ->
      let x = param "x" Dtype.F32 (vec 2) in
      let s = map (fun e -> mul e e) x in
      let g = Graph.create ~name:"g"
          ~outputs:[ ("a", Tensor.P (map neg s)); ("c", Tensor.P (reduce add ~init:(const Dtype.F32 0.0) s)) ] in
      let o = run g [ ("x", f32 [ 3.; 4. ]) ] in
      C.bool ~expect:true (List.map fst o = [ "a"; "c" ]);
      C.floats ~expect:[ -9.; -16. ] (floats o "a");
      C.floats ~expect:[ 25. ] (floats o "c"));
  C.test "missing input raises" (fun () ->
      let x = param "x" Dtype.F32 (vec 1) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map neg x)) in
      C.raises (fun () -> ignore (run g [])));
  C.test "input with wrong shape raises" (fun () ->
      let x = param "x" Dtype.F32 (vec 2) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map neg x)) in
      C.raises (fun () -> ignore (run g [ ("x", f32 [ 1. ]) ])));
  C.test "input with wrong dtype raises" (fun () ->
      let x = param "x" Dtype.F32 (vec 1) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map neg x)) in
      C.raises (fun () -> ignore (run g [ ("x", i32 [ 1l ]) ])));
  C.test "every registered example runs on the interpreter" (fun () ->
      List.iter
        (fun (e : Cudacaml_examples.Programs.t) -> ignore (run (e.graph ()) (e.inputs ())))
        Cudacaml_examples.Programs.all);
  C.run ()
