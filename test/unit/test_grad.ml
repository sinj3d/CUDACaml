(* T18: Grad.grad against bump-and-revalue on the interpreter, F64.
   T19 replaces the one assertion marked "T18 only" (see T19's spec). *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let f64 dims l = Value.P (Value.of_list Dtype.F64 (Shape.of_dims dims) l)
let k v = const Dtype.F64 v
let zero = k 0.0
let p name dims = param name Dtype.F64 (Shape.of_dims dims)

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.F64 -> Value.to_list v | _ -> failwith "not f64")

let dims_of o name = match List.assoc name o with Value.P v -> Shape.dims (Value.shape v)
let sum l = List.fold_left ( +. ) 0.0 l

(* Central difference of (sum of [output]) with respect to every element of
   input [wrt], by rerunning the ORIGINAL graph. *)
let fd g ~output ~wrt inputs =
  let h = 1e-6 in
  let base : float Value.t =
    match List.assoc wrt inputs with Value.P v -> ( match Value.dtype v with Dtype.F64 -> v | _ -> failwith "wrt not f64")
  in
  let n = Value.numel base in
  List.init n (fun i ->
      let bump delta =
        let v = Value.create Dtype.F64 (Value.shape base) in
        for j = 0 to n - 1 do Value.set v j (Value.get base j) done;
        Value.set v i (Value.get base i +. delta);
        let inputs = List.map (fun (nm, x) -> if nm = wrt then (nm, Value.P v) else (nm, x)) inputs in
        sum (floats (run g inputs) output)
      in
      (bump h -. bump (-.h)) /. (2.0 *. h))

let close ?(tol = 1e-5) ~what expect got =
  if List.length expect <> List.length got then C.fail "%s: length %d vs %d" what (List.length expect) (List.length got);
  List.iteri
    (fun i (e, g) -> if Float.abs (e -. g) > tol *. (1.0 +. Float.abs e) then C.fail "%s[%d]: fd %g, grad %g" what i e g)
    (List.combine expect got)

(* grad graph run once; gradient of [output] wrt [wrt] as floats *)
let grad_vs_fd ?tol g ~output ~wrt inputs =
  let gg = Grad.grad g ~output ~wrt in
  let o = run gg inputs in
  List.iter (fun w -> close ?tol ~what:w (fd g ~output ~wrt:w inputs) (floats o (Grad.grad_name ~output ~wrt:w))) wrt;
  o

let one name t = [ (name, Tensor.P t) ]
let xs4 = [ 1.0; -2.0; 0.5; 3.0 ]
let ys4 = [ 0.3; 1.5; -1.0; 2.0 ]

let () =
  C.test "sum x^2 -> 2x" (fun () ->
      let x = p "x" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map (fun e -> mul e e) x))) in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] [ ("x", f64 [ 4 ] xs4) ] in
      C.floats ~tol:1e-9 ~expect:(List.map (fun v -> 2.0 *. v) xs4) (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "sum x*y -> y, x" (fun () ->
      let x = p "x" [ 4 ] and y = p "y" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map2 mul x y))) in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x"; "y" ] [ ("x", f64 [ 4 ] xs4); ("y", f64 [ 4 ] ys4) ] in
      C.floats ~tol:1e-9 ~expect:ys4 (floats o (Grad.grad_name ~output:"s" ~wrt:"x"));
      C.floats ~tol:1e-9 ~expect:xs4 (floats o (Grad.grad_name ~output:"s" ~wrt:"y")));
  C.test "chain: sum (exp x * y + sin (x*y))" (fun () ->
      let x = p "x" [ 4 ] and y = p "y" [ 4 ] in
      let r = map2 (fun a b -> add (mul (exp a) b) (sin (mul a b))) x y in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero r)) in
      ignore (grad_vs_fd g ~output:"s" ~wrt:[ "x"; "y" ] [ ("x", f64 [ 4 ] xs4); ("y", f64 [ 4 ] ys4) ]));
  C.test "reduce max -> indicator of the argmax" (fun () ->
      let x = p "x" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "m" (reduce max ~init:(k (-1e300)) x)) in
      let o = grad_vs_fd g ~output:"m" ~wrt:[ "x" ] [ ("x", f64 [ 4 ] [ 1.; 5.; 3.; 2. ]) ] in
      C.floats ~tol:1e-12 ~expect:[ 0.; 1.; 0.; 0. ] (floats o (Grad.grad_name ~output:"m" ~wrt:"x")));
  C.test "reduce_rows add then weighted sum -> weights per row" (fun () ->
      let x = p "x" [ 2; 3 ] and w = p "w" [ 2 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map2 mul w (reduce_rows add ~init:zero x)))) in
      let inputs = [ ("x", f64 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]); ("w", f64 [ 2 ] [ 10.; 20. ]) ] in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x"; "w" ] inputs in
      C.floats ~tol:1e-9 ~expect:[ 10.; 10.; 10.; 20.; 20.; 20. ] (floats o (Grad.grad_name ~output:"s" ~wrt:"x"));
      C.bool ~expect:true (dims_of o (Grad.grad_name ~output:"s" ~wrt:"x") = [ 2; 3 ]));
  C.test "scan add: sum (w * cumsum x) -> reverse cumulative sums of w" (fun () ->
      let x = p "x" [ 4 ] and w = p "w" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map2 mul w (scan add ~init:zero x)))) in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] [ ("x", f64 [ 4 ] xs4); ("w", f64 [ 4 ] ys4) ] in
      C.floats ~tol:1e-9 ~expect:[ 2.8; 2.5; 1.0; 2.0 ] (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "scan_rows add over [2;3]" (fun () ->
      let x = p "x" [ 2; 3 ] and w = p "w" [ 2; 3 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map2 mul w (scan_rows add ~init:zero x)))) in
      let inputs = [ ("x", f64 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]); ("w", f64 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) ] in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] inputs in
      C.floats ~tol:1e-9 ~expect:[ 6.; 5.; 3.; 15.; 11.; 6. ] (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "broadcast: sum (x * broadcast s) -> ds = sum x, shape []" (fun () ->
      let x = p "x" [ 4 ] and s = scalar "s" Dtype.F64 in
      let g = Graph.create ~name:"g" ~outputs:(one "t" (reduce add ~init:zero (map2 mul x (broadcast (vec 4) s)))) in
      let inputs = [ ("x", f64 [ 4 ] xs4); ("s", f64 [] [ 1.7 ]) ] in
      let o = grad_vs_fd g ~output:"t" ~wrt:[ "s"; "x" ] inputs in
      C.floats ~tol:1e-9 ~expect:[ sum xs4 ] (floats o (Grad.grad_name ~output:"t" ~wrt:"s"));
      C.bool ~expect:true (dims_of o (Grad.grad_name ~output:"t" ~wrt:"s") = []));
  C.test "reshape passes the adjoint through with the param's shape" (fun () ->
      let x = p "x" [ 2; 3 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map (fun e -> mul e e) (reshape (vec 6) x)))) in
      let vals = [ 1.; 2.; 3.; 4.; 5.; 6. ] in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] [ ("x", f64 [ 2; 3 ] vals) ] in
      C.bool ~expect:true (dims_of o (Grad.grad_name ~output:"s" ~wrt:"x") = [ 2; 3 ]);
      C.floats ~tol:1e-9 ~expect:(List.map (fun v -> 2.0 *. v) vals) (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "non-scalar output is seeded with ones" (fun () ->
      let x = p "x" [ 4 ] and y = p "y" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "r" (map2 mul x y)) in
      let o = grad_vs_fd g ~output:"r" ~wrt:[ "x" ] [ ("x", f64 [ 4 ] xs4); ("y", f64 [ 4 ] ys4) ] in
      C.floats ~tol:1e-9 ~expect:ys4 (floats o (Grad.grad_name ~output:"r" ~wrt:"x")));
  C.test "unused param gets zeros of its shape" (fun () ->
      let x = p "x" [ 4 ] and y = p "y" [ 2; 2 ] in
      let g =
        Graph.create ~name:"g"
          ~outputs:[ ("s", Tensor.P (reduce add ~init:zero x)); ("y", Tensor.P y) ]
      in
      let o = run (Grad.grad g ~output:"s" ~wrt:[ "y" ]) [ ("x", f64 [ 4 ] xs4); ("y", f64 [ 2; 2 ] ys4) ] in
      C.floats ~tol:0.0 ~expect:[ 0.; 0.; 0.; 0. ] (floats o (Grad.grad_name ~output:"s" ~wrt:"y"));
      C.bool ~expect:true (dims_of o (Grad.grad_name ~output:"s" ~wrt:"y") = [ 2; 2 ]));
  C.test "unrecognised reduce operator is Not_differentiable" (fun () ->
      let x = p "x" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce (fun a b -> add (mul a b) a) ~init:zero x)) in
      match Grad.grad g ~output:"s" ~wrt:[ "x" ] with
      | exception Grad.Not_differentiable m -> C.contains ~sub:"Reduce" m
      | _ -> C.fail "expected Not_differentiable");
  (* T19: the Gather adjoint is a scatter_add, so a repeated index
     accumulates and a source element nobody reads gets exactly zero. *)
  C.test "sum (gather idx x), idx = [0;0;1] -> [2; 1; 0]" (fun () ->
      let x = p "x" [ 3 ] and idx = param "idx" Dtype.I32 (vec 3) in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (gather idx x))) in
      let inputs =
        [ ("x", f64 [ 3 ] [ 1.; 2.; 3. ]);
          ("idx", Value.P (Value.of_list Dtype.I32 (vec 3) [ 0l; 0l; 1l ])) ]
      in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] inputs in
      C.floats ~tol:1e-12 ~expect:[ 2.; 1.; 0. ] (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "reverse (gather by n-1-i) with weights -> the reversed weights" (fun () ->
      let n = 4 in
      let x = p "x" [ n ] and w = p "w" [ n ] in
      let rev = map (fun i -> sub (const Dtype.I32 (Int32.of_int (n - 1))) i) (iota (vec n)) in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (map2 mul w (gather rev x)))) in
      let inputs = [ ("x", f64 [ n ] xs4); ("w", f64 [ n ] ys4) ] in
      let o = grad_vs_fd g ~output:"s" ~wrt:[ "x" ] inputs in
      C.floats ~tol:1e-9 ~expect:(List.rev ys4) (floats o (Grad.grad_name ~output:"s" ~wrt:"x")));
  C.test "original outputs are preserved; names and count" (fun () ->
      let x = p "x" [ 4 ] and y = p "y" [ 4 ] in
      let r = map2 mul x y in
      let g = Graph.create ~name:"g" ~outputs:[ ("r", Tensor.P r); ("s", Tensor.P (reduce add ~init:zero r)) ] in
      let inputs = [ ("x", f64 [ 4 ] xs4); ("y", f64 [ 4 ] ys4) ] in
      let gg = Grad.grad g ~output:"s" ~wrt:[ "x"; "y" ] in
      let o = run g inputs and og = run gg inputs in
      C.int ~expect:4 (List.length og);
      C.bool ~expect:true
        (List.map fst og = [ "r"; "s"; Grad.grad_name ~output:"s" ~wrt:"x"; Grad.grad_name ~output:"s" ~wrt:"y" ]);
      C.floats ~tol:0.0 ~expect:(floats o "r") (floats og "r");
      C.floats ~tol:0.0 ~expect:(floats o "s") (floats og "s");
      C.string ~expect:"ds/dx" (Grad.grad_name ~output:"s" ~wrt:"x"));
  C.test "unknown names raise Invalid_argument" (fun () ->
      let x = p "x" [ 4 ] in
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero x)) in
      C.raises (fun () -> ignore (Grad.grad g ~output:"nope" ~wrt:[ "x" ]));
      C.raises (fun () -> ignore (Grad.grad g ~output:"s" ~wrt:[ "nope" ])));
  C.run ()
