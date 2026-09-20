(* Matmul node, tiled 2-D lowering, adjoint. No GPU. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check
module K = Kernel_ir

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let f64 dims l = Value.P (Value.of_list Dtype.F64 (Shape.of_dims dims) l)
let p name dims = param name Dtype.F64 (Shape.of_dims dims)
let one name t = [ (name, Tensor.P t) ]
let zero = const Dtype.F64 0.0
let last l = List.nth l (List.length l - 1)
let names (bs : K.buffer list) = List.map (fun (b : K.buffer) -> b.name) bs

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.F64 -> Value.to_list v | _ -> failwith "not f64")

let dims_of o name = match List.assoc name o with Value.P v -> Shape.dims (Value.shape v)

let ref_matmul ~m ~k ~n a b =
  List.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let acc = ref 0.0 in
      for q = 0 to k - 1 do
        acc := !acc +. (List.nth a ((i * k) + q) *. List.nth b ((q * n) + j))
      done;
      !acc)

let matmul_interp ~m ~k ~n a b =
  let g = Graph.create ~name:"mm" ~outputs:(one "c" (matmul (p "a" [ m; k ]) (p "b" [ k; n ]))) in
  let o = run g [ ("a", f64 [ m; k ] a); ("b", f64 [ k; n ] b) ] in
  (floats o "c", dims_of o "c")

let rec mentions_e pred : K.expr -> bool = function
  | e when pred e -> true
  | K.Load { index; _ } -> mentions_e pred index
  | K.Binop (_, _, a, b) | K.Cmp (_, a, b) | K.Logic (_, a, b) -> mentions_e pred a || mentions_e pred b
  | K.Unop (_, _, a) | K.Not a | K.Cast (_, a) -> mentions_e pred a
  | K.Select (a, b, c) -> mentions_e pred a || mentions_e pred b || mentions_e pred c
  | _ -> false

let rec mentions pred : K.stmt list -> bool = function
  | [] -> false
  | K.Let { value; _ } :: r | K.Assign { value; _ } :: r -> mentions_e pred value || mentions pred r
  | K.Store { index; value; _ } :: r -> mentions_e pred index || mentions_e pred value || mentions pred r
  | K.For { lo; hi; step; body; _ } :: r ->
      mentions_e pred lo || mentions_e pred hi || mentions_e pred step || mentions pred body || mentions pred r
  | K.If { cond; then_; else_ } :: r -> mentions_e pred cond || mentions pred then_ || mentions pred else_ || mentions pred r
  | _ :: r -> mentions pred r

let rec syncs : K.stmt list -> int = function
  | [] -> 0
  | K.Sync_threads :: r -> 1 + syncs r
  | K.For { body; _ } :: r -> syncs body + syncs r
  | K.If { then_; else_; _ } :: r -> syncs then_ + syncs else_ + syncs r
  | _ :: r -> syncs r

let count_sub ~sub s =
  let n = String.length sub and m = String.length s in
  let rec go i acc = if i + n > m then acc else go (i + 1) (if String.sub s i n = sub then acc + 1 else acc) in
  go 0 0

(* bump-and-revalue, as in test_grad *)
let fd g ~output ~wrt inputs =
  let h = 1e-6 in
  let base : float Value.t =
    match List.assoc wrt inputs with Value.P v -> ( match Value.dtype v with Dtype.F64 -> v | _ -> failwith "f64")
  in
  let n = Value.numel base in
  let sum l = List.fold_left ( +. ) 0.0 l in
  List.init n (fun i ->
      let bump d =
        let v = Value.create Dtype.F64 (Value.shape base) in
        for j = 0 to n - 1 do Value.set v j (Value.get base j) done;
        Value.set v i (Value.get base i +. d);
        sum (floats (run g (List.map (fun (nm, x) -> if nm = wrt then (nm, Value.P v) else (nm, x)) inputs)) output)
      in
      (bump h -. bump (-.h)) /. (2.0 *. h))

let check_grad g inputs =
  let gg = Grad.grad g ~output:"s" ~wrt:[ "a"; "b" ] in
  let o = run gg inputs in
  List.iter
    (fun w ->
      let e = fd g ~output:"s" ~wrt:w inputs and got = floats o (Grad.grad_name ~output:"s" ~wrt:w) in
      List.iteri (fun i (e, g) -> if Float.abs (e -. g) > 1e-5 *. (1.0 +. Float.abs e) then C.fail "%s[%d]: %g vs %g" w i e g) (List.combine e got))
    [ "a"; "b" ]

let () =
  C.test "interp: 2x3 . 3x2" (fun () ->
      let c, d = matmul_interp ~m:2 ~k:3 ~n:2 [ 1.; 2.; 3.; 4.; 5.; 6. ] [ 7.; 8.; 9.; 10.; 11.; 12. ] in
      C.floats ~expect:[ 58.; 64.; 139.; 154. ] c;
      C.bool ~expect:true (d = [ 2; 2 ]));
  C.test "interp: identity" (fun () ->
      let a = [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. ] and id = [ 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. ] in
      C.floats ~expect:a (fst (matmul_interp ~m:3 ~k:3 ~n:3 a id)));
  C.test "interp: 5x7 . 7x3 against a reference" (fun () ->
      let a = List.init 35 (fun i -> float_of_int ((i * 7) mod 11) -. 5.0) and b = List.init 21 (fun i -> float_of_int ((i * 5) mod 13) /. 4.0) in
      C.floats ~tol:1e-12 ~expect:(ref_matmul ~m:5 ~k:7 ~n:3 a b) (fst (matmul_interp ~m:5 ~k:7 ~n:3 a b)));
  C.test "shape checks" (fun () ->
      C.raises (fun () -> ignore (matmul (p "a" [ 2; 3 ]) (p "b" [ 2; 3 ])));
      C.raises (fun () -> ignore (matmul (p "a" [ 6 ]) (p "b" [ 6 ])));
      C.raises (fun () -> ignore (matmul (p "a" [ 2; 2; 2 ]) (p "b" [ 2; 2 ]))));
  C.test "fusion: matmul is a barrier and fuses its operand maps" (fun () ->
      let a = p "a" [ 20; 30 ] and b = p "b" [ 30; 10 ] in
      let c = matmul (map exp a) b in
      let g = Graph.create ~name:"mm" ~outputs:(one "c" c) in
      C.bool ~expect:true (Fusion.is_materialized (Fusion.plan g) (Tensor.P c));
      let prog = Lower.program g in
      C.int ~expect:1 (List.length prog.kernels);
      C.bool ~expect:true (names (List.hd prog.kernels).params = [ "p_a"; "p_b"; (last (List.hd prog.kernels).params).name ]));
  C.test "lower: tiled 2-D launch" (fun () ->
      let prog = Lower.program (Graph.create ~name:"mm" ~outputs:(one "c" (matmul (p "a" [ 37; 50 ]) (p "b" [ 50; 20 ])))) in
      let k = List.hd prog.kernels in
      let t = Schedule.tile in
      C.int ~expect:16 t;
      C.int ~expect:((20 + t - 1) / t) k.launch.grid;
      C.int ~expect:((37 + t - 1) / t) k.launch.grid_y;
      C.int ~expect:t k.launch.block;
      C.int ~expect:t k.launch.block_y;
      C.int ~expect:2 (List.length k.shared);
      List.iter (fun (b : K.buffer) -> C.int ~expect:(t * t) b.numel; C.bool ~expect:true (b.memspace = K.Shared)) k.shared;
      C.bool ~expect:true (mentions (function K.Block_id_y -> true | _ -> false) k.body);
      C.bool ~expect:true (mentions (function K.Local_thread_id_y -> true | _ -> false) k.body);
      C.bool ~expect:true (syncs k.body >= 2);
      C.int ~expect:(37 * 20) (last k.params).numel);
  C.test "1-D launches keep grid_y = block_y = 1" (fun () ->
      let l = Schedule.grid_stride ~numel:1000 in
      C.int ~expect:1 l.grid_y;
      C.int ~expect:1 l.block_y;
      let r = Schedule.rows_block ~rows:3 in
      C.int ~expect:1 r.grid_y;
      C.int ~expect:1 r.block_y);
  C.test "emit: blockIdx.y, threadIdx.y, two shared tiles" (fun () ->
      let src = Backend_cuda.source (Graph.create ~name:"mm" ~outputs:(one "c" (matmul (p "a" [ 4; 4 ]) (p "b" [ 4; 4 ])))) in
      C.contains ~sub:"blockIdx.y" src;
      C.contains ~sub:"threadIdx.y" src;
      C.int ~expect:2 (count_sub ~sub:"__shared__" src));
  C.test "grad: sum (A . B) and sum (exp A . B)" (fun () ->
      let a = p "a" [ 2; 3 ] and b = p "b" [ 3; 2 ] in
      let inputs = [ ("a", f64 [ 2; 3 ] [ 1.; -2.; 0.5; 3.; 1.5; -1. ]); ("b", f64 [ 3; 2 ] [ 0.3; 1.5; -1.; 2.; 0.7; -0.2 ]) ] in
      check_grad (Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (matmul a b)))) inputs;
      check_grad (Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (matmul (map exp a) b)))) inputs);
  C.run ()
