(* Broadcast node, Dsl.broadcast / full / scalar. No GPU. *)
open Cudacaml
open Dsl
module C = Cudacaml_testlib.Check
module K = Kernel_ir

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let f32 l = Value.P (Value.of_list Dtype.F32 (vec (List.length l)) l)
let f32_scalar v = Value.P (Value.of_list Dtype.F32 Shape.scalar [ v ])

let floats outputs name : float list =
  match List.assoc name outputs with
  | Value.P v -> (
      match Value.dtype v with
      | Dtype.F32 -> Value.to_list v
      | Dtype.F64 -> Value.to_list v
      | _ -> failwith "not a float tensor")

let one name t = [ (name, Tensor.P t) ]
let last l = List.nth l (List.length l - 1)
let names (bs : K.buffer list) = List.map (fun (b : K.buffer) -> b.name) bs

(* (buffer name, index expression) of every Load, walking with wildcards so
   later Kernel_ir additions do not break this file. *)
let rec loads_e acc : K.expr -> (string * K.expr) list = function
  | K.Load { buf; index } -> loads_e ((buf.name, index) :: acc) index
  | K.Binop (_, _, a, b) | K.Cmp (_, a, b) | K.Logic (_, a, b) -> loads_e (loads_e acc a) b
  | K.Unop (_, _, a) | K.Not a | K.Cast (_, a) -> loads_e acc a
  | K.Select (a, b, c) -> loads_e (loads_e (loads_e acc a) b) c
  | _ -> acc

let rec loads_s acc : K.stmt list -> (string * K.expr) list = function
  | [] -> acc
  | K.Let { value; _ } :: r | K.Assign { value; _ } :: r -> loads_s (loads_e acc value) r
  | K.Store { index; value; _ } :: r -> loads_s (loads_e (loads_e acc index) value) r
  | K.For { lo; hi; step; body; _ } :: r -> loads_s (loads_s (loads_e (loads_e (loads_e acc lo) hi) step) body) r
  | K.If { cond; then_; else_ } :: r -> loads_s (loads_s (loads_s (loads_e acc cond) then_) else_) r
  | _ :: r -> loads_s acc r

let is_zero_lit = function K.Lit (K.I 0L, _) -> true | _ -> false

let () =
  C.test "interp: broadcast a scalar param" (fun () ->
      let s = scalar "s" Dtype.F32 in
      let g = Graph.create ~name:"b" ~outputs:(one "r" (broadcast (vec 4) s)) in
      C.floats ~expect:[ 2.5; 2.5; 2.5; 2.5 ] (floats (run g [ ("s", f32_scalar 2.5) ]) "r"));
  C.test "interp: x * broadcast (sum x)" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) in
      let total = reduce add ~init:(const Dtype.F32 0.0) x in
      let r = map2 mul x (broadcast (Tensor.shape (Tensor.P x)) total) in
      let g = Graph.create ~name:"b" ~outputs:(one "r" r) in
      C.floats ~expect:[ 6.; 12.; 18. ] (floats (run g [ ("x", f32 [ 1.; 2.; 3. ]) ]) "r"));
  C.test "broadcast rejects a source with numel <> 1" (fun () ->
      let x = param "x" Dtype.F32 (vec 3) in
      C.raises (fun () -> ignore (broadcast (vec 6) x)));
  C.test "full: constant tensor with no params" (fun () ->
      let g = Graph.create ~name:"f" ~outputs:(one "r" (full Dtype.F32 (vec 3) 7.0)) in
      C.int ~expect:0 (List.length (Graph.params g));
      C.floats ~expect:[ 7.; 7.; 7. ] (floats (run g []) "r");
      let prog = Lower.program g in
      C.int ~expect:0 (List.length (List.filter (function K.Upload _ -> true | _ -> false) prog.plan)));
  C.test "scalar is a Shape.scalar param" (fun () ->
      let s = scalar "s" Dtype.F64 in
      C.bool ~expect:true (Shape.equal Shape.scalar (Tensor.shape (Tensor.P s)));
      C.int ~expect:1 (Shape.numel (Tensor.shape (Tensor.P s))));
  C.test "fusion: broadcast with fan-out 1 is inlined" (fun () ->
      let x = param "x" Dtype.F32 (vec 8) in
      let s = scalar "s" Dtype.F32 in
      let b = broadcast (vec 8) s in
      let r = map2 mul x b in
      let g = Graph.create ~name:"b" ~outputs:(one "r" r) in
      let plan = Fusion.plan g in
      C.bool ~expect:false (Fusion.is_materialized plan (Tensor.P b));
      C.int ~expect:1 (List.length (Fusion.kernel_roots plan));
      C.int ~expect:1 (List.length (Lower.program g).kernels));
  C.test "lower: a broadcast reduce result is loaded at literal index 0" (fun () ->
      let x = param "x" Dtype.F32 (vec 100) in
      let total = reduce add ~init:(const Dtype.F32 0.0) x in
      let r = map2 mul x (broadcast (vec 100) total) in
      let prog = Lower.program (Graph.create ~name:"b" ~outputs:(one "r" r)) in
      C.int ~expect:2 (List.length prog.kernels);
      let reduce_k = List.nth prog.kernels 0 and map_k = List.nth prog.kernels 1 in
      let total_buf = (last reduce_k.params).name in
      let loads = List.filter (fun (n, _) -> n = total_buf) (loads_s [] map_k.body) in
      C.bool ~expect:true (loads <> []);
      C.bool ~expect:true (List.for_all (fun (_, idx) -> is_zero_lit idx) loads));
  C.test "lower: broadcasting a param gives params [p_s; out] and a grid-stride loop" (fun () ->
      let s = scalar "s" Dtype.F32 in
      let prog = Lower.program (Graph.create ~name:"b" ~outputs:(one "r" (broadcast (vec 4) s))) in
      let k = List.hd prog.kernels in
      C.bool ~expect:true (names k.params = [ "p_s"; (last k.params).name ]);
      C.int ~expect:4 (last k.params).numel;
      (match k.body with
      | [ K.For { lo = K.Global_thread_id; hi = K.Lit (K.I 4L, _); step = K.Global_size; body = [ K.Store _ ]; _ } ] -> ()
      | _ -> C.fail "expected one grid-stride loop over 4 elements with one Store"));
  C.test "to_dot names the Broadcast node" (fun () ->
      let s = scalar "s" Dtype.F32 in
      let g = Graph.create ~name:"b" ~outputs:(one "r" (broadcast (vec 4) s)) in
      C.contains ~sub:"Broadcast" (Graph.to_dot g));
  C.test "deps of a Broadcast is exactly its source" (fun () ->
      let s = scalar "s" Dtype.F32 in
      let b = broadcast (vec 4) s in
      match Tensor.deps (Tensor.P b) with
      | [ d ] -> C.bool ~expect:true (Uid.equal (Tensor.uid d) (Tensor.uid (Tensor.P s)))
      | _ -> C.fail "expected one dependency");
  C.run ()
