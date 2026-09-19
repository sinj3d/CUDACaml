(* T14: last-axis Reduce/Scan, reduce_rows / scan_rows / transpose. No GPU. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let f32 dims l = Value.P (Value.of_list Dtype.F32 (Shape.of_dims dims) l)
let zero = const Dtype.F32 0.0
let one name t = [ (name, Tensor.P t) ]
let last l = List.nth l (List.length l - 1)

let out outputs name =
  match List.assoc name outputs with
  | Value.P v -> (
      match Value.dtype v with
      | Dtype.F32 -> (Value.to_list v, Shape.dims (Value.shape v))
      | _ -> failwith "not f32")

let x23 () = param "x" Dtype.F32 (Shape.of_dims [ 2; 3 ])
let in23 = [ ("x", f32 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]) ]

let rec mentions_block_id_e : K.expr -> bool = function
  | K.Block_id -> true
  | K.Load { index; _ } -> mentions_block_id_e index
  | K.Binop (_, _, a, b) | K.Cmp (_, a, b) | K.Logic (_, a, b) -> mentions_block_id_e a || mentions_block_id_e b
  | K.Unop (_, _, a) | K.Not a | K.Cast (_, a) -> mentions_block_id_e a
  | K.Select (a, b, c) -> mentions_block_id_e a || mentions_block_id_e b || mentions_block_id_e c
  | _ -> false

let rec mentions_block_id : K.stmt list -> bool = function
  | [] -> false
  | K.Let { value; _ } :: r | K.Assign { value; _ } :: r -> mentions_block_id_e value || mentions_block_id r
  | K.Store { index; value; _ } :: r -> mentions_block_id_e index || mentions_block_id_e value || mentions_block_id r
  | K.For { lo; hi; step; body; _ } :: r ->
      mentions_block_id_e lo || mentions_block_id_e hi || mentions_block_id_e step || mentions_block_id body
      || mentions_block_id r
  | K.If { cond; then_; else_ } :: r ->
      mentions_block_id_e cond || mentions_block_id then_ || mentions_block_id else_ || mentions_block_id r
  | _ :: r -> mentions_block_id r

let () =
  C.test "interp: reduce_rows add over [2;3]" (fun () ->
      let g = Graph.create ~name:"g" ~outputs:(one "r" (reduce_rows add ~init:zero (x23 ()))) in
      let vals, dims = out (run g in23) "r" in
      C.floats ~expect:[ 6.; 15. ] vals;
      C.bool ~expect:true (dims = [ 2 ]));
  C.test "interp: whole-tensor reduce of a 2-D tensor is a scalar" (fun () ->
      let g = Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero (x23 ()))) in
      let vals, dims = out (run g in23) "s" in
      C.floats ~expect:[ 21. ] vals;
      C.bool ~expect:true (dims = []));
  C.test "interp: scan_rows add and flat scan" (fun () ->
      let x = x23 () in
      let g =
        Graph.create ~name:"g"
          ~outputs:[ ("rows", Tensor.P (scan_rows add ~init:zero x)); ("flat", Tensor.P (scan add ~init:zero x)) ]
      in
      let o = run g in23 in
      let rows, rd = out o "rows" and flat, fd = out o "flat" in
      C.floats ~expect:[ 1.; 3.; 6.; 4.; 9.; 15. ] rows;
      C.bool ~expect:true (rd = [ 2; 3 ]);
      C.floats ~expect:[ 1.; 3.; 6.; 10.; 15.; 21. ] flat;
      C.bool ~expect:true (fd = [ 2; 3 ]));
  C.test "interp: reduce_rows max" (fun () ->
      let g = Graph.create ~name:"g" ~outputs:(one "m" (reduce_rows max ~init:(const Dtype.F32 (-1e30)) (x23 ()))) in
      C.floats ~expect:[ 3.; 6. ] (fst (out (run g in23) "m")));
  C.test "interp: transpose and column sums" (fun () ->
      let x = x23 () in
      let t = transpose x in
      let g =
        Graph.create ~name:"g"
          ~outputs:[ ("t", Tensor.P t); ("cols", Tensor.P (reduce_rows add ~init:zero (transpose x))) ]
      in
      let o = run g in23 in
      let tv, td = out o "t" in
      C.floats ~expect:[ 1.; 4.; 2.; 5.; 3.; 6. ] tv;
      C.bool ~expect:true (td = [ 3; 2 ]);
      C.floats ~expect:[ 5.; 7.; 9. ] (fst (out o "cols")));
  C.test "rank checks raise Invalid_argument" (fun () ->
      let v = param "v" Dtype.F32 (vec 5) in
      let r3 = param "w" Dtype.F32 (Shape.of_dims [ 2; 2; 2 ]) in
      C.raises (fun () -> ignore (reduce_rows add ~init:zero v));
      C.raises (fun () -> ignore (scan_rows add ~init:zero v));
      C.raises (fun () -> ignore (transpose v));
      C.raises (fun () -> ignore (transpose r3)));
  C.test "lower: reduce_rows launches one block per row" (fun () ->
      let prog = Lower.program (Graph.create ~name:"g" ~outputs:(one "r" (reduce_rows add ~init:zero (x23 ())))) in
      C.int ~expect:1 (List.length prog.kernels);
      let k = List.hd prog.kernels in
      C.int ~expect:2 k.launch.grid;
      C.int ~expect:Schedule.block_size k.launch.block;
      C.int ~expect:2 (last k.params).numel;
      C.bool ~expect:true (mentions_block_id k.body));
  C.test "lower: scan_rows launches one block per row" (fun () ->
      let x = param "x" Dtype.F32 (Shape.of_dims [ 4; 10 ]) in
      let prog = Lower.program (Graph.create ~name:"g" ~outputs:(one "p" (scan_rows add ~init:zero x))) in
      let k = last prog.kernels in
      C.int ~expect:4 k.launch.grid;
      C.bool ~expect:true (mentions_block_id k.body));
  C.test "lower: rank-1 reduce is unchanged (grid 1)" (fun () ->
      let x = param "x" Dtype.F32 (vec 1000) in
      let prog = Lower.program (Graph.create ~name:"g" ~outputs:(one "s" (reduce add ~init:zero x))) in
      C.int ~expect:1 (List.length prog.kernels);
      C.int ~expect:1 (List.hd prog.kernels).launch.grid);
  C.test "emit: rows reduce mentions blockIdx.x" (fun () ->
      let src = Backend_cuda.source (Graph.create ~name:"g" ~outputs:(one "r" (reduce_rows add ~init:zero (x23 ())))) in
      C.contains ~sub:"blockIdx.x" src;
      C.contains ~sub:"extern \"C\"" src);
  C.test "fusion: transpose is inlined into a rows reduce" (fun () ->
      let x = param "x" Dtype.F32 (Shape.of_dims [ 8; 16 ]) in
      let t = transpose x in
      let r = reduce_rows add ~init:zero (map (fun e -> mul e e) t) in
      let g = Graph.create ~name:"g" ~outputs:(one "r" r) in
      C.bool ~expect:false (Fusion.is_materialized (Fusion.plan g) (Tensor.P t));
      C.int ~expect:1 (List.length (Lower.program g).kernels));
  C.run ()
