(* T19: Scatter_add node, Atomic_add lowering, zero-fill kernel. No GPU. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let i32 l = Value.P (Value.of_list Dtype.I32 (vec (List.length l)) l)
let f32 l = Value.P (Value.of_list Dtype.F32 (vec (List.length l)) l)
let one name t = [ (name, Tensor.P t) ]
let last l = List.nth l (List.length l - 1)

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.F32 -> Value.to_list v | _ -> failwith "not f32")

let ints o name : int32 list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.I32 -> Value.to_list v | _ -> failwith "not i32")

let is_zero_lit = function K.Lit (K.F 0.0, _) | K.Lit (K.I 0L, _) -> true | _ -> false

let rec has_atomic name : K.stmt list -> bool = function
  | [] -> false
  | K.Atomic_add { buf; _ } :: r -> buf.name = name || has_atomic name r
  | K.For { body; _ } :: r -> has_atomic name body || has_atomic name r
  | K.If { then_; else_; _ } :: r -> has_atomic name then_ || has_atomic name else_ || has_atomic name r
  | _ :: r -> has_atomic name r

let idx6 = [ 0l; 1l; 1l; 3l; 3l; 3l ]

let () =
  C.test "interp: f32 histogram" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 6) in
      let g = Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4))) in
      C.floats ~expect:[ 1.; 2.; 0.; 3. ] (floats (run g [ ("idx", i32 idx6); ("src", f32 (List.init 6 (fun _ -> 1.0))) ]) "h"));
  C.test "interp: i32 histogram is exact" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.I32 (vec 6) in
      let g = Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4))) in
      let got = ints (run g [ ("idx", i32 idx6); ("src", i32 [ 1l; 2l; 3l; 4l; 5l; 6l ]) ]) "h" in
      C.bool ~expect:true (List.for_all2 Int32.equal [ 1l; 5l; 0l; 15l ] got));
  C.test "interp: out-of-range index raises" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 1) and src = param "src" Dtype.F32 (vec 1) in
      let g = Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4))) in
      C.raises (fun () -> ignore (run g [ ("idx", i32 [ 4l ]); ("src", f32 [ 1.0 ]) ])));
  C.test "shape mismatch between idx and src is rejected" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 5) in
      C.raises (fun () -> ignore (scatter_add idx src (vec 4))));
  C.test "fusion: scatter_add is a barrier" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 6) in
      let s = scatter_add idx src (vec 4) in
      let g = Graph.create ~name:"h" ~outputs:(one "r" (map neg s)) in
      C.bool ~expect:true (Fusion.is_materialized (Fusion.plan g) (Tensor.P s));
      C.int ~expect:2 (List.length (Fusion.kernel_roots (Fusion.plan g))));
  C.test "lower: zero-fill kernel then atomic kernel, two launches in order" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 6) in
      let prog = Lower.program (Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4)))) in
      C.int ~expect:2 (List.length prog.kernels);
      let z = List.nth prog.kernels 0 and a = List.nth prog.kernels 1 in
      let out = (last a.params).name in
      C.string ~expect:out (last z.params).name;
      (match z.body with
      | [ K.For { body = [ K.Store { buf; value; _ } ]; _ } ] ->
          C.string ~expect:out buf.name;
          C.bool ~expect:true (is_zero_lit value)
      | _ -> C.fail "zero kernel must be one grid-stride loop with one Store of zero");
      C.bool ~expect:true (has_atomic out a.body);
      let launches = List.filter_map (function K.Launch { kernel; _ } -> Some kernel | _ -> None) prog.plan in
      C.bool ~expect:true (launches = [ z.name; a.name ]));
  C.test "emit: atomicAdd" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 6) in
      C.contains ~sub:"atomicAdd(&" (Backend_cuda.source (Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4))))));
  C.test "to_dot names Scatter_add" (fun () ->
      let idx = param "idx" Dtype.I32 (vec 6) and src = param "src" Dtype.F32 (vec 6) in
      C.contains ~sub:"Scatter_add" (Graph.to_dot (Graph.create ~name:"h" ~outputs:(one "h" (scatter_add idx src (vec 4))))));
  C.run ()
