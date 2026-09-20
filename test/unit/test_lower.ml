(* Schedule + Lower. Structural checks on Kernel_ir; no GPU. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir

let vec n = Shape.of_dims [ n ]
let example name = Option.get (Ocaml_cuda_examples.Programs.find name)
let saxpy () = Lower.program (Ocaml_cuda_examples.Saxpy.program ~n:1000 ~a:2.0)

let rec loads_in_expr acc : K.expr -> string list = function
  | K.Load { buf; index } -> loads_in_expr (buf.name :: acc) index
  | K.Binop (_, _, a, b) | K.Cmp (_, a, b) | K.Logic (_, a, b) -> loads_in_expr (loads_in_expr acc a) b
  | K.Unop (_, _, a) | K.Not a | K.Cast (_, a) -> loads_in_expr acc a
  | K.Select (a, b, c) -> loads_in_expr (loads_in_expr (loads_in_expr acc a) b) c
  | K.Var _ | K.Lit _ | K.Global_thread_id | K.Global_size | K.Local_thread_id
  | K.Local_thread_id_y | K.Block_id | K.Block_id_y | K.Block_dim ->
      acc

let rec loads_in_stmts acc = function
  | [] -> acc
  | K.Let { value; _ } :: rest | K.Assign { value; _ } :: rest -> loads_in_stmts (loads_in_expr acc value) rest
  | K.Store { buf; index; value } :: rest | K.Atomic_add { buf; index; value } :: rest ->
      loads_in_stmts (loads_in_expr (loads_in_expr (buf.name :: acc) index) value) rest
  | K.For { lo; hi; step; body; _ } :: rest ->
      let acc = loads_in_expr (loads_in_expr (loads_in_expr acc lo) hi) step in
      loads_in_stmts (loads_in_stmts acc body) rest
  | K.If { cond; then_; else_ } :: rest ->
      loads_in_stmts (loads_in_stmts (loads_in_stmts (loads_in_expr acc cond) then_) else_) rest
  | K.Sync_threads :: rest -> loads_in_stmts acc rest

let count f l = List.length (List.filter f l)
let is_alloc = function K.Alloc _ -> true | _ -> false
let is_upload = function K.Upload _ -> true | _ -> false
let is_launch = function K.Launch _ -> true | _ -> false
let is_download = function K.Download _ -> true | _ -> false
let is_free = function K.Free _ -> true | _ -> false

let index_of f l =
  let rec go i = function [] -> -1 | x :: xs -> if f x then i else go (i + 1) xs in
  go 0 l

let last_index_of f l = List.length l - 1 - index_of f (List.rev l)
let names (bs : K.buffer list) = List.map (fun (b : K.buffer) -> b.name) bs
let last l = List.nth l (List.length l - 1)

let () =
  C.test "grid_stride geometry" (fun () ->
      let l = Schedule.grid_stride ~numel:1000 in
      C.int ~expect:256 l.block;
      C.int ~expect:4 l.grid;
      C.int ~expect:0 l.shared_bytes;
      C.int ~expect:1 (Schedule.grid_stride ~numel:0).grid;
      C.int ~expect:1 (Schedule.grid_stride ~numel:1).grid;
      C.int ~expect:1 (Schedule.grid_stride ~numel:256).grid;
      C.int ~expect:2 (Schedule.grid_stride ~numel:257).grid;
      C.int ~expect:1024 (Schedule.grid_stride ~numel:10_000_000).grid);
  C.test "saxpy: one fused map kernel + one reduce kernel, in that order" (fun () ->
      let prog = saxpy () in
      C.int ~expect:2 (List.length prog.kernels);
      C.string ~expect:"saxpy" prog.name;
      let k = List.nth prog.kernels 0 in
      C.int ~expect:3 (List.length k.params);
      C.bool ~expect:true (List.filteri (fun i _ -> i < 2) (names k.params) = [ "p_x"; "p_y" ]);
      C.bool ~expect:true (List.for_all (fun (b : K.buffer) -> b.memspace = K.Global) k.params);
      C.int ~expect:1000 (last k.params).numel;
      C.int ~expect:0 (List.length k.shared);
      (match k.body with
      | [ K.For { lo = K.Global_thread_id; step = K.Global_size; body = [ K.Store _ ]; _ } ] -> ()
      | _ -> C.fail "map kernel body must be exactly one grid-stride For containing one Store");
      C.int ~expect:4 k.launch.grid;
      C.int ~expect:256 k.launch.block;
      (* only the two params and the output are touched: 2*x was inlined *)
      let touched = List.sort_uniq compare (loads_in_stmts [] k.body) in
      C.bool ~expect:true (touched = List.sort_uniq compare (names k.params)));
  C.test "reduce kernel shape" (fun () ->
      let prog = saxpy () in
      let k = List.nth prog.kernels 1 in
      C.int ~expect:1 k.launch.grid;
      C.int ~expect:Schedule.block_size k.launch.block;
      C.int ~expect:1 (List.length k.shared);
      C.int ~expect:Schedule.block_size (List.hd k.shared).numel;
      C.bool ~expect:true ((List.hd k.shared).memspace = K.Shared);
      C.int ~expect:1 (last k.params).numel;
      C.bool ~expect:true (List.exists (fun s -> s = K.Sync_threads) k.body);
      (* the reduce reads the fused output buffer, not p_x / p_y *)
      let map_out = (last (List.nth prog.kernels 0).params).name in
      C.bool ~expect:true (List.mem map_out (names k.params)));
  C.test "host plan ordering" (fun () ->
      let plan = (saxpy ()).plan in
      C.int ~expect:4 (count is_alloc plan);
      C.int ~expect:2 (count is_upload plan);
      C.int ~expect:2 (count is_launch plan);
      C.int ~expect:2 (count is_download plan);
      C.int ~expect:4 (count is_free plan);
      C.bool ~expect:true (last_index_of is_alloc plan < index_of is_launch plan);
      C.bool ~expect:true (last_index_of is_upload plan < index_of is_launch plan);
      C.bool ~expect:true (last_index_of is_launch plan < index_of is_download plan);
      C.bool ~expect:true (last_index_of is_download plan < index_of is_free plan);
      C.bool ~expect:true
        (List.filter_map (function K.Download { output; _ } -> Some output | _ -> None) plan = [ "r"; "s" ]);
      C.bool ~expect:true
        (List.filter_map (function K.Upload { param; _ } -> Some param | _ -> None) plan = [ "x"; "y" ]);
      C.bool ~expect:true
        (List.for_all
           (function K.Download { from; shape; _ } -> from.numel = Shape.numel shape | _ -> true)
           plan));
  C.test "no-input program: iota+cast inlined, kernel has only the output param" (fun () ->
      let r = map (fun i -> let f = cast Dtype.F32 i in mul f f) (iota (vec 64)) in
      let prog = Lower.program (Graph.create ~name:"sq" ~outputs:[ ("r", Tensor.P r) ]) in
      C.int ~expect:1 (List.length prog.kernels);
      C.int ~expect:1 (List.length (List.hd prog.kernels).params);
      C.int ~expect:0 (count is_upload prog.plan);
      C.int ~expect:1 (count is_alloc prog.plan));
  C.test "chain of five maps is one kernel" (fun () ->
      C.int ~expect:1 (List.length (Lower.program ((example "chain").graph ())).kernels));
  C.test "fanout: shared intermediate is one buffer read by three kernels" (fun () ->
      let prog = Lower.program ((example "fanout").graph ()) in
      C.int ~expect:4 (List.length prog.kernels);
      C.int ~expect:5 (count is_alloc prog.plan));
  (* A row that fits in one chunk is still a single kernel and a single
     block, but that block is now a whole [Schedule.scan_chunk] of threads
     running a Hillis-Steele scan instead of one sequential thread. *)
  C.test "scan kernel runs on one block" (fun () ->
      let x = param "x" Dtype.F32 (vec 10) in
      let g = Graph.create ~name:"g" ~outputs:[ ("p", Tensor.P (scan add ~init:(const Dtype.F32 0.0) x)) ] in
      let k = List.hd (Lower.program g).kernels in
      C.int ~expect:1 k.launch.grid;
      C.int ~expect:Schedule.scan_chunk k.launch.block);
  C.test "output that is a Param: no kernel, download straight from the param buffer" (fun () ->
      let x = param "x" Dtype.F32 (vec 4) in
      let prog = Lower.program (Graph.create ~name:"id" ~outputs:[ ("x", Tensor.P x) ]) in
      C.int ~expect:0 (List.length prog.kernels);
      C.int ~expect:1 (count is_download prog.plan));
  C.test "kernel names are unique, C identifiers, and match Launch ops in order" (fun () ->
      let prog = Lower.program ((example "fanout").graph ()) in
      let ks = List.map (fun (k : K.kernel) -> k.name) prog.kernels in
      C.int ~expect:(List.length ks) (List.length (List.sort_uniq compare ks));
      List.iter
        (fun n ->
          String.iter
            (fun c ->
              if not (c = '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) then
                C.fail "bad kernel name %s" n)
            n)
        ks;
      let launched = List.filter_map (function K.Launch { kernel; _ } -> Some kernel | _ -> None) prog.plan in
      C.bool ~expect:true (launched = ks));
  C.test "every Launch arg was allocated earlier in the plan" (fun () ->
      let prog = Lower.program ((example "fanout").graph ()) in
      let seen = Hashtbl.create 8 in
      List.iter
        (function
          | K.Alloc b -> Hashtbl.replace seen b.name ()
          | K.Launch { args; _ } ->
              List.iter (fun (b : K.buffer) -> if not (Hashtbl.mem seen b.name) then C.fail "%s used before Alloc" b.name) args
          | _ -> ())
        prog.plan);
  C.run ()
