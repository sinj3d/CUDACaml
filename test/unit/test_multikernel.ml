(* Grid-wide reduce and parallel scan, structural. No GPU. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir
module Programs = Ocaml_cuda_examples.Programs

let vec n = Shape.of_dims [ n ]
let zero = const Dtype.F32 0.0
let one name t = [ (name, Tensor.P t) ]
let last l = List.nth l (List.length l - 1)
let count f l = List.length (List.filter f l)
let is_alloc = function K.Alloc _ -> true | _ -> false
let is_free = function K.Free _ -> true | _ -> false

let rec syncs : K.stmt list -> int = function
  | [] -> 0
  | K.Sync_threads :: r -> 1 + syncs r
  | K.For { body; _ } :: r -> syncs body + syncs r
  | K.If { then_; else_; _ } :: r -> syncs then_ + syncs else_ + syncs r
  | _ :: r -> syncs r

(* Every buffer a Launch/Upload/Download names was allocated before and is
   freed after; nothing is allocated twice or freed twice; nothing leaks. *)
let well_formed (plan : K.host_op list) =
  let live = Hashtbl.create 16 in
  let need (b : K.buffer) = if not (Hashtbl.mem live b.name) then C.fail "buffer %s used before Alloc / after Free" b.name in
  List.iter
    (function
      | K.Alloc b ->
          if Hashtbl.mem live b.name then C.fail "buffer %s allocated twice" b.name;
          Hashtbl.add live b.name ()
      | K.Upload { into; _ } -> need into
      | K.Launch { args; _ } -> List.iter need args
      | K.Download { from; _ } -> need from
      | K.Free b ->
          need b;
          Hashtbl.remove live b.name)
    plan;
  C.int ~expect:0 (Hashtbl.length live)

let reduce_prog n = Lower.program (Graph.create ~name:"r" ~outputs:(one "s" (reduce add ~init:zero (param "x" Dtype.F32 (vec n)))))
let scan_prog n = Lower.program (Graph.create ~name:"p" ~outputs:(one "p" (scan add ~init:zero (param "x" Dtype.F32 (vec n)))))

let () =
  C.test "reduce_blocks" (fun () ->
      C.int ~expect:1 (Schedule.reduce_blocks ~row_len:1000);
      C.int ~expect:1 (Schedule.reduce_blocks ~row_len:Schedule.reduce_threshold);
      C.int ~expect:4 (Schedule.reduce_blocks ~row_len:8192);
      C.int ~expect:128 (Schedule.reduce_blocks ~row_len:1_000_000));
  C.test "small reduce is one kernel (v1)" (fun () ->
      let prog = reduce_prog 1000 in
      C.int ~expect:1 (List.length prog.kernels);
      well_formed prog.plan);
  C.test "large reduce is two kernels with a scratch buffer" (fun () ->
      let x = param "x" Dtype.F32 (vec (1 lsl 20)) in
      let s = reduce add ~init:zero x in
      let prog = Lower.program (Graph.create ~name:"r" ~outputs:(one "s" s)) in
      C.int ~expect:2 (List.length prog.kernels);
      C.int ~expect:3 (count is_alloc prog.plan);
      C.int ~expect:3 (count is_free prog.plan);
      let a = List.nth prog.kernels 0 and b = List.nth prog.kernels 1 in
      C.int ~expect:128 a.launch.grid;
      C.int ~expect:1 b.launch.grid;
      C.string ~expect:("k_" ^ string_of_int (Uid.to_int (Tensor.uid (Tensor.P s)))) b.name;
      C.int ~expect:1 (last b.params).numel;
      well_formed prog.plan);
  C.test "rows reduce multiplies the grid by the row count" (fun () ->
      let x = param "x" Dtype.F32 (Shape.of_dims [ 4; 1 lsl 16 ]) in
      let prog = Lower.program (Graph.create ~name:"r" ~outputs:(one "s" (reduce_rows add ~init:zero x))) in
      C.int ~expect:2 (List.length prog.kernels);
      C.int ~expect:(4 * Schedule.reduce_blocks ~row_len:(1 lsl 16)) (List.nth prog.kernels 0).launch.grid;
      C.int ~expect:4 (List.nth prog.kernels 1).launch.grid;
      well_formed prog.plan);
  C.test "scan within one chunk: one block-scan kernel" (fun () ->
      let prog = scan_prog 100 in
      C.int ~expect:1 (List.length prog.kernels);
      let k = List.hd prog.kernels in
      C.int ~expect:1 k.launch.grid;
      C.int ~expect:Schedule.scan_chunk k.launch.block;
      C.int ~expect:1 (List.length k.shared);
      C.bool ~expect:true ((List.hd k.shared).numel >= Schedule.scan_chunk);
      C.bool ~expect:true (syncs k.body >= 2);
      well_formed prog.plan);
  C.test "scan over many chunks: three kernels" (fun () ->
      let prog = scan_prog 5000 in
      C.int ~expect:3 (List.length prog.kernels);
      C.int ~expect:3 (Schedule.scan_kernels ~row_len:5000);
      C.int ~expect:1 (Schedule.scan_kernels ~row_len:100);
      let a = List.nth prog.kernels 0 and b = List.nth prog.kernels 1 and c = List.nth prog.kernels 2 in
      C.int ~expect:20 a.launch.grid;
      C.int ~expect:1 b.launch.grid;
      C.int ~expect:1 b.launch.block;
      (match c.body with
      | [ K.For { lo = K.Global_thread_id; step = K.Global_size; _ } ] -> ()
      | _ -> C.fail "third scan kernel must be a grid-stride loop");
      C.int ~expect:3 (count is_alloc prog.plan);
      well_formed prog.plan);
  C.test "scan_rows over [3;700]: grid 9 then 3" (fun () ->
      let x = param "x" Dtype.F32 (Shape.of_dims [ 3; 700 ]) in
      let prog = Lower.program (Graph.create ~name:"p" ~outputs:(one "p" (scan_rows add ~init:zero x))) in
      C.int ~expect:3 (List.length prog.kernels);
      C.int ~expect:9 (List.nth prog.kernels 0).launch.grid;
      C.int ~expect:3 (List.nth prog.kernels 1).launch.grid;
      well_formed prog.plan);
  C.test "every example lowers to a well-formed plan" (fun () ->
      List.iter (fun (e : Programs.t) -> well_formed (Lower.program (e.graph ())).plan) Programs.all);
  C.test "saxpy is still two kernels" (fun () ->
      C.int ~expect:2 (List.length (Lower.program (Ocaml_cuda_examples.Saxpy.program ~n:1000 ~a:2.0)).kernels));
  C.run ()
