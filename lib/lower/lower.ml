(** [Graph.t] -> [Kernel_ir.program]: the compiler proper.

    Every decision about what runs on the GPU is taken here. [Emit] is a
    pure pretty-printer, so anything left implicit at this layer cannot be
    fixed downstream.

    Two questions are answered per node, and only two: does it own a global
    buffer (asked of [Fusion], never decided here), and if it is a kernel
    root, what does its body look like. Everything else -- fusion, index
    arithmetic, launch geometry -- falls out of those.

    Node identity is the [Uid] and nothing else: a [Tensor.packed] wraps a
    whole GADT tree, so structural comparison would be both wrong and slow.
    Every table here is keyed by [Uid.to_int], and no hash-table iteration
    order reaches an output: the kernel list follows [Fusion.kernel_roots]
    and the host plan follows [Graph.params] / [Graph.topological_order] /
    [Graph.outputs]. Lowering is therefore deterministic, and it never
    mutates the graph. *)

open Ocaml_cuda_ir
module K = Kernel_ir

(* ------------------------------------------------------------------ *)
(* Small helpers                                                       *)
(* ------------------------------------------------------------------ *)

(* Index arithmetic is C [int] everywhere: thread ids, loop bounds, gather
   indices and buffer offsets. Never [long long]: a 64-bit index costs
   registers and doubles the address arithmetic for no benefit at v1
   sizes. *)
let int_dtype : Dtype.packed = Dtype.P Dtype.I32

let int_lit (n : int) : K.expr = K.Lit (K.I (Int64.of_int n), int_dtype)
let key (p : Tensor.packed) = Uid.to_int (Tensor.uid p)

(* Opening the existential is the only way to look at a node's constructor. *)
let node_is_param (Tensor.P t) =
  match t.Tensor.node with Tensor.Param _ -> true | _ -> false

(* A device literal, typed by the dtype index: the [Dtype.t] witness is what
   tells us whether the OCaml value is a float, an int32, an int64 or a
   bool. *)
let lit : type a. a Dtype.t -> a -> K.literal =
 fun d v ->
  match d with
  | Dtype.F32 -> K.F v
  | Dtype.F64 -> K.F v
  | Dtype.I32 -> K.I (Int64.of_int32 v)
  | Dtype.I64 -> K.I v
  | Dtype.Bool -> K.B v

(* ------------------------------------------------------------------ *)
(* Buffers: one per materialised node                                  *)
(* ------------------------------------------------------------------ *)

type buffers = (int, K.buffer) Hashtbl.t

(* Param buffers are named after the param so the host plan stays readable;
   everything else is named after its uid, which is already unique. Both are
   valid C identifiers whenever the param name is -- [Mangle] (T06) is what
   makes that unconditional, so no mangling happens here. *)
let buffer_of_node (Tensor.P t) : K.buffer =
  let name =
    match t.Tensor.node with
    | Tensor.Param n -> "p_" ^ n
    | _ -> "t" ^ string_of_int (Uid.to_int t.Tensor.uid)
  in
  {
    K.name;
    dtype = Dtype.P t.Tensor.dtype;
    memspace = K.Global;
    numel = Shape.numel t.Tensor.shape;
  }

let build_buffers (plan : Fusion.plan) (g : Graph.t) : buffers =
  let tbl : buffers = Hashtbl.create 64 in
  List.iter
    (fun p ->
      if Fusion.is_materialized plan p then Hashtbl.replace tbl (key p) (buffer_of_node p))
    (Graph.topological_order g);
  tbl

let find_buffer (tbl : buffers) (p : Tensor.packed) : K.buffer =
  match Hashtbl.find_opt tbl (key p) with
  | Some b -> b
  | None ->
      failwith
        (Printf.sprintf "Lower: node %s has no buffer (not materialised)"
           (Uid.to_string (Tensor.uid p)))

(* ------------------------------------------------------------------ *)
(* Scalar expressions                                                  *)
(* ------------------------------------------------------------------ *)

(* [env] binds [Arg i]; [index] binds [Index]. Polymorphic-recursive
   ([type a.]) because [Cmp] and [Cast] recurse at a type unrelated to
   their result. *)
let rec expr_of : type a. a Expr.t -> env:K.expr list -> index:K.expr -> K.expr =
 fun e ~env ~index ->
  let go : type b. b Expr.t -> K.expr = fun s -> expr_of s ~env ~index in
  match e.Expr.node with
  | Expr.Const v -> K.Lit (lit e.Expr.dtype v, Dtype.P e.Expr.dtype)
  | Expr.Arg i -> (
      match List.nth_opt env i with
      | Some x -> x
      | None -> failwith (Printf.sprintf "Lower: unbound Arg %d" i))
  | Expr.Index -> index
  | Expr.Binop (op, a, b) -> K.Binop (Dtype.P e.Expr.dtype, op, go a, go b)
  | Expr.Unop (op, a) -> K.Unop (Dtype.P e.Expr.dtype, op, go a)
  | Expr.Cmp (c, a, b) -> K.Cmp (c, go a, go b)
  | Expr.Logic (l, a, b) -> K.Logic (l, go a, go b)
  | Expr.Not a -> K.Not (go a)
  | Expr.Select (c, a, b) -> K.Select (go c, go a, go b)
  | Expr.Cast (a, d) -> K.Cast (Dtype.P d, go a)

(* ------------------------------------------------------------------ *)
(* Element expressions: this is the fusion mechanism                   *)
(* ------------------------------------------------------------------ *)

(* The value of element [index] of [p], as an expression in the consuming
   kernel. A materialised node is a [Load]; anything else is inlined, which
   is exactly what makes map-map, map-map2 and map-into-reduce fusion fall
   out with no n-ary node in the IR. *)
let rec elem (plan : Fusion.plan) (bufs : buffers) (p : Tensor.packed) ~(index : K.expr) :
    K.expr =
  if Fusion.is_materialized plan p then K.Load { buf = find_buffer bufs p; index }
  else inline_node plan bufs p ~index

(* Split out from [elem] so a kernel root can be inlined WITHOUT the
   materialisation test: calling [elem] on the root would return
   [Load out[i]], i.e. a kernel that reads its own output instead of
   computing it. *)
and inline_node (plan : Fusion.plan) (bufs : buffers) (Tensor.P t : Tensor.packed)
    ~(index : K.expr) : K.expr =
  match t.Tensor.node with
  | Tensor.Iota -> index (* iota[i] = i, and the flat index is already an int *)
  | Tensor.Map (fn, a) ->
      expr_of fn.Expr.body1 ~env:[ elem plan bufs (Tensor.P a) ~index ] ~index
  | Tensor.Map2 (fn, a, b) ->
      expr_of fn.Expr.body2
        ~env:[ elem plan bufs (Tensor.P a) ~index; elem plan bufs (Tensor.P b) ~index ]
        ~index
  | Tensor.Gather (idx, src) ->
      (* out[i] = src[idx[i]]: the index expression is itself an element
         expression, so a computed-index gather fuses too. *)
      elem plan bufs (Tensor.P src) ~index:(elem plan bufs (Tensor.P idx) ~index)
  | Tensor.Reshape (_, a) -> elem plan bufs (Tensor.P a) ~index (* flat index unchanged *)
  | Tensor.Param _ | Tensor.Reduce _ | Tensor.Scan _ ->
      (* [Fusion] materialises all three unconditionally. *)
      failwith "Lower: Param/Reduce/Scan must be materialised"

(* ------------------------------------------------------------------ *)
(* Kernel inputs                                                       *)
(* ------------------------------------------------------------------ *)

(* DFS from the root's DEPS (not the root itself, which is the output),
   stopping at and collecting materialised nodes. First-visit order, deduped
   by uid so an input feeding the kernel twice -- [map2 add s s] -- yields
   one param, not two. *)
let inputs_of (plan : Fusion.plan) (root : Tensor.packed) : Tensor.packed list =
  let seen = Hashtbl.create 16 in
  let acc = ref [] in
  let rec visit p =
    let k = key p in
    if not (Hashtbl.mem seen k) then begin
      Hashtbl.add seen k ();
      if Fusion.is_materialized plan p then acc := p :: !acc
      else List.iter visit (Tensor.deps p)
    end
  in
  List.iter visit (Tensor.deps root);
  List.rev !acc

(* ------------------------------------------------------------------ *)
(* One kernel per root                                                 *)
(* ------------------------------------------------------------------ *)

let kernel_of (plan : Fusion.plan) (bufs : buffers) (root : Tensor.packed) : K.kernel =
  let out = find_buffer bufs root in
  (* The root's own buffer is always the LAST param: [Executor] and the
     tests both rely on it. *)
  let params = List.map (find_buffer bufs) (inputs_of plan root) @ [ out ] in
  match root with
  | Tensor.P t -> (
      let name = "k_" ^ string_of_int (Uid.to_int t.Tensor.uid) in
      let dtype = Dtype.P t.Tensor.dtype in
      match t.Tensor.node with
      (* Element-wise: one grid-stride loop, one store. The whole fused
         sub-tree is a single expression inside the Store. *)
      | Tensor.Map _ | Tensor.Map2 _ | Tensor.Iota | Tensor.Gather _ | Tensor.Reshape _ ->
          let numel = Shape.numel t.Tensor.shape in
          let body =
            [
              K.For
                {
                  var = "i";
                  lo = K.Global_thread_id;
                  hi = int_lit numel;
                  step = K.Global_size;
                  body =
                    [
                      K.Store
                        {
                          buf = out;
                          index = K.Var "i";
                          value = inline_node plan bufs root ~index:(K.Var "i");
                        };
                    ];
                };
            ]
          in
          { K.name; params; shared = []; body; launch = Schedule.grid_stride ~numel }
      (* Reduction: classic shared-memory tree over ONE block. The tree
         assumes a single block, hence [Schedule.single_block] and
         [Local_thread_id]/[Block_dim] rather than the global ids. *)
      | Tensor.Reduce (fn, init, src) ->
          let n = Shape.numel (Tensor.shape (Tensor.P src)) in
          let sdata =
            { K.name = "sdata"; dtype; memspace = K.Shared; numel = Schedule.block_size }
          in
          let combine a b = expr_of fn.Expr.body2 ~env:[ a; b ] ~index:(int_lit 0) in
          let tid = K.Local_thread_id in
          (* block_size/2 down to 1, unrolled. Derived from [block_size] so
             it follows the schedule instead of contradicting it. *)
          let steps = List.init 8 (fun k -> Schedule.block_size lsr (k + 1)) in
          let tree =
            List.concat_map
              (fun s ->
                [
                  K.If
                    {
                      cond = K.Cmp (Expr.Lt, tid, int_lit s);
                      then_ =
                        [
                          K.Store
                            {
                              buf = sdata;
                              index = tid;
                              value =
                                combine
                                  (K.Load { buf = sdata; index = tid })
                                  (K.Load
                                     {
                                       buf = sdata;
                                       index = K.Binop (int_dtype, Expr.Add, tid, int_lit s);
                                     });
                            };
                        ];
                      else_ = [];
                    };
                  (* Between every step, without exception: step k+1 reads
                     what step k wrote from other threads' lanes. *)
                  K.Sync_threads;
                ])
              steps
          in
          let body =
            [
              (* Every one of the block_size threads starts its private [acc]
                 at [init], so [init] is folded into the result block_size
                 times. That is only correct because [init] is required to be
                 the IDENTITY of the operator -- the [Dsl.reduce] contract.
                 A non-identity [init] would be silently multiplied here. *)
              K.Let { var = "acc"; dtype; value = expr_of init ~env:[] ~index:(int_lit 0) };
              K.For
                {
                  var = "i";
                  lo = tid;
                  hi = int_lit n;
                  step = K.Block_dim;
                  body =
                    [
                      K.Assign
                        {
                          var = "acc";
                          value =
                            combine (K.Var "acc")
                              (elem plan bufs (Tensor.P src) ~index:(K.Var "i"));
                        };
                    ];
                };
              K.Store { buf = sdata; index = tid; value = K.Var "acc" };
              K.Sync_threads;
            ]
            @ tree
            (* Outside the tree loop, and guarded: exactly one thread writes
               the scalar result. *)
            @ [
                K.If
                  {
                    cond = K.Cmp (Expr.Eq, tid, int_lit 0);
                    then_ =
                      [
                        K.Store
                          {
                            buf = out;
                            index = int_lit 0;
                            value = K.Load { buf = sdata; index = int_lit 0 };
                          };
                      ];
                    else_ = [];
                  };
              ]
          in
          { K.name; params; shared = [ sdata ]; body; launch = Schedule.single_block }
      (* Scan: sequential on one thread in v1. Correct and obviously so; a
         Blelloch scan is a schedule change, not an IR change. *)
      | Tensor.Scan (fn, init, src) ->
          let n = Shape.numel (Tensor.shape (Tensor.P src)) in
          let combine a b = expr_of fn.Expr.body2 ~env:[ a; b ] ~index:(int_lit 0) in
          let body =
            [
              K.Let { var = "acc"; dtype; value = expr_of init ~env:[] ~index:(int_lit 0) };
              K.For
                {
                  var = "i";
                  lo = int_lit 0;
                  hi = int_lit n;
                  step = int_lit 1;
                  body =
                    [
                      K.Assign
                        {
                          var = "acc";
                          value =
                            combine (K.Var "acc")
                              (elem plan bufs (Tensor.P src) ~index:(K.Var "i"));
                        };
                      (* Inclusive scan: store after combining. *)
                      K.Store { buf = out; index = K.Var "i"; value = K.Var "acc" };
                    ];
                };
            ]
          in
          { K.name; params; shared = []; body; launch = Schedule.single_thread }
      | Tensor.Param _ ->
          (* Params are uploaded, never computed: [Fusion.kernel_roots]
             excludes them. *)
          failwith "Lower: a Param never gets a kernel")

(* ------------------------------------------------------------------ *)
(* Host plan                                                           *)
(* ------------------------------------------------------------------ *)

let program (g : Graph.t) : K.program =
  let plan = Fusion.plan g in
  let bufs = build_buffers plan g in
  let kernels = List.map (kernel_of plan bufs) (Fusion.kernel_roots plan) in
  (* Alloc order: params first (so uploads can follow in the same order),
     then every other materialised node in topological order. Frees mirror
     it exactly. *)
  let param_nodes = List.map snd (Graph.params g) in
  let computed =
    List.filter
      (fun p -> Fusion.is_materialized plan p && not (node_is_param p))
      (Graph.topological_order g)
  in
  let alloc_order = List.map (find_buffer bufs) (param_nodes @ computed) in
  let allocs = List.map (fun b -> K.Alloc b) alloc_order in
  let uploads =
    List.map (fun (n, p) -> K.Upload { param = n; into = find_buffer bufs p }) (Graph.params g)
  in
  (* Same order as [kernels], which is [Fusion.kernel_roots], which is
     topological: every input was written by an earlier launch or uploaded. *)
  let launches =
    List.map (fun (k : K.kernel) -> K.Launch { kernel = k.K.name; args = k.K.params }) kernels
  in
  (* An output that is a Param has no kernel; it downloads straight from the
     param buffer, which [find_buffer] gives us for free. *)
  let downloads =
    List.map
      (fun (n, p) ->
        K.Download { from = find_buffer bufs p; output = n; shape = Tensor.shape p })
      (Graph.outputs g)
  in
  let frees = List.map (fun b -> K.Free b) alloc_order in
  { K.name = Graph.name g; kernels; plan = allocs @ uploads @ launches @ downloads @ frees }
