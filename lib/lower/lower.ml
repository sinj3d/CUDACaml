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

(* [Reduce] and [Scan] run along the LAST axis, so a kernel's geometry is
   (rows, row length) rather than a single element count. Both fall out of
   the shapes: the row length is the last extent of the source and the row
   count is the numel of the reduce's output, or numel/row_length for a
   scan. Neither is re-derived from the node's constructor. *)
let row_length (shape : Shape.t) : int =
  match List.rev (Shape.dims shape) with [] -> 1 | n :: _ -> n

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

(* The additive identity at an element type: what the zero-fill kernel of a
   [Scatter_add] stores before any atomic lands. *)
let zero_lit : type a. a Dtype.t -> K.literal = function
  | Dtype.F32 | Dtype.F64 -> K.F 0.0
  | Dtype.I32 | Dtype.I64 -> K.I 0L
  | Dtype.Bool -> K.B false

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
  | Tensor.Broadcast (_, a) ->
      (* out[i] = src[0] for every i: the consumer's [index] is DISCARDED.
         The source has numel 1, so any other index would read out of
         bounds, silently, on the device. *)
      elem plan bufs (Tensor.P a) ~index:(int_lit 0)
  | Tensor.Param _ | Tensor.Reduce _ | Tensor.Scan _ | Tensor.Scatter_add _
  | Tensor.Matmul _ ->
      (* [Fusion] materialises all five unconditionally. A [Scatter_add] has
         no per-element expression at all -- output element [i] is the sum of
         an unknown subset of the source -- so there is nothing to inline
         even in principle, and a [Matmul] element is a whole inner product
         computed by a co-operating tile. *)
      failwith "Lower: Param/Reduce/Scan/Scatter_add/Matmul must be materialised"

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
(* Kernels per root                                                    *)
(* ------------------------------------------------------------------ *)

(* What one kernel root lowers to: the kernels, in launch order, plus the
   scratch buffers those kernels need. A root usually lowers to exactly one
   kernel and no scratch, but not always: a [Scatter_add] needs a zero-fill
   pass before its atomics, and a large [Reduce] or a multi-chunk [Scan]
   becomes two or three passes communicating through a scratch buffer.
   [program] allocates the scratch immediately before the root's first
   launch and frees it immediately after its last, so a scratch buffer is
   live for exactly as long as the root that owns it. *)
type lowered = { kernels : K.kernel list; scratch : K.buffer list }

let kernels_of (plan : Fusion.plan) (bufs : buffers) (root : Tensor.packed) : lowered =
  let out = find_buffer bufs root in
  let ins = List.map (find_buffer bufs) (inputs_of plan root) in
  (* The root's own buffer is always the LAST param: [Executor] and the
     tests both rely on it. *)
  let params = ins @ [ out ] in
  let add a b = K.Binop (int_dtype, Expr.Add, a, b) in
  let sub a b = K.Binop (int_dtype, Expr.Sub, a, b) in
  let mul a b = K.Binop (int_dtype, Expr.Mul, a, b) in
  let div a b = K.Binop (int_dtype, Expr.Div, a, b) in
  match root with
  | Tensor.P t -> (
      let name = "k_" ^ string_of_int (Uid.to_int t.Tensor.uid) in
      let scratch_name suffix = "t" ^ string_of_int (Uid.to_int t.Tensor.uid) ^ suffix in
      let dtype = Dtype.P t.Tensor.dtype in
      match t.Tensor.node with
      (* Element-wise: one grid-stride loop, one store. The whole fused
         sub-tree is a single expression inside the Store. *)
      | Tensor.Map _ | Tensor.Map2 _ | Tensor.Iota | Tensor.Gather _ | Tensor.Reshape _
      | Tensor.Broadcast _ ->
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
          {
            kernels =
              [ { K.name; params; shared = []; body; launch = Schedule.grid_stride ~numel } ];
            scratch = [];
          }
      (* Reduction: classic shared-memory tree, one block PER ROW when the
         row is short enough that a single block saturates it. The tree
         co-operates within a block, so the ids are [Local_thread_id] and
         [Block_dim], never the global ones -- [Global_thread_id] already
         folds in [blockIdx.x] and would skew every row after the first.

         A long row instead gets [g = Schedule.reduce_blocks] blocks, and
         the same tree runs twice: once per (row, block) pair into a
         partials buffer, then once per row over that buffer. Both passes
         go through [reduction], which differs only in what it folds and in
         where the block leader puts the answer. *)
      | Tensor.Reduce (fn, init, src) ->
          let n = row_length (Tensor.shape (Tensor.P src)) in
          let rows = Shape.numel t.Tensor.shape in
          let g = Schedule.reduce_blocks ~row_len:n in
          let sdata =
            { K.name = "sdata"; dtype; memspace = K.Shared; numel = Schedule.block_size }
          in
          let combine a b = expr_of fn.Expr.body2 ~env:[ a; b ] ~index:(int_lit 0) in
          let identity () = expr_of init ~env:[] ~index:(int_lit 0) in
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
                                  (K.Load { buf = sdata; index = add tid (int_lit s) });
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
          (* [prelude] declares whatever the loop bounds mention; the loop
             folds [load i] for [i] from [lo] to [hi] by [step]; thread 0
             stores the block result at [dst.(dst_index)]. *)
          let reduction ~prelude ~lo ~hi ~step ~load ~dst ~dst_index =
            prelude
            @ [
                (* Every one of the block_size threads starts its private
                   [acc] at [init], so [init] is folded into the result
                   block_size times. That is only correct because [init] is
                   required to be the IDENTITY of the operator -- the
                   [Dsl.reduce] contract. A non-identity [init] would be
                   silently multiplied here. *)
                K.Let { var = "acc"; dtype; value = identity () };
                K.For
                  {
                    var = "i";
                    lo;
                    hi;
                    step;
                    body =
                      [
                        K.Assign
                          { var = "acc"; value = combine (K.Var "acc") (load (K.Var "i")) };
                      ];
                  };
                K.Store { buf = sdata; index = tid; value = K.Var "acc" };
                K.Sync_threads;
              ]
            @ tree
            (* Outside the tree, and guarded: exactly one thread per block
               writes that block's result. *)
            @ [
                K.If
                  {
                    cond = K.Cmp (Expr.Eq, tid, int_lit 0);
                    then_ =
                      [
                        K.Store
                          {
                            buf = dst;
                            index = dst_index;
                            value = K.Load { buf = sdata; index = int_lit 0 };
                          };
                      ];
                    else_ = [];
                  };
              ]
          in
          if g = 1 then
            (* v1, unchanged: [Block_id] is the row, and it appears only in
               the [row * n] offsets and in the final store. For a rank-1
               source [rows = 1] and every [blockIdx.x * n] term is zero. *)
            let row_base = mul K.Block_id (int_lit n) in
            let body =
              reduction ~prelude:[] ~lo:(add row_base tid) ~hi:(add row_base (int_lit n))
                ~step:K.Block_dim
                ~load:(fun i -> elem plan bufs (Tensor.P src) ~index:i)
                ~dst:out ~dst_index:K.Block_id
            in
            {
              kernels =
                [
                  {
                    K.name;
                    params;
                    shared = [ sdata ];
                    body;
                    launch = Schedule.rows_block ~rows;
                  };
                ];
              scratch = [];
            }
          else
            (* [out] cannot double as the partials buffer: it holds one
               element per row, not [g] of them. *)
            let partials =
              {
                K.name = scratch_name "_partials";
                dtype;
                memspace = K.Global;
                numel = rows * g;
              }
            in
            (* Pass 1: the grid is (row, block-within-row) flattened, so
               [blockIdx.x] no longer IS the row and both halves have to be
               recovered by division. The stride is a whole row's worth of
               threads, [g * block_size], which keeps each block's reads
               coalesced across its slice of the row. *)
            let row = K.Var "row" and blk = K.Var "blk" in
            let row_base = mul row (int_lit n) in
            let partial_body =
              reduction
                ~prelude:
                  [
                    K.Let
                      { var = "row"; dtype = int_dtype; value = div K.Block_id (int_lit g) };
                    K.Let
                      {
                        var = "blk";
                        dtype = int_dtype;
                        value = sub K.Block_id (mul row (int_lit g));
                      };
                  ]
                ~lo:(add row_base (add (mul blk (int_lit Schedule.block_size)) tid))
                ~hi:(add row_base (int_lit n))
                ~step:(int_lit (g * Schedule.block_size))
                ~load:(fun i -> elem plan bufs (Tensor.P src) ~index:i)
                ~dst:partials ~dst_index:K.Block_id
            in
            let partial_kernel =
              {
                K.name = name ^ "_partial";
                params = ins @ [ partials ];
                shared = [ sdata ];
                body = partial_body;
                launch = Schedule.rows_block ~rows:(rows * g);
              }
            in
            (* Pass 2: one block per row again, folding the [g] partials of
               that row. [g <= 128 <= block_size], so the loop body runs at
               most once per thread and the tree does the rest. *)
            let p_base = mul K.Block_id (int_lit g) in
            let final_body =
              reduction ~prelude:[] ~lo:(add p_base tid) ~hi:(add p_base (int_lit g))
                ~step:K.Block_dim
                ~load:(fun i -> K.Load { buf = partials; index = i })
                ~dst:out ~dst_index:K.Block_id
            in
            let final_kernel =
              {
                K.name;
                params = [ partials; out ];
                shared = [ sdata ];
                body = final_body;
                launch = Schedule.rows_block ~rows;
              }
            in
            { kernels = [ partial_kernel; final_kernel ]; scratch = [ partials ] }
      (* Scan: a Hillis-Steele inclusive scan in shared memory, one chunk of
         [Schedule.scan_chunk] elements per block. A row that fits in one
         chunk is one kernel; a longer row takes the classic three passes,
         because nothing short of a kernel boundary synchronises blocks.

         [combine] is only required to be ASSOCIATIVE, not commutative, so
         every combination below keeps the earlier element on the left. *)
      | Tensor.Scan (fn, init, src) ->
          let src_shape = Tensor.shape (Tensor.P src) in
          let n = row_length src_shape in
          let rows = if n = 0 then 0 else Shape.numel src_shape / n in
          let c = Schedule.scan_chunk in
          (* Chunks per row. An empty row still needs one legal chunk. *)
          let nb = if n <= 0 then 1 else (n + c - 1) / c in
          let combine a b = expr_of fn.Expr.body2 ~env:[ a; b ] ~index:(int_lit 0) in
          let identity () = expr_of init ~env:[] ~index:(int_lit 0) in
          let sdata = { K.name = "sdata"; dtype; memspace = K.Shared; numel = c } in
          let tid = K.Local_thread_id in
          let s_at i = K.Load { buf = sdata; index = i } in
          (* Offsets 1, 2, 4, ... < c, unrolled: [K.For] steps by addition
             and this schedule doubles. *)
          let offsets =
            let rec go o acc = if o >= c then List.rev acc else go (o * 2) (o :: acc) in
            go 1 []
          in
          (* TWO barriers per step, not one: the first publishes the
             previous step's writes, the second stops this step's write from
             landing in a lane another thread has not read yet. The value in
             between lives in a register, one [Let] per step -- the names
             have to differ because the unrolled steps share a C scope. *)
          let hillis_steele =
            List.concat_map
              (fun o ->
                let v = Printf.sprintf "v%d" o in
                [
                  K.Sync_threads;
                  K.Let
                    {
                      var = v;
                      dtype;
                      value =
                        K.Select
                          ( K.Cmp (Expr.Ge, tid, int_lit o),
                            combine (s_at (sub tid (int_lit o))) (s_at tid),
                            s_at tid );
                    };
                  K.Sync_threads;
                  K.Store { buf = sdata; index = tid; value = K.Var v };
                ])
              offsets
            (* The chunk total is read by thread 0 alone, out of a lane it
               did not write, so the last write needs a barrier too. *)
            @ [ K.Sync_threads ]
          in
          (* One chunk of [len] valid elements starting at flat index
             [base]. Lanes past [len] hold the identity, which leaves the
             valid prefixes untouched and never reads out of bounds: the
             emitted [?:] does not evaluate the load it guards. *)
          let chunk_scan ~base ~len =
            [
              K.Store
                {
                  buf = sdata;
                  index = tid;
                  value =
                    K.Select
                      ( K.Cmp (Expr.Lt, tid, len),
                        elem plan bufs (Tensor.P src) ~index:(add base tid),
                        identity () );
                };
            ]
            @ hillis_steele
            @ [
                K.If
                  {
                    cond = K.Cmp (Expr.Lt, tid, len);
                    then_ = [ K.Store { buf = out; index = add base tid; value = s_at tid } ];
                    else_ = [];
                  };
              ]
          in
          let chunk_launch ~blocks =
            {
              Schedule.grid = max 1 blocks;
              grid_y = 1;
              block = c;
              block_y = 1;
              shared_bytes = 0;
            }
          in
          if nb = 1 then
            {
              kernels =
                [
                  {
                    K.name;
                    params;
                    shared = [ sdata ];
                    body = chunk_scan ~base:(mul K.Block_id (int_lit n)) ~len:(int_lit n);
                    launch = chunk_launch ~blocks:rows;
                  };
                ];
              scratch = [];
            }
          else
            let sums =
              { K.name = scratch_name "_sums"; dtype; memspace = K.Global; numel = rows * nb }
            in
            (* Pass 1: the grid is (row, chunk) flattened. The last valid
               element of a chunk is that chunk's inclusive total, which is
               what pass 2 scans. *)
            let row = K.Var "row" and blk = K.Var "blk" in
            let chunks_body =
              [
                K.Let { var = "row"; dtype = int_dtype; value = div K.Block_id (int_lit nb) };
                K.Let
                  {
                    var = "blk";
                    dtype = int_dtype;
                    value = sub K.Block_id (mul row (int_lit nb));
                  };
                K.Let
                  {
                    var = "base";
                    dtype = int_dtype;
                    value = add (mul row (int_lit n)) (mul blk (int_lit c));
                  };
                (* The final chunk of a row is short unless [c] divides [n]. *)
                K.Let
                  {
                    var = "len";
                    dtype = int_dtype;
                    value =
                      (let rest = sub (int_lit n) (mul blk (int_lit c)) in
                       K.Select (K.Cmp (Expr.Lt, rest, int_lit c), rest, int_lit c));
                  };
              ]
              @ chunk_scan ~base:(K.Var "base") ~len:(K.Var "len")
              @ [
                  K.If
                    {
                      cond = K.Cmp (Expr.Eq, tid, int_lit 0);
                      then_ =
                        [
                          K.Store
                            {
                              buf = sums;
                              index = K.Block_id;
                              value = s_at (sub (K.Var "len") (int_lit 1));
                            };
                        ];
                      else_ = [];
                    };
                ]
            in
            let chunks_kernel =
              {
                K.name = name ^ "_chunks";
                params = ins @ [ sums; out ];
                shared = [ sdata ];
                body = chunks_body;
                launch = chunk_launch ~blocks:(rows * nb);
              }
            in
            (* Pass 2: [nb] chunk totals per row is far too little work to
               parallelise, so this is the v1 sequential scan body, in
               place, one thread per row. *)
            let s_base = mul K.Block_id (int_lit nb) in
            let sums_kernel =
              {
                K.name = name ^ "_sums";
                params = [ sums ];
                shared = [];
                body =
                  [
                    K.Let { var = "acc"; dtype; value = identity () };
                    K.For
                      {
                        var = "i";
                        lo = s_base;
                        hi = add s_base (int_lit nb);
                        step = int_lit 1;
                        body =
                          [
                            K.Assign
                              {
                                var = "acc";
                                value =
                                  combine (K.Var "acc")
                                    (K.Load { buf = sums; index = K.Var "i" });
                              };
                            K.Store { buf = sums; index = K.Var "i"; value = K.Var "acc" };
                          ];
                      };
                  ];
                launch = Schedule.rows_thread ~rows;
              }
            in
            (* Pass 3: chunk [blk] is missing the total of chunks
               [0 .. blk-1], which is the INCLUSIVE [sums[row*nb + blk - 1]]
               -- pass 2 left those totals inclusive on purpose. Chunk 0 is
               already right. Each element is read and written by the one
               thread that owns it, so no atomics and no barrier. *)
            let i = K.Var "i" in
            let row3 = K.Var "row" in
            let final_kernel =
              {
                K.name;
                params = [ sums; out ];
                shared = [];
                body =
                  [
                    K.For
                      {
                        var = "i";
                        lo = K.Global_thread_id;
                        hi = int_lit (rows * n);
                        step = K.Global_size;
                        body =
                          [
                            K.Let { var = "row"; dtype = int_dtype; value = div i (int_lit n) };
                            K.Let
                              {
                                var = "blk";
                                dtype = int_dtype;
                                value = div (sub i (mul row3 (int_lit n))) (int_lit c);
                              };
                            K.If
                              {
                                cond = K.Cmp (Expr.Gt, K.Var "blk", int_lit 0);
                                then_ =
                                  [
                                    K.Store
                                      {
                                        buf = out;
                                        index = i;
                                        value =
                                          combine
                                            (K.Load
                                               {
                                                 buf = sums;
                                                 index =
                                                   sub
                                                     (add (mul row3 (int_lit nb))
                                                        (K.Var "blk"))
                                                     (int_lit 1);
                                               })
                                            (K.Load { buf = out; index = i });
                                      };
                                  ];
                                else_ = [];
                              };
                          ];
                      };
                  ];
                launch = Schedule.grid_stride ~numel:(rows * n);
              }
            in
            { kernels = [ chunks_kernel; sums_kernel; final_kernel ]; scratch = [ sums ] }
      (* A root that is two kernels over one buffer. The output buffer is
         written twice: a grid-stride zero fill over the OUTPUT, then a
         grid-stride pass over the SOURCE in which every element lands in
         the output through an atomic add.

         The zero fill cannot be folded into the scatter kernel: by the time
         a block reached the output elements it would clear, other blocks
         may already have added to them. Two launches on the same stream are
         ordered, so no explicit synchronisation is needed between them. *)
      | Tensor.Scatter_add (idx, src, _) ->
          let m = Shape.numel t.Tensor.shape in
          let n = Shape.numel (Tensor.shape (Tensor.P src)) in
          let zero_body =
            [
              K.For
                {
                  var = "i";
                  lo = K.Global_thread_id;
                  hi = int_lit m;
                  step = K.Global_size;
                  body =
                    [
                      K.Store
                        {
                          buf = out;
                          index = K.Var "i";
                          value = K.Lit (zero_lit t.Tensor.dtype, dtype);
                        };
                    ];
                };
            ]
          in
          (* The zero kernel touches nothing but the output, so it takes the
             output alone -- still the LAST param, as every kernel's output
             is. *)
          let zero =
            {
              K.name = name ^ "_zero";
              params = [ out ];
              shared = [];
              body = zero_body;
              launch = Schedule.grid_stride ~numel:m;
            }
          in
          (* [idx] and [src] are both element expressions, so a computed
             index or a fused source costs no buffer. The index is NOT
             bounds-checked here: the graph-level contract says the device
             kernel is unguarded, and the interpreter is where an
             out-of-range index is caught. *)
          let scatter_body =
            [
              K.For
                {
                  var = "i";
                  lo = K.Global_thread_id;
                  hi = int_lit n;
                  step = K.Global_size;
                  body =
                    [
                      K.Atomic_add
                        {
                          buf = out;
                          index = elem plan bufs (Tensor.P idx) ~index:(K.Var "i");
                          value = elem plan bufs (Tensor.P src) ~index:(K.Var "i");
                        };
                    ];
                };
            ]
          in
          let scatter =
            {
              K.name;
              params;
              shared = [];
              body = scatter_body;
              launch = Schedule.grid_stride ~numel:n;
            }
          in
          { kernels = [ zero; scatter ]; scratch = [] }
      (* Dense matmul, blocked by [Schedule.tile]: the one 2-D launch in the
         compiler. A block owns one [tile x tile] patch of the output and
         walks the shared dimension a tile at a time, staging a tile of each
         operand in shared memory, so every element read from global memory
         is reused [tile] times.

         x indexes COLUMNS and y indexes ROWS -- on the grid and inside the
         block alike -- so that threads adjacent in x read adjacent columns
         of [b] and write adjacent columns of the output, which is what
         makes the global accesses coalesce. Swapping the two roles is not a
         performance detail: it computes a different matrix.

         The tile loads go through [elem], so an inlineable operand (a map,
         a transpose, a broadcast) is fused into the load and costs no
         buffer. They are guarded rather than assumed in range: the emitted
         [?:] does not evaluate the branch it does not take, so an [m], [n]
         or [k] that is not a multiple of [tile] reads nothing out of bounds
         and the padding lanes hold the additive identity, which leaves the
         inner product unchanged. *)
      | Tensor.Matmul (a, b) ->
          let m, n =
            match Shape.dims t.Tensor.shape with
            | [ m; n ] -> (m, n)
            | dims ->
                failwith
                  (Printf.sprintf "Lower: Matmul output has rank %d, expected 2"
                     (List.length dims))
          in
          let k = row_length (Tensor.shape (Tensor.P a)) in
          let tile = Schedule.tile in
          (* Tiles along the shared dimension. [k = 0] gives none, and the
             kernel then stores the identity, exactly as the interpreter
             does for an empty inner product. *)
          let nt = (k + tile - 1) / tile in
          let tx = K.Local_thread_id and ty = K.Local_thread_id_y in
          let sa = { K.name = "sa"; dtype; memspace = K.Shared; numel = tile * tile } in
          let sb = { K.name = "sb"; dtype; memspace = K.Shared; numel = tile * tile } in
          let zero = K.Lit (zero_lit t.Tensor.dtype, dtype) in
          let row = K.Var "row" and col = K.Var "col" in
          let lt x lim = K.Cmp (Expr.Lt, x, int_lit lim) in
          let both x y = K.Logic (Expr.And, x, y) in
          (* Every thread stages exactly one element of each tile, at its own
             [(ty, tx)] slot. *)
          let slot = add (mul ty (int_lit tile)) tx in
          let step_body =
            [
              (* Within a tile step, the element of [a] a thread loads is in
                 its OWN row and the x-th column of the tile; the element of
                 [b] is in the y-th row of the tile and its own column. *)
              K.Let
                {
                  var = "ka";
                  dtype = int_dtype;
                  value = add (mul (K.Var "kt") (int_lit tile)) tx;
                };
              K.Let
                {
                  var = "kb";
                  dtype = int_dtype;
                  value = add (mul (K.Var "kt") (int_lit tile)) ty;
                };
              K.Store
                {
                  buf = sa;
                  index = slot;
                  value =
                    K.Select
                      ( both (lt row m) (lt (K.Var "ka") k),
                        elem plan bufs (Tensor.P a)
                          ~index:(add (mul row (int_lit k)) (K.Var "ka")),
                        zero );
                };
              K.Store
                {
                  buf = sb;
                  index = slot;
                  value =
                    K.Select
                      ( both (lt (K.Var "kb") k) (lt col n),
                        elem plan bufs (Tensor.P b)
                          ~index:(add (mul (K.Var "kb") (int_lit n)) col),
                        zero );
                };
              (* Before the tile is read: every lane must be published. *)
              K.Sync_threads;
              K.For
                {
                  var = "p";
                  lo = int_lit 0;
                  hi = int_lit tile;
                  step = int_lit 1;
                  body =
                    [
                      K.Assign
                        {
                          var = "acc";
                          value =
                            K.Binop
                              ( dtype,
                                Expr.Add,
                                K.Var "acc",
                                K.Binop
                                  ( dtype,
                                    Expr.Mul,
                                    K.Load
                                      {
                                        buf = sa;
                                        index = add (mul ty (int_lit tile)) (K.Var "p");
                                      },
                                    K.Load
                                      {
                                        buf = sb;
                                        index = add (mul (K.Var "p") (int_lit tile)) tx;
                                      } ) );
                        };
                    ];
                };
              (* And after: the next step overwrites these very lanes, which
                 a thread still inside the loop above would then read. One
                 barrier per step is the classic bug. *)
              K.Sync_threads;
            ]
          in
          let body =
            [
              K.Let
                {
                  var = "row";
                  dtype = int_dtype;
                  value = add (mul K.Block_id_y (int_lit tile)) ty;
                };
              K.Let
                {
                  var = "col";
                  dtype = int_dtype;
                  value = add (mul K.Block_id (int_lit tile)) tx;
                };
              (* The accumulator is a register for the whole kernel: one
                 [Let] here, an [Assign] per product. *)
              K.Let { var = "acc"; dtype; value = zero };
              K.For
                {
                  var = "kt";
                  lo = int_lit 0;
                  hi = int_lit nt;
                  step = int_lit 1;
                  body = step_body;
                };
              (* A block on the ragged edge of the output has threads with no
                 element to write. They still ran the loop above, because
                 they take part in the staging and in every barrier. *)
              K.If
                {
                  cond = both (lt row m) (lt col n);
                  then_ =
                    [
                      K.Store
                        {
                          buf = out;
                          index = add (mul row (int_lit n)) col;
                          value = K.Var "acc";
                        };
                    ];
                  else_ = [];
                };
            ]
          in
          {
            kernels =
              [ { K.name; params; shared = [ sa; sb ]; body; launch = Schedule.tiled_2d ~m ~n } ];
            scratch = [];
          }
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
  let lowered = List.map (kernels_of plan bufs) (Fusion.kernel_roots plan) in
  let kernels = List.concat_map (fun l -> l.kernels) lowered in
  (* Alloc order: params first (so uploads can follow in the same order),
     then every other materialised node in topological order. Frees mirror
     it exactly. Scratch buffers are NOT in here: they are allocated and
     freed around the launches of the single root that owns them. *)
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
     topological: every input was written by an earlier launch or uploaded.
     A root's scratch is bracketed by that root's own launches, so it lives
     for the shortest window that is correct and every [Launch] argument is
     still allocated before the launch and freed after it. *)
  let launches =
    List.concat_map
      (fun l ->
        List.map (fun b -> K.Alloc b) l.scratch
        @ List.map
            (fun (k : K.kernel) -> K.Launch { kernel = k.K.name; args = k.K.params })
            l.kernels
        @ List.map (fun b -> K.Free b) l.scratch)
      lowered
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
