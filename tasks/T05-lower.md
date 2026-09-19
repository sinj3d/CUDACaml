# T05 — `Schedule` + `Lower`: graph → imperative kernels + host plan

## Goal

Implement `lib/lower/schedule.ml` (launch geometry) and `lib/lower/lower.ml`
(the compiler proper). `Lower.program` turns a `Graph.t` into a
`Kernel_ir.program`: one kernel per `Fusion.kernel_roots` entry plus the
host plan (alloc / upload / launch / download / free). Everything about
*what runs on the GPU* is decided here. `Emit` (T06) only prints the
result, so any performance or correctness decision you leave implicit
cannot be fixed downstream.

Depends on: T02, T04.

## Files you own

- `lib/lower/schedule.ml` — implement `grid_stride`; keep the rest.
- `lib/lower/lower.ml` — replace the stub.

Do not edit `kernel_ir.ml`, `fusion.*`, or any `.mli`.

## Interfaces

`lib/lower/schedule.mli`:

```ocaml
type launch = { grid : int; block : int; shared_bytes : int }
val block_size : int                      (* = 256, already set *)
val grid_stride : numel:int -> launch     (* YOU implement *)
val single_block : launch                 (* {grid=1; block=256; shared_bytes=0}, done *)
val single_thread : launch                (* {grid=1; block=1;   shared_bytes=0}, done *)
```

`grid_stride ~numel = { grid = max 1 (min 1024 ((numel + 255) / 256)); block = 256; shared_bytes = 0 }`.

`lib/lower/lower.mli`: `val program : Graph.t -> Kernel_ir.program`.

The target (`lib/lower/kernel_ir.ml`, verbatim — you construct these):

```ocaml
type memspace = Global | Shared
type buffer = { name : string; dtype : Dtype.packed; memspace : memspace; numel : int }
type literal = F of float | I of int64 | B of bool
type expr =
  | Var of string
  | Lit of literal * Dtype.packed
  | Global_thread_id            (* blockIdx.x * blockDim.x + threadIdx.x *)
  | Global_size                 (* gridDim.x * blockDim.x *)
  | Local_thread_id             (* threadIdx.x *)
  | Block_dim                   (* blockDim.x *)
  | Load of { buf : buffer; index : expr }
  | Binop of Dtype.packed * Expr.binop * expr * expr   (* dtype = result type *)
  | Unop of Dtype.packed * Expr.unop * expr
  | Cmp of Expr.cmp * expr * expr
  | Logic of Expr.logic * expr * expr
  | Not of expr
  | Select of expr * expr * expr
  | Cast of Dtype.packed * expr
type stmt =
  | Let of { var : string; dtype : Dtype.packed; value : expr }
  | Assign of { var : string; value : expr }
  | Store of { buf : buffer; index : expr; value : expr }
  | For of { var : string; lo : expr; hi : expr; step : expr; body : stmt list }  (* for (int v=lo; v<hi; v+=step) *)
  | If of { cond : expr; then_ : stmt list; else_ : stmt list }
  | Sync_threads
type kernel = { name : string; params : buffer list; shared : buffer list; body : stmt list; launch : Schedule.launch }
type host_op =
  | Alloc of buffer
  | Upload of { param : string; into : buffer }
  | Launch of { kernel : string; args : buffer list }
  | Download of { from : buffer; output : string; shape : Shape.t }
  | Free of buffer
type program = { name : string; kernels : kernel list; plan : host_op list }
```

Sources: `Tensor.node` and `Expr.node` (both listed in T02/T03; read
`lib/ir/tensor.ml` and `lib/ir/expr.ml`). `Fusion.plan/is_materialized/kernel_roots`.
`Graph.name/params/outputs/topological_order`. `Dtype.P`, `Dtype.name`.
Index expressions are always C `int` (`Dtype.P Dtype.I32`).

## Implementation

### Buffers

One `buffer` per materialised node, `memspace = Global`, `numel = Shape.numel shape`:
- `Param name` → `buffer.name = "p_" ^ name`
- anything else → `"t" ^ string_of_int (Uid.to_int uid)`

Build a `(int, buffer) Hashtbl.t` keyed by uid up front for all
materialised nodes (Params included). Buffer names are already valid C
identifiers if the param name is; do not mangle here (T06 does).

### Literals

```ocaml
let lit : type a. a Dtype.t -> a -> Kernel_ir.literal = fun d v ->
  match d with F32 -> F v | F64 -> F v | I32 -> I (Int64.of_int32 v) | I64 -> I v | Bool -> B v
```

### Element expressions (the fusion mechanism)

`elem : plan -> buffers -> Tensor.packed -> index:Kernel_ir.expr -> Kernel_ir.expr`
gives the value of element `index` of a tensor:

- If the node is materialised → `Load { buf; index }`.
  (Params, Reduce, Scan, outputs, fan-out ≥ 2 all land here.)
- Else inline by node:
  - `Iota` → `index` (already an int expression).
  - `Map (fn, a)` → `expr_of fn.body1 ~env:[elem a ~index] ~index`
  - `Map2 (fn, a, b)` → env `[elem a ~index; elem b ~index]`
  - `Gather (idx, src)` → `elem src ~index:(elem idx ~index)`
  - `Reshape (_, a)` → `elem a ~index` (flat index unchanged)
  - `Reduce`/`Scan`/`Param` → unreachable (always materialised); raise.

`expr_of : 'a Expr.t -> env:Kernel_ir.expr list -> index:Kernel_ir.expr -> Kernel_ir.expr`:
- `Const v` → `Lit (lit e.dtype v, P e.dtype)`
- `Arg i` → `List.nth env i`
- `Index` → `index`
- `Binop (op, a, b)` → `Binop (P e.dtype, op, go a, go b)`
- `Unop (op, a)` → `Unop (P e.dtype, op, go a)`
- `Cmp`, `Logic`, `Not`, `Select` → structural
- `Cast (a, d)` → `Cast (P d, go a)`

`expr_of` must be `type a.` polymorphic-recursive because `Cmp`'s
operands have a different type from the result.

### Inputs of a kernel

`inputs_of root`: DFS from the root's **deps** (not the root itself),
following `Tensor.deps`, stopping at (and collecting) materialised nodes.
Dedupe by uid, keep first-visit order. These become the kernel's leading
params; the root's own buffer is appended last. So `params = inputs @ [out]`.

### Kernel per root

Root name: `"k_" ^ string_of_int (Uid.to_int uid)`.

**Element-wise root** (`Map`, `Map2`, `Iota`, `Gather`, `Reshape`):

```
launch = Schedule.grid_stride ~numel
body = [ For { var = "i"; lo = Global_thread_id; hi = Lit (I numel, I32); step = Global_size;
               body = [ Store { buf = out; index = Var "i"; value = <root's element expr at index Var "i"> } ] } ]
```
The root's element expression is computed by the *inline* rules above,
not by `elem` (which would return a `Load` of the root itself). Factor
`inline_node` out of `elem` and call it directly for the root.

**Reduce root** (`Reduce (fn, init, src)`), `d = root dtype`, `n = numel src`:

```
launch = Schedule.single_block
shared = [ { name = "sdata"; dtype = P d; memspace = Shared; numel = Schedule.block_size } ]
body =
  Let { var = "acc"; dtype = P d; value = expr_of init ~env:[] ~index:(Lit (I 0, I32)) }
  For { var = "i"; lo = Local_thread_id; hi = Lit n; step = Block_dim;
        body = [ Assign { var = "acc"; value = combine (Var "acc") (elem src ~index:(Var "i")) } ] }
  Store { buf = sdata; index = Local_thread_id; value = Var "acc" }
  Sync_threads
  for s in [128; 64; 32; 16; 8; 4; 2; 1]:          (* block_size / 2 down to 1, unrolled *)
    If { cond = Cmp (Lt, Local_thread_id, Lit s);
         then_ = [ Store { buf = sdata; index = Local_thread_id;
                           value = combine (Load sdata[tid]) (Load sdata[tid + s]) } ];
         else_ = [] }
    Sync_threads
  If { cond = Cmp (Eq, Local_thread_id, Lit 0); then_ = [ Store { buf = out; index = Lit 0; value = Load sdata[0] } ]; else_ = [] }
```
`combine x y = expr_of fn.body2 ~env:[x; y] ~index:(Lit 0)`. Compute the
unrolled step list as `List.init 8 (fun k -> block_size lsr (k + 1))`, not
as a hard-coded literal, so it follows `block_size`. Every thread's `acc`
starts at `init`, so `init` is folded in 256 times — this is only correct
if `init` is the identity of the op, which the `Dsl.reduce` contract
requires. Document this in a comment.

**Scan root** (`Scan (fn, init, src)`), sequential in v1:

```
launch = Schedule.single_thread ; shared = []
body =
  Let acc = init
  For i = 0 .. n step 1:
     Assign acc = combine (Var "acc") (elem src ~index:(Var "i"))
     Store out[i] = Var "acc"
```

### Host plan

In this exact order:
1. `Alloc` every buffer: Params first in `Graph.params` order, then the
   other materialised nodes in topological order.
2. `Upload { param = name; into = buffer }` for each Param, same order.
3. `Launch { kernel = k.name; args = k.params }` for each kernel, in
   `kernel_roots` order.
4. `Download { from = buffer_of node; output = name; shape = Tensor.shape node }`
   for each `(name, node)` in `Graph.outputs`, in that order. An output
   that is a Param downloads from the param buffer.
5. `Free` every buffer, in Alloc order.

`program = { name = Graph.name g; kernels; plan }`.

## Invariants

- The kernel list and the Launch ops are in the same order, and every
  kernel's inputs were produced by an earlier kernel or are Params.
- Every `Load` in a kernel refers to a buffer in that kernel's `params`
  or `shared`. Never load a non-materialised node.
- Index arithmetic is `int`. Do not emit `int64` indices.
- `Lower.program` is deterministic and does not mutate the graph.

## Failure modes to avoid

- Calling `elem` on the root (gives `Load out[i]` = reads the output).
- Materialising a node yourself because it seemed convenient. Only
  `Fusion` decides; if you disagree, report it.
- Forgetting `Sync_threads` between reduction steps, or putting the
  final `Store out[0]` inside the loop.
- Using `Schedule.grid_stride` for reduce (must be `single_block`: the
  tree assumes exactly one block).
- Reduce kernel loop `lo = Global_thread_id`; with one block that is the
  same value as `Local_thread_id`, but the tests match on `Local_thread_id`
  semantics through the shape checks — use `Local_thread_id`/`Block_dim`.
- Non-`type a.` `expr_of`: the `Cmp` case will not type-check.
- Duplicate params when the same input feeds a kernel twice (dedupe by uid).

## Verify

```
dune build 2>&1 && dune exec test/unit/test_lower.exe
```

Expected: `11 tests, 0 failures`.

## Tests (already written: `test/unit/test_lower.ml`)

grid_stride geometry (0, 1, 256, 257, 1000, 10M); saxpy → 2 kernels, map
kernel params `[p_x; p_y; out]`, body is exactly one grid-stride For with
one Store, grid 4, only params/out touched; reduce kernel: grid 1, block
256, one shared buffer of 256, has Sync_threads, reads the map's output;
host plan counts and ordering, download names `[r; s]`, upload names
`[x; y]`, download numel matches shape; iota+cast program has one param
and no uploads; chain of five maps → 1 kernel; fanout → 4 kernels, 5
allocs; scan launch is 1×1; Param-as-output → 0 kernels, 1 download;
kernel names unique C identifiers matching Launch order; every Launch arg
allocated earlier.
