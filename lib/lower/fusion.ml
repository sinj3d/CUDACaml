(** Fusion, as an analysis rather than a rewrite.

    The whole pass is one stable walk of [Graph.topological_order] that
    answers a single question per node: does it need its own global buffer?
    [Lower] inlines every node that does not.

    Node identity is the [Uid] and nothing else: a [Tensor.packed] wraps a
    whole GADT tree, so structural comparison ([=], [List.mem],
    [Hashtbl.hash]) would be both wrong and slow. Every table here is keyed
    by [Uid.to_int], and no hash-table iteration order reaches an output:
    [roots] is built by walking the topological order. *)

open Ocaml_cuda_ir

type plan = { materialized : (int, unit) Hashtbl.t; roots : Tensor.packed list }

let key (p : Tensor.packed) = Uid.to_int (Tensor.uid p)

(* Opening the existential is the only way to look at a node's constructor. *)
let is_param (Tensor.P t) = match t.node with Tensor.Param _ -> true | _ -> false

(* Fusion barriers: the result of a [Reduce]/[Scan] is produced by a whole
   kernel co-operating, not by one thread per element, so it cannot be
   inlined into a consumer's element expression. [Scatter_add] is a barrier
   for the same reason from the other side: its output element [i] is the
   sum of an unknown set of source elements, so there is no per-element
   expression to inline, and it is written by two kernels (zero-fill, then
   atomics) that must both have finished before a consumer reads it. Kept as
   its own rule even where an output or fan-out would also force a buffer.
   [Matmul] is a barrier for the same reason as [Reduce]: one output
   element is a whole inner product, computed by a co-operating tile of
   threads, so there is no per-element expression a consumer could inline.
   Its OPERANDS are not special -- a map feeding a [Matmul] still inlines,
   into the tile loads.

   Everything else -- [Map], [Map2], [Iota], [Gather], [Reshape],
   [Broadcast] -- is element-wise: one thread computes one output element,
   so it inlines and only the generic rules (output, fan-out >= 2) can
   force it into a buffer. [Broadcast] inlines to a load at index 0, which
   is why it is not a barrier even though its source is often a [Reduce]. *)
let is_barrier (Tensor.P t) =
  match t.node with
  | Tensor.Reduce _ | Tensor.Scan _ | Tensor.Scatter_add _ | Tensor.Matmul _ -> true
  | _ -> false

let output_uids g =
  let tbl = Hashtbl.create 16 in
  List.iter (fun (_, p) -> Hashtbl.replace tbl (key p) ()) (Graph.outputs g);
  tbl

(* The four rules are independent: a node satisfying two of them is still
   just materialised once. [Graph.fan_out] counts USES (edges), so
   [map2 add s s] materialises [s]. *)
let needs_buffer g outs p =
  is_param p || is_barrier p || Hashtbl.mem outs (key p) || Graph.fan_out g p >= 2

let plan g =
  let outs = output_uids g in
  let materialized = Hashtbl.create 64 in
  let roots = ref [] in
  List.iter
    (fun p ->
      if needs_buffer g outs p then begin
        Hashtbl.replace materialized (key p) ();
        (* Params already live in a buffer: they are uploaded, not computed,
           so they never get a kernel. *)
        if not (is_param p) then roots := p :: !roots
      end)
    (Graph.topological_order g);
  { materialized; roots = List.rev !roots }

let is_materialized plan p = Hashtbl.mem plan.materialized (key p)
let kernel_roots plan = plan.roots
