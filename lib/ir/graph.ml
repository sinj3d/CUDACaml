(** A whole program: named parameters in, named tensors out.

    Everything is computed once in [create] and stored, because every layer
    above walks [topological_order] and asks [fan_out] repeatedly.

    Node identity is the [Uid] and nothing else: a [Tensor.packed] wraps a
    whole GADT tree, so structural comparison ([=], [List.mem], [List.assoc],
    [Hashtbl.hash]) would be both wrong and slow. Every table here is keyed
    by [Uid.to_int]. No hash-table iteration order reaches any output: all
    output is produced by walking [order]. *)

type t = {
  name : string;
  outputs : (string * Tensor.packed) list;
  params : (string * Tensor.packed) list;
  order : Tensor.packed list;
  fan_out : (int, int) Hashtbl.t;  (** keyed by [Uid.to_int]; missing = 0 *)
}

let key (p : Tensor.packed) = Uid.to_int (Tensor.uid p)

(* Opening the existential is the only way to look at a node's constructor. *)
let param_name (Tensor.P t) = match t.node with Tensor.Param n -> Some n | _ -> None

let node_kind (Tensor.P t) =
  match t.node with
  | Tensor.Param n -> "Param " ^ n
  | Tensor.Iota -> "Iota"
  | Tensor.Map _ -> "Map"
  | Tensor.Map2 _ -> "Map2"
  | Tensor.Reduce _ -> "Reduce"
  | Tensor.Scan _ -> "Scan"
  | Tensor.Gather _ -> "Gather"
  | Tensor.Reshape _ -> "Reshape"
  | Tensor.Broadcast _ -> "Broadcast"

let dtype_name (Tensor.P t) = Dtype.name t.dtype

let check_output_names outputs =
  let seen = Hashtbl.create 16 in
  List.iter
    (fun (n, _) ->
      if Hashtbl.mem seen n then
        invalid_arg (Printf.sprintf "Graph.create: duplicate output name %S" n);
      Hashtbl.add seen n ())
    outputs

(* Depth-first POST-order: a node is appended only after all of its
   dependencies, so deps always precede dependents. Outputs are visited in
   list order and deps in argument order, which makes the result stable. *)
let topo_sort outputs =
  let visited = Hashtbl.create 64 in
  let acc = ref [] in
  let rec visit p =
    let k = key p in
    if not (Hashtbl.mem visited k) then begin
      Hashtbl.add visited k ();
      List.iter visit (Tensor.deps p);
      acc := p :: !acc
    end
  in
  List.iter (fun (_, p) -> visit p) outputs;
  List.rev !acc

(* Counts USES (edges), not distinct consumers: [Map2 (f, s, s)] contributes
   2 to [s]. Program outputs are not uses. *)
let count_fan_out order =
  let tbl = Hashtbl.create 64 in
  List.iter
    (fun p ->
      List.iter
        (fun d ->
          let k = key d in
          let n = match Hashtbl.find_opt tbl k with Some n -> n | None -> 0 in
          Hashtbl.replace tbl k (n + 1))
        (Tensor.deps p))
    order;
  tbl

let collect_params order =
  let params =
    List.filter_map (fun p -> Option.map (fun n -> (n, p)) (param_name p)) order
  in
  let seen = Hashtbl.create 16 in
  List.iter
    (fun (n, p) ->
      match Hashtbl.find_opt seen n with
      | Some u when u <> key p ->
          invalid_arg (Printf.sprintf "Graph.create: two Param nodes named %s" n)
      | Some _ -> ()
      | None -> Hashtbl.add seen n (key p))
    params;
  params

let create ~name ~outputs =
  (match outputs with
  | [] -> invalid_arg "Graph.create: a graph needs at least one output"
  | _ :: _ -> ());
  check_output_names outputs;
  let order = topo_sort outputs in
  { name; outputs; params = collect_params order; order; fan_out = count_fan_out order }

let name t = t.name
let params t = t.params
let outputs t = t.outputs
let topological_order t = t.order
let iter t ~f = List.iter f t.order
let fold t ~init ~f = List.fold_left f init t.order

let fan_out t p =
  match Hashtbl.find_opt t.fan_out (key p) with Some n -> n | None -> 0

let dot_escape s =
  let buf = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match c with
      | '"' | '\\' ->
          Buffer.add_char buf '\\';
          Buffer.add_char buf c
      | '\n' -> Buffer.add_string buf "\\n"
      | c -> Buffer.add_char buf c)
    s;
  Buffer.contents buf

let to_dot t =
  let buf = Buffer.create 1024 in
  Buffer.add_string buf (Printf.sprintf "digraph \"%s\" {\n" (dot_escape t.name));
  List.iter
    (fun p ->
      let u = Uid.to_string (Tensor.uid p) in
      let label =
        Printf.sprintf "%s %s %s %s" u (node_kind p)
          (Shape.to_string (Tensor.shape p))
          (dtype_name p)
      in
      Buffer.add_string buf (Printf.sprintf "  %s [label=\"%s\"];\n" u (dot_escape label)))
    t.order;
  List.iter
    (fun p ->
      let dst = Uid.to_string (Tensor.uid p) in
      List.iter
        (fun d -> Buffer.add_string buf
            (Printf.sprintf "  %s -> %s;\n" (Uid.to_string (Tensor.uid d)) dst))
        (Tensor.deps p))
    t.order;
  Buffer.add_string buf "}\n";
  Buffer.contents buf
