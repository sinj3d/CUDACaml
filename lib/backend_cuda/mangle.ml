(* Identifier legalisation for the C++ target.

   Two tables: [seen] gives stability (the same input always maps to the
   same output) and [used] gives uniqueness within one translation unit
   (two different inputs never map to the same output). Neither table is
   ever iterated, so nothing about hash order can reach the output. *)

type t = { seen : (string, string) Hashtbl.t; used : (string, unit) Hashtbl.t }

(* C, C++ and CUDA words a generated identifier must not be equal to. The
   device builtins are in here too: a parameter called [threadIdx] would
   shadow the builtin and break every kernel that uses it. *)
let reserved =
  [
    (* C *)
    "auto";
    "break";
    "case";
    "char";
    "const";
    "continue";
    "default";
    "do";
    "double";
    "else";
    "enum";
    "extern";
    "float";
    "for";
    "goto";
    "if";
    "int";
    "long";
    "register";
    "restrict";
    "return";
    "short";
    "signed";
    "sizeof";
    "static";
    "struct";
    "switch";
    "typedef";
    "union";
    "unsigned";
    "void";
    "volatile";
    "while";
    (* C++ *)
    "bool";
    "true";
    "false";
    "class";
    "namespace";
    "template";
    "typename";
    "this";
    "new";
    "delete";
    "inline";
    "operator";
    "private";
    "protected";
    "public";
    "using";
    "virtual";
    (* CUDA builtins and qualifiers *)
    "blockDim";
    "blockIdx";
    "gridDim";
    "threadIdx";
    "warpSize";
    "__shared__";
    "__global__";
    "__device__";
    "__host__";
    "__syncthreads";
  ]

let is_reserved s = List.mem s reserved

let is_legal_char c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c = '_'

let is_digit c = c >= '0' && c <= '9'
let create () = { seen = Hashtbl.create 64; used = Hashtbl.create 64 }

let identifier t s =
  match Hashtbl.find_opt t.seen s with
  | Some legal -> legal
  | None ->
      let base = String.map (fun c -> if is_legal_char c then c else '_') s in
      let base =
        if String.length base = 0 then "_"
        else if is_digit base.[0] then "_" ^ base
        else base
      in
      let base = if is_reserved base then base ^ "_" else base in
      let rec uniquify candidate =
        if Hashtbl.mem t.used candidate then uniquify (candidate ^ "_") else candidate
      in
      let legal = uniquify base in
      Hashtbl.replace t.seen s legal;
      Hashtbl.replace t.used legal ();
      legal
