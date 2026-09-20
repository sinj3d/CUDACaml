(* Kernel_ir -> CUDA C++ text. A pure pretty-printer: every construct has
   exactly one spelling, every compound expression is parenthesised, and no
   decision (folding, reordering, scheduling) is taken here. No [#include]
   is emitted: NVRTC has no default include path and supplies [threadIdx],
   [__syncthreads], [fminf], [sqrtf] and friends as builtins. *)

module Dtype = Ocaml_cuda_ir.Dtype
module Expr = Ocaml_cuda_ir.Expr
module K = Ocaml_cuda_lower.Kernel_ir

(* Which C spelling family a dtype belongs to: it selects the math
   intrinsic and the float literal suffix. *)
type cls = Float32 | Float64 | Integral

let classify (d : Dtype.packed) : cls =
  match d with
  | Dtype.P Dtype.F32 -> Float32
  | Dtype.P Dtype.F64 -> Float64
  | Dtype.P Dtype.I32 | Dtype.P Dtype.I64 | Dtype.P Dtype.Bool -> Integral

(* [long long], never [long]: [long] is 32-bit on Windows. *)
let ctype (d : Dtype.packed) : string =
  match d with
  | Dtype.P Dtype.F32 -> "float"
  | Dtype.P Dtype.F64 -> "double"
  | Dtype.P Dtype.I32 -> "int"
  | Dtype.P Dtype.I64 -> "long long"
  | Dtype.P Dtype.Bool -> "bool"

(* A float literal must carry an [f] suffix for f32 (or the kernel silently
   does double math) and must contain a [.] or an exponent ([2f] is not
   valid C). NaN and the infinities are spelled as divisions because
   [nanf]/[INFINITY] need a header. *)
let float_literal (d : Dtype.packed) (v : float) : string =
  let sfx = match classify d with Float64 -> "" | Float32 | Integral -> "f" in
  if Float.is_nan v then Printf.sprintf "(0.0%s/0.0%s)" sfx sfx
  else if v = Float.infinity then Printf.sprintf "(1.0%s/0.0%s)" sfx sfx
  else if v = Float.neg_infinity then Printf.sprintf "(-1.0%s/0.0%s)" sfx sfx
  else
    let s = Printf.sprintf "%.9g" v in
    let s =
      if String.exists (fun c -> c = '.' || c = 'e' || c = 'n') s then s else s ^ ".0"
    in
    s ^ sfx

let int_literal (d : Dtype.packed) (v : int64) : string =
  match d with
  | Dtype.P Dtype.I64 -> Printf.sprintf "%LdLL" v
  | Dtype.P Dtype.F32 | Dtype.P Dtype.F64 | Dtype.P Dtype.I32 | Dtype.P Dtype.Bool ->
      Printf.sprintf "%Ld" v

let literal (l : K.literal) (d : Dtype.packed) : string =
  match l with
  | K.F v -> float_literal d v
  | K.I v -> int_literal d v
  | K.B b -> if b then "true" else "false"

(* Bitwise operators are integer only. [bool] is an integral type in C and
   [&]/[|]/[^] are well defined on it, so it is let through. *)
let bitwise (d : Dtype.packed) ~sym ~name (a : string) (b : string) : string =
  match classify d with
  | Integral -> Printf.sprintf "(%s %s %s)" a sym b
  | Float32 | Float64 ->
      failwith (Printf.sprintf "Emit: %s on a float dtype" name)

(* Shifts go through the unsigned type of the same width: a left shift that
   overflows a signed [int] is undefined, and [>>] on a negative signed
   value is implementation-defined (an arithmetic shift in practice). The
   count is masked to the bit width so that no count is undefined either --
   the interpreter masks the same way, which is what keeps the two bit
   identical. *)
let shift (d : Dtype.packed) ~sym ~name (a : string) (b : string) : string =
  match d with
  | Dtype.P Dtype.I32 ->
      Printf.sprintf "((int)((unsigned int)(%s) %s ((%s) & 31)))" a sym b
  | Dtype.P Dtype.I64 ->
      Printf.sprintf "((long long)((unsigned long long)(%s) %s ((%s) & 63)))" a sym b
  | Dtype.P Dtype.Bool -> failwith (Printf.sprintf "Emit: %s on bool" name)
  | Dtype.P Dtype.F32 | Dtype.P Dtype.F64 ->
      failwith (Printf.sprintf "Emit: %s on a float dtype" name)

(* [a] and [b] are already-printed operands. Never [min(]/[max(]: those
   need <algorithm>. *)
let binop (d : Dtype.packed) (op : Expr.binop) (a : string) (b : string) : string =
  match op with
  | Expr.Add -> Printf.sprintf "(%s + %s)" a b
  | Expr.Sub -> Printf.sprintf "(%s - %s)" a b
  | Expr.Mul -> Printf.sprintf "(%s * %s)" a b
  | Expr.Div -> Printf.sprintf "(%s / %s)" a b
  | Expr.Min -> (
      match classify d with
      | Float32 -> Printf.sprintf "fminf(%s, %s)" a b
      | Float64 -> Printf.sprintf "fmin(%s, %s)" a b
      | Integral -> Printf.sprintf "((%s < %s) ? %s : %s)" a b a b)
  | Expr.Max -> (
      match classify d with
      | Float32 -> Printf.sprintf "fmaxf(%s, %s)" a b
      | Float64 -> Printf.sprintf "fmax(%s, %s)" a b
      | Integral -> Printf.sprintf "((%s > %s) ? %s : %s)" a b a b)
  | Expr.Bit_and -> bitwise d ~sym:"&" ~name:"bit_and" a b
  | Expr.Bit_or -> bitwise d ~sym:"|" ~name:"bit_or" a b
  | Expr.Bit_xor -> bitwise d ~sym:"^" ~name:"bit_xor" a b
  | Expr.Shl -> shift d ~sym:"<<" ~name:"shl" a b
  | Expr.Shr -> shift d ~sym:">>" ~name:"shr" a b

(* Unary minus of an operand that already starts with a sign needs a space:
   [(--7)] lexes as the C decrement operator and does not compile. *)
let neg (a : string) : string =
  if String.length a > 0 && (a.[0] = '-' || a.[0] = '+') then Printf.sprintf "(- %s)" a
  else Printf.sprintf "(-%s)" a

let unop (d : Dtype.packed) (op : Expr.unop) (a : string) : string =
  let math ~f32 ~f64 name =
    match classify d with
    | Float32 -> Printf.sprintf "%s(%s)" f32 a
    | Float64 -> Printf.sprintf "%s(%s)" f64 a
    | Integral -> failwith (Printf.sprintf "Emit: %s on a non-float dtype" name)
  in
  match op with
  | Expr.Neg -> neg a
  | Expr.Sqrt -> math ~f32:"sqrtf" ~f64:"sqrt" "sqrt"
  | Expr.Exp -> math ~f32:"expf" ~f64:"exp" "exp"
  | Expr.Log -> math ~f32:"logf" ~f64:"log" "log"
  | Expr.Abs -> (
      match classify d with
      | Float32 -> Printf.sprintf "fabsf(%s)" a
      | Float64 -> Printf.sprintf "fabs(%s)" a
      | Integral -> Printf.sprintf "((%s < 0) ? %s : %s)" a (neg a) a)
  | Expr.Sin -> math ~f32:"sinf" ~f64:"sin" "sin"
  | Expr.Cos -> math ~f32:"cosf" ~f64:"cos" "cos"
  (* The [f] suffix is not decoration: [erf] on a float argument is the
     double routine and silently costs twice as much. *)
  | Expr.Erf -> math ~f32:"erff" ~f64:"erf" "erf"
  | Expr.Erfinv -> math ~f32:"erfinvf" ~f64:"erfinv" "erfinv"

let cmp_op (c : Expr.cmp) : string =
  match c with
  | Expr.Eq -> "=="
  | Expr.Ne -> "!="
  | Expr.Lt -> "<"
  | Expr.Le -> "<="
  | Expr.Gt -> ">"
  | Expr.Ge -> ">="

let logic_op (l : Expr.logic) : string =
  match l with Expr.And -> "&&" | Expr.Or -> "||"

(* Every compound form is parenthesised; [Load] is the one exception, since
   [x[i]] already binds tighter than anything around it. *)
let rec expr (e : K.expr) : string =
  match e with
  | K.Var v -> v
  | K.Lit (l, d) -> literal l d
  | K.Global_thread_id -> "(blockIdx.x * blockDim.x + threadIdx.x)"
  | K.Global_size -> "(gridDim.x * blockDim.x)"
  | K.Local_thread_id -> "threadIdx.x"
  | K.Block_id -> "blockIdx.x"
  | K.Block_dim -> "blockDim.x"
  | K.Load { buf; index } -> Printf.sprintf "%s[%s]" buf.K.name (expr index)
  | K.Binop (d, op, a, b) ->
      let sa = expr a in
      let sb = expr b in
      binop d op sa sb
  | K.Unop (d, op, a) -> unop d op (expr a)
  | K.Cmp (c, a, b) ->
      let sa = expr a in
      let sb = expr b in
      Printf.sprintf "(%s %s %s)" sa (cmp_op c) sb
  | K.Logic (l, a, b) ->
      let sa = expr a in
      let sb = expr b in
      Printf.sprintf "(%s %s %s)" sa (logic_op l) sb
  | K.Not a -> Printf.sprintf "(!%s)" (expr a)
  | K.Select (c, a, b) ->
      let sc = expr c in
      let sa = expr a in
      let sb = expr b in
      Printf.sprintf "(%s ? %s : %s)" sc sa sb
  | K.Cast (d, a) -> Printf.sprintf "((%s)(%s))" (ctype d) (expr a)

let rec stmt ~indent (s : K.stmt) : string =
  let pad = String.make (2 * indent) ' ' in
  match s with
  | K.Let { var; dtype; value } ->
      Printf.sprintf "%s%s %s = %s;\n" pad (ctype dtype) var (expr value)
  | K.Assign { var; value } -> Printf.sprintf "%s%s = %s;\n" pad var (expr value)
  | K.Store { buf; index; value } ->
      let si = expr index in
      let sv = expr value in
      Printf.sprintf "%s%s[%s] = %s;\n" pad buf.K.name si sv
  (* CUDA supplies [atomicAdd] for [float], [double] (sm_60+) and [int].
     [unsigned long long] has an overload but plain [long long] does not,
     and [bool] is not addressable as an atomic at all, so both are refused
     here rather than emitting something that fails inside NVRTC. *)
  | K.Atomic_add { buf; index; value } ->
      let si = expr index in
      let sv = expr value in
      (match buf.K.dtype with
      | Dtype.P Dtype.F32 | Dtype.P Dtype.F64 | Dtype.P Dtype.I32 -> ()
      | Dtype.P Dtype.I64 -> failwith "Emit: atomicAdd on an i64 buffer"
      | Dtype.P Dtype.Bool -> failwith "Emit: atomicAdd on a bool buffer");
      Printf.sprintf "%satomicAdd(&%s[%s], %s);\n" pad buf.K.name si sv
  | K.For { var; lo; hi; step; body } ->
      let slo = expr lo in
      let shi = expr hi in
      let sstep = expr step in
      Printf.sprintf "%sfor (int %s = %s; %s < %s; %s += %s) {\n%s%s}\n" pad var slo var shi
        var sstep
        (block ~indent:(indent + 1) body)
        pad
  | K.If { cond; then_; else_ } -> (
      let head =
        Printf.sprintf "%sif (%s) {\n%s" pad (expr cond) (block ~indent:(indent + 1) then_)
      in
      match else_ with
      | [] -> Printf.sprintf "%s%s}\n" head pad
      | _ :: _ ->
          Printf.sprintf "%s%s} else {\n%s%s}\n" head pad
            (block ~indent:(indent + 1) else_)
            pad)
  | K.Sync_threads -> Printf.sprintf "%s__syncthreads();\n" pad

and block ~indent (stmts : K.stmt list) : string =
  String.concat "" (List.map (stmt ~indent) stmts)

(* [extern "C"] is mandatory: without it the symbol is C++-mangled and
   [Jit.get_kernel] cannot find it by name. Parameters are plain pointers
   (no const, no __restrict__ in v1). *)
let kernel (k : K.kernel) : string =
  let param (b : K.buffer) = Printf.sprintf "%s* %s" (ctype b.K.dtype) b.K.name in
  let shared (b : K.buffer) =
    Printf.sprintf "  __shared__ %s %s[%d];\n" (ctype b.K.dtype) b.K.name b.K.numel
  in
  let params = String.concat ", " (List.map param k.K.params) in
  let shared = String.concat "" (List.map shared k.K.shared) in
  Printf.sprintf "extern \"C\" __global__ void %s(%s) {\n%s%s}\n" k.K.name params shared
    (block ~indent:1 k.K.body)

(* Name legalisation runs in two passes so that nothing about the order in
   which OCaml happens to evaluate arguments can reach the output: first
   every name is registered with the mangler in one fixed traversal order,
   then the tree is rewritten with pure (stable) lookups. *)
let rec collect_expr (acc : string list) (e : K.expr) : string list =
  match e with
  | K.Var _ | K.Lit _ | K.Global_thread_id | K.Global_size | K.Local_thread_id
  | K.Block_id | K.Block_dim ->
      acc
  | K.Load { buf; index } -> collect_expr (buf.K.name :: acc) index
  | K.Binop (_, _, a, b) -> collect_expr (collect_expr acc a) b
  | K.Unop (_, _, a) -> collect_expr acc a
  | K.Cmp (_, a, b) -> collect_expr (collect_expr acc a) b
  | K.Logic (_, a, b) -> collect_expr (collect_expr acc a) b
  | K.Not a -> collect_expr acc a
  | K.Select (c, a, b) -> collect_expr (collect_expr (collect_expr acc c) a) b
  | K.Cast (_, a) -> collect_expr acc a

let rec collect_stmt (acc : string list) (s : K.stmt) : string list =
  match s with
  | K.Let { var = _; dtype = _; value } -> collect_expr acc value
  | K.Assign { var = _; value } -> collect_expr acc value
  | K.Store { buf; index; value } | K.Atomic_add { buf; index; value } ->
      collect_expr (collect_expr (buf.K.name :: acc) index) value
  | K.For { var = _; lo; hi; step; body } ->
      let acc = collect_expr (collect_expr (collect_expr acc lo) hi) step in
      List.fold_left collect_stmt acc body
  | K.If { cond; then_; else_ } ->
      let acc = collect_expr acc cond in
      let acc = List.fold_left collect_stmt acc then_ in
      List.fold_left collect_stmt acc else_
  | K.Sync_threads -> acc

let collect_kernel (acc : string list) (k : K.kernel) : string list =
  let acc = k.K.name :: acc in
  let names acc (bs : K.buffer list) =
    List.fold_left (fun acc (b : K.buffer) -> b.K.name :: acc) acc bs
  in
  let acc = names acc k.K.params in
  let acc = names acc k.K.shared in
  List.fold_left collect_stmt acc k.K.body

let rename_buf m (b : K.buffer) : K.buffer = { b with K.name = Mangle.identifier m b.K.name }

let rec rename_expr m (e : K.expr) : K.expr =
  match e with
  | K.Var _ | K.Lit _ | K.Global_thread_id | K.Global_size | K.Local_thread_id
  | K.Block_id | K.Block_dim ->
      e
  | K.Load { buf; index } -> K.Load { buf = rename_buf m buf; index = rename_expr m index }
  | K.Binop (d, op, a, b) -> K.Binop (d, op, rename_expr m a, rename_expr m b)
  | K.Unop (d, op, a) -> K.Unop (d, op, rename_expr m a)
  | K.Cmp (c, a, b) -> K.Cmp (c, rename_expr m a, rename_expr m b)
  | K.Logic (l, a, b) -> K.Logic (l, rename_expr m a, rename_expr m b)
  | K.Not a -> K.Not (rename_expr m a)
  | K.Select (c, a, b) -> K.Select (rename_expr m c, rename_expr m a, rename_expr m b)
  | K.Cast (d, a) -> K.Cast (d, rename_expr m a)

let rec rename_stmt m (s : K.stmt) : K.stmt =
  match s with
  | K.Let { var; dtype; value } -> K.Let { var; dtype; value = rename_expr m value }
  | K.Assign { var; value } -> K.Assign { var; value = rename_expr m value }
  | K.Store { buf; index; value } ->
      K.Store
        { buf = rename_buf m buf; index = rename_expr m index; value = rename_expr m value }
  | K.Atomic_add { buf; index; value } ->
      K.Atomic_add
        { buf = rename_buf m buf; index = rename_expr m index; value = rename_expr m value }
  | K.For { var; lo; hi; step; body } ->
      K.For
        {
          var;
          lo = rename_expr m lo;
          hi = rename_expr m hi;
          step = rename_expr m step;
          body = List.map (rename_stmt m) body;
        }
  | K.If { cond; then_; else_ } ->
      K.If
        {
          cond = rename_expr m cond;
          then_ = List.map (rename_stmt m) then_;
          else_ = List.map (rename_stmt m) else_;
        }
  | K.Sync_threads -> K.Sync_threads

let rename_kernel m (k : K.kernel) : K.kernel =
  {
    k with
    K.name = Mangle.identifier m k.K.name;
    params = List.map (rename_buf m) k.K.params;
    shared = List.map (rename_buf m) k.K.shared;
    body = List.map (rename_stmt m) k.K.body;
  }

let program (p : K.program) : string =
  let m = Mangle.create () in
  let names = List.rev (List.fold_left collect_kernel [] p.K.kernels) in
  List.iter (fun n -> ignore (Mangle.identifier m n : string)) names;
  let kernels = List.map (rename_kernel m) p.K.kernels in
  Printf.sprintf "// generated by ocaml-cuda: %s\n%s" p.K.name
    (String.concat "\n" (List.map kernel kernels))
