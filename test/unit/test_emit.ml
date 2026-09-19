(* T06: Mangle + Emit. String-level checks; no GPU, no compiler. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir
module E = Backend_cuda.Emit

let f32 = Dtype.P Dtype.F32
let f64 = Dtype.P Dtype.F64
let i32 = Dtype.P Dtype.I32
let i64 = Dtype.P Dtype.I64
let lit_f32 v = K.Lit (K.F v, f32)
let buf name dtype numel = { K.name; dtype; memspace = K.Global; numel }

let balanced s =
  let d = ref 0 in
  String.iter (fun c -> if c = '(' then incr d else if c = ')' then decr d) s;
  !d = 0

let () =
  C.test "float literals always have a decimal point and f suffix for f32" (fun () ->
      C.string ~expect:"2.0f" (E.expr (lit_f32 2.0));
      C.string ~expect:"0.5f" (E.expr (lit_f32 0.5));
      C.string ~expect:"2.0" (E.expr (K.Lit (K.F 2.0, f64)));
      C.contains ~sub:"-1e+30" (E.expr (lit_f32 (-1e30))));
  C.test "int and bool literals" (fun () ->
      C.contains ~sub:"-7" (E.expr (K.Lit (K.I (-7L), i32)));
      C.contains ~sub:"LL" (E.expr (K.Lit (K.I 5L, i64)));
      C.string ~expect:"true" (E.expr (K.Lit (K.B true, f32)));
      C.string ~expect:"false" (E.expr (K.Lit (K.B false, f32))));
  C.test "builtins" (fun () ->
      C.contains ~sub:"blockIdx.x * blockDim.x + threadIdx.x" (E.expr K.Global_thread_id);
      C.contains ~sub:"gridDim.x * blockDim.x" (E.expr K.Global_size);
      C.string ~expect:"threadIdx.x" (E.expr K.Local_thread_id);
      C.string ~expect:"blockDim.x" (E.expr K.Block_dim));
  C.test "binops are fully parenthesised" (fun () ->
      let e = K.Binop (f32, Expr.Mul, K.Binop (f32, Expr.Add, K.Var "a", K.Var "b"), K.Var "c") in
      C.string ~expect:"((a + b) * c)" (E.expr e));
  C.test "min/max use fminf/fmaxf for f32, fmin/fmax for f64, ternary for ints" (fun () ->
      C.contains ~sub:"fminf(" (E.expr (K.Binop (f32, Expr.Min, K.Var "a", K.Var "b")));
      C.contains ~sub:"fmax(" (E.expr (K.Binop (f64, Expr.Max, K.Var "a", K.Var "b")));
      let s = E.expr (K.Binop (i32, Expr.Min, K.Var "a", K.Var "b")) in
      C.contains ~sub:"?" s;
      C.contains ~sub:"<" s);
  C.test "unary math picks the f32 or f64 intrinsic" (fun () ->
      C.contains ~sub:"sqrtf(" (E.expr (K.Unop (f32, Expr.Sqrt, K.Var "a")));
      C.contains ~sub:"sqrt(" (E.expr (K.Unop (f64, Expr.Sqrt, K.Var "a")));
      C.contains ~sub:"expf(" (E.expr (K.Unop (f32, Expr.Exp, K.Var "a")));
      C.contains ~sub:"logf(" (E.expr (K.Unop (f32, Expr.Log, K.Var "a")));
      C.contains ~sub:"fabsf(" (E.expr (K.Unop (f32, Expr.Abs, K.Var "a")));
      C.string ~expect:"(-a)" (E.expr (K.Unop (f32, Expr.Neg, K.Var "a"))));
  C.test "compare, logic, not, select, cast, load" (fun () ->
      C.string ~expect:"(a < b)" (E.expr (K.Cmp (Expr.Lt, K.Var "a", K.Var "b")));
      C.string ~expect:"(a == b)" (E.expr (K.Cmp (Expr.Eq, K.Var "a", K.Var "b")));
      C.string ~expect:"(a && b)" (E.expr (K.Logic (Expr.And, K.Var "a", K.Var "b")));
      C.string ~expect:"(!a)" (E.expr (K.Not (K.Var "a")));
      C.string ~expect:"(c ? a : b)" (E.expr (K.Select (K.Var "c", K.Var "a", K.Var "b")));
      C.string ~expect:"((float)(a))" (E.expr (K.Cast (f32, K.Var "a")));
      C.string ~expect:"((long long)(a))" (E.expr (K.Cast (i64, K.Var "a")));
      C.string ~expect:"x[i]" (E.expr (K.Load { buf = buf "x" f32 4; index = K.Var "i" })));
  C.test "statements" (fun () ->
      C.contains ~sub:"float acc = 0.0f;" (E.stmt ~indent:0 (K.Let { var = "acc"; dtype = f32; value = lit_f32 0.0 }));
      C.contains ~sub:"acc = a;" (E.stmt ~indent:0 (K.Assign { var = "acc"; value = K.Var "a" }));
      C.contains ~sub:"out[i] = a;" (E.stmt ~indent:0 (K.Store { buf = buf "out" f32 4; index = K.Var "i"; value = K.Var "a" }));
      C.contains ~sub:"__syncthreads();" (E.stmt ~indent:0 K.Sync_threads);
      let f = E.stmt ~indent:0 (K.For { var = "i"; lo = K.Lit (K.I 0L, i32); hi = K.Var "n"; step = K.Lit (K.I 1L, i32); body = [ K.Sync_threads ] }) in
      C.contains ~sub:"for (int i = " f;
      C.contains ~sub:"i < n" f;
      C.contains ~sub:"i += " f;
      let s = E.stmt ~indent:0 (K.If { cond = K.Var "c"; then_ = [ K.Sync_threads ]; else_ = [] }) in
      C.contains ~sub:"if (c)" s);
  C.test "kernel signature is extern C, __global__, pointer params, shared decls" (fun () ->
      let k = { K.name = "k_1"; params = [ buf "p_x" f32 8; buf "out" f32 8 ]; shared = [ { K.name = "sdata"; dtype = f32; memspace = K.Shared; numel = 256 } ];
                body = [ K.Sync_threads ]; launch = Schedule.single_block } in
      let s = E.kernel k in
      C.contains ~sub:"extern \"C\" __global__ void k_1(" s;
      C.contains ~sub:"float* p_x" s;
      C.contains ~sub:"float* out" s;
      C.contains ~sub:"__shared__ float sdata[256];" s;
      C.bool ~expect:true (balanced s));
  C.test "program emits every kernel of a lowered example and balances parens" (fun () ->
      List.iter
        (fun (e : Ocaml_cuda_examples.Programs.t) ->
          let prog = Lower.program (e.graph ()) in
          let src = E.program prog in
          List.iter (fun (k : K.kernel) -> C.contains ~sub:("void " ^ k.name ^ "(") src) prog.kernels;
          C.bool ~expect:true (balanced src))
        Ocaml_cuda_examples.Programs.all);
  C.test "mangle: reserved words and illegal characters, stable" (fun () ->
      let m = Backend_cuda.Mangle.create () in
      let a = Backend_cuda.Mangle.identifier m "float" in
      C.bool ~expect:true (a <> "float");
      C.string ~expect:a (Backend_cuda.Mangle.identifier m "float");
      let b = Backend_cuda.Mangle.identifier m "my-name.1" in
      String.iter (fun c -> if not (c = '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) then C.fail "bad char %c" c) b;
      C.bool ~expect:true (b.[0] < '0' || b.[0] > '9');
      C.bool ~expect:true (Backend_cuda.Mangle.identifier m "x" <> Backend_cuda.Mangle.identifier m "y"));
  C.run ()
