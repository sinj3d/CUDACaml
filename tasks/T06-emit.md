# T06 — `Mangle` + `Emit`: Kernel_ir → CUDA C++ text

## Goal

Implement `lib/backend_cuda/mangle.ml` and `lib/backend_cuda/emit.ml`.
`Emit.program` prints a `Kernel_ir.program` as one CUDA C++ translation
unit that NVRTC can compile **with no `#include`s**. It is a pure
pretty-printer: it makes no decisions. The output must be byte-for-byte
deterministic for the same input.

Depends on: T05 (tests lower the example programs and print them).

## Files you own

- `lib/backend_cuda/mangle.ml`
- `lib/backend_cuda/emit.ml`

Do not edit the `.mli`s or `kernel_ir.ml`.

## Interfaces

`lib/backend_cuda/emit.mli`:

```ocaml
val program : Kernel_ir.program -> string
val expr : Kernel_ir.expr -> string
val stmt : indent:int -> Kernel_ir.stmt -> string   (* indent = number of 2-space levels *)
val kernel : Kernel_ir.kernel -> string
```

`lib/backend_cuda/mangle.mli`:

```ocaml
type t
val create : unit -> t
val identifier : t -> string -> string   (* stable: same input, same output *)
```

Input type: `Kernel_ir` (verbatim in T05 — read `lib/lower/kernel_ir.ml`).
`Expr.binop/unop/cmp/logic` constructors are listed in T03.
`Dtype.packed = P : _ Dtype.t -> packed`.

## Implementation — Mangle

```ocaml
type t = { seen : (string, string) Hashtbl.t; used : (string, unit) Hashtbl.t }
```
`identifier t s`:
1. If `s` is in `seen`, return the stored result (stability).
2. Replace every char that is not `[A-Za-z0-9_]` with `_`. If the result
   is empty or starts with a digit, prefix `_`.
3. If the result is a C/CUDA reserved word, append `_`. Reserved list must
   include at least: `auto break case char const continue default do double
   else enum extern float for goto if int long register return short signed
   sizeof static struct switch typedef union unsigned void volatile while
   bool true false class namespace template this new delete inline`.
4. While the result is in `used`, append `_` (uniqueness within a
   translation unit; different inputs never collide).
5. Record in `seen` and `used`; return.

## Implementation — Emit

### Types

```
F32 → "float"   F64 → "double"   I32 → "int"   I64 → "long long"   Bool → "bool"
```

### Literals (`Lit (l, P d)`)

- `F v` with `d = F32`: `Printf.sprintf "%.9g" v`, then if the string
  contains none of `.`, `e`, `n` (nan/inf) append `.0`; then append `f`.
  → `2.0f`, `0.5f`, `-1e+30f`, `1.5e-07f`.
- `F v` with `d = F64`: same without the `f`. (If `d` is not a float type
  still print as F32 form; it will not happen.)
- NaN/infinite floats: emit `(0.0f/0.0f)`, `(1.0f/0.0f)`, `(-1.0f/0.0f)`
  (drop the `f` for F64). Do this **before** the `%.9g` path.
- `I v` with `d = I64`: `Printf.sprintf "%LdLL" v`. With `d = I32` (or
  anything else): `Printf.sprintf "%Ld" v`.
- `B true`/`B false` → `true`/`false`.

### Expressions (`expr`) — every compound form is wrapped in parentheses

| Node | Output |
|---|---|
| `Var v` | `v` |
| `Global_thread_id` | `(blockIdx.x * blockDim.x + threadIdx.x)` |
| `Global_size` | `(gridDim.x * blockDim.x)` |
| `Local_thread_id` | `threadIdx.x` |
| `Block_dim` | `blockDim.x` |
| `Load {buf; index}` | `buf.name[index]` → `x[i]` (no outer parens) |
| `Binop (_, Add, a, b)` | `(a + b)`; Sub `-`, Mul `*`, Div `/` |
| `Binop (P F32, Min, a, b)` | `fminf(a, b)`; Max → `fmaxf` |
| `Binop (P F64, Min/Max, ...)` | `fmin(a, b)` / `fmax(a, b)` |
| `Binop (int or bool, Min, a, b)` | `((a < b) ? a : b)`; Max → `((a > b) ? a : b)` |
| `Unop (_, Neg, a)` | `(-a)` |
| `Unop (P F32, Sqrt/Exp/Log/Abs, a)` | `sqrtf(a)` `expf(a)` `logf(a)` `fabsf(a)` |
| `Unop (P F64, ...)` | `sqrt(a)` `exp(a)` `log(a)` `fabs(a)` |
| `Unop (int, Abs, a)` | `((a < 0) ? (-a) : a)`; Sqrt/Exp/Log on ints: `failwith` |
| `Cmp (Eq/Ne/Lt/Le/Gt/Ge, a, b)` | `(a == b)` `(a != b)` `(a < b)` `(a <= b)` `(a > b)` `(a >= b)` |
| `Logic (And/Or, a, b)` | `(a && b)` / `(a \|\| b)` |
| `Not a` | `(!a)` |
| `Select (c, a, b)` | `(c ? a : b)` |
| `Cast (P d, a)` | `((T)(a))` with T from the type table → `((float)(a))` |

`a`, `b`, `c` above are the recursive results.

### Statements (`stmt ~indent`) — each line ends with `\n`, indented by `2*indent` spaces

- `Let {var; dtype; value}` → `T var = e;`
- `Assign {var; value}` → `var = e;`
- `Store {buf; index; value}` → `buf[idx] = e;`
- `For {var; lo; hi; step; body}` →
  ```
  for (int var = lo; var < hi; var += step) {
    ...body at indent+1
  }
  ```
- `If {cond; then_; else_}` → `if (c) {` … `}` and, only when `else_ <> []`,
  `} else {` … `}`.
- `Sync_threads` → `__syncthreads();`

### Kernel (`kernel`)

```
extern "C" __global__ void NAME(T1* p1, T2* p2, ..., Tn* out) {
  __shared__ T sdata[256];        // one line per entry in k.shared
  ...body at indent 1
}
```
Every param is a plain pointer `T* name` (no `const`, no `__restrict__`
in v1 — simplicity over speed). **`extern "C"` is mandatory**; without it
the name is C++-mangled and `Jit.get_kernel` cannot find it.

Pass buffer and kernel names through **one** `Mangle.t` per `program`
call. Since T05 already produces valid identifiers this normally changes
nothing, but it guarantees a user param named `float` cannot break
compilation.

### Program (`program`)

```
// generated by ocaml-cuda: <program.name>
<kernel 1>

<kernel 2>
```
No includes, no `using`, nothing else. NVRTC supplies `threadIdx`,
`__syncthreads`, `fminf`, `sqrtf` etc. as builtins.

## Invariants

- Parentheses balance in every output (tests count them).
- Zero decisions: no constant folding, no reordering, no "optimising".
- Deterministic: no Hashtbl iteration in output order.

## Failure modes to avoid

- `2f` for a float literal (invalid C). Always ensure a `.` or exponent.
- `1e30f` → fine; `1e+30f` → fine; `1E30f` → fine. `nanf` → not fine
  without a header. Use the division forms above.
- Emitting `min(`/`max(`: those need `<algorithm>`; use `fminf`/ternary.
- Forgetting the `f` suffix on F32 literals — the kernel silently does
  double math and mismatches the interpreter.
- `long` instead of `long long` for I64 (`long` is 32-bit on Windows).
- Adding `#include <cuda_runtime.h>`: NVRTC has no default include path.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_emit.exe
```

Expected: `11 tests, 0 failures`.

## Tests (already written: `test/unit/test_emit.ml`)

float literal forms (`2.0f`, `0.5f`, `2.0`, `-1e+30`); int/bool literals
(`-7`, `LL` suffix, `true`, `false`); builtins text; nested binop fully
parenthesised `((a + b) * c)`; min/max intrinsic selection by dtype;
unary intrinsics by dtype and `(-a)`; cmp/logic/not/select/cast/load exact
strings; statements contain the expected fragments; kernel header has
`extern "C" __global__ void k_1(`, pointer params, `__shared__ float sdata[256];`,
balanced parens; every lowered example prints every kernel and balances;
mangle handles reserved words, illegal chars, stability, distinctness.
