# T18 — `Grad`: reverse-mode AD over the tensor DAG

## Goal

The headline feature. `Grad.grad g ~output ~wrt` returns a new graph that
computes everything `g` computed **plus** one gradient tensor per requested
parameter, by walking the DAG backwards and accumulating adjoints. It is a
graph-to-graph transform built entirely from `Dsl` calls, so it needs no
new node kinds (T13's `Broadcast`, T14's row forms, and later T19's
`Scatter_add` are the only IR it leans on), it runs on both backends
unchanged, and it is differential-testable against bump-and-revalue on the
interpreter.

Depends on: T13, T14, T17. Phase 3. Gather adjoints arrive in T19; until
then a `Gather` on a differentiated path raises `Not_differentiable`.

## Files you own

- `lib/ad/grad.ml`, `lib/ad/grad.mli` (new)
- `lib/ocaml_cuda.ml` (add `module Grad = Ocaml_cuda_ad.Grad`)
- `test/staged/unit/test_grad.ml` → promote

## Interfaces (`grad.mli`)

```ocaml
open Ocaml_cuda_ir

exception Not_differentiable of string
(** A node on a path from a [wrt] param to [output] has no adjoint rule.
    The message names the node kind and, for Reduce/Scan, the operator. *)

(** ["d" ^ output ^ "/d" ^ wrt]. *)
val grad_name : output:string -> wrt:string -> string

(** [grad g ~output ~wrt] : a graph named [Graph.name g ^ "_grad"] whose
    outputs are those of [g], in order, followed by
    [grad_name ~output ~wrt:p] for each [p] in [wrt] (in order), each with
    the dtype and shape of param [p], holding ∂(Σ output)/∂p.

    [output] must name an output of [g] with a float dtype; if it is not a
    scalar its elements are summed (seed adjoint = ones).
    A [wrt] param that [output] does not depend on gets an all-zero gradient.
    [Invalid_argument] for an unknown output or param name, or a non-float
    output. *)
val grad : Graph.t -> output:string -> wrt:string list -> Graph.t
```

## Algorithm

Work over `Graph.topological_order g` reversed. Keep `adj : (int, Tensor.packed) Hashtbl.t`
keyed by `Uid.to_int`, mapping a node to its accumulated adjoint tensor
(same dtype and shape as the node).

```
accumulate n t  =  adj[n] <- (match adj[n] with None -> t | Some a -> map2 add a t)
seed: accumulate out_node (full dtype shape 1)
for n in reversed topo order, when adj[n] = Some ā:
  match node n with
  | Param _ | Iota                        -> ()              (leaf)
  | Map (f, a)     when float a           -> accumulate a (map2 (fun adj x -> adj * apply1 (Deriv.fn1 f) x) ā a)
  | Map2 (f, a, b)                        -> for each float operand k ∈ {a, b}:
                                              let dk = map2 (fun x y -> apply2 (Deriv.fn2 f ~wrt:k) x y) a b in
                                              accumulate k (map2 mul ā dk)
  | Reduce (f, init, src)  (last axis)    -> see below
  | Scan (f, init, src)                   -> see below
  | Broadcast (_, s)                      -> accumulate s (reshape (shape s) (reduce add ā))
  | Reshape (_, a)                        -> accumulate a (reshape (shape a) ā)
  | Gather (idx, src)                     -> raise Not_differentiable "Gather"   (T19 replaces this)
```

"when float a": an operand whose dtype is not float receives no adjoint
(`Dtype.is_float`). An integer-dtype *node* with an adjoint cannot happen:
adjoints are only created for float nodes because `Deriv.d` raises on
integer bodies; catch that `Invalid_argument` and re-raise as
`Not_differentiable` with the node kind.

**Reduce adjoint.** Let `n` = row length, `R` = rows (see T14). Replicate
a row-shaped tensor `t` (numel R) to the source shape with
`rows_of t = gather (map (fun k -> k / n) (iota src_shape)) t` (for a
rank-1 source `ā` is scalar and the index is `k / n = 0`).

- body is `Add (Arg 0, Arg 1)` or `Add (Arg 1, Arg 0)`: `accumulate src (rows_of ā)`
- `Max`/`Min`: `m = rows_of n_itself` (the reduce's own value, replicated),
  `mask = map2 (fun x m -> select (eq x m) 1 0) src m`, `accumulate src (map2 mul (rows_of ā) mask)`.
  Ties: every maximal element gets the full adjoint; document it.
- `Mul (Arg 0, Arg 1)`: `accumulate src (map2 (fun a q -> a * q) (rows_of ā) (map2 div (rows_of n_itself) src))`; document `x ≠ 0`.
- anything else: `Not_differentiable "Reduce <op>"`.

Recognising the operator: match `f.body2.node` against `Binop (Add, {node = Arg 0}, {node = Arg 1})`
and the swapped form. Do not attempt algebraic normalisation.

**Scan adjoint** (`Add` only, else `Not_differentiable "Scan <op>"`):
`adj_src[i] = Σ_{j ≥ i in row} ā[j] = total_row − inclusive_prefix(ā)[i] + ā[i]`.
Build `total = rows_of (reduce_last ā)`, `prefix = scan_last ā`, then
`map2 add (map2 sub total prefix) ā`. `reduce_last`/`scan_last` dispatch on
rank: `reduce`/`scan` for rank 1, `reduce_rows`/`scan_rows` otherwise.

**Assembling the result.** For each `p` in `wrt`: find the Param node by
name in `Graph.params g`; its gradient is `adj[p]` or `full dtype shape 0`.
`Graph.create ~name ~outputs:(Graph.outputs g @ grads)`.

## Invariants

- The original outputs are unchanged nodes (same uids), so the forward
  values in the grad graph are bit-identical to those of `g`.
- `grad` never mutates `g` and never evaluates anything.
- No node kind is created that `Fusion`/`Lower` do not already handle.

## Failure modes to avoid

- Seeding with `map (fun _ -> 1) out_node`: it adds a use of the output
  node and changes its fan-out. Use `Dsl.full`.
- Accumulating with `add` into the *forward* node instead of a fresh table
  entry.
- Using `rows_of` with `Broadcast` for rank-1 only and forgetting rank ≥ 2;
  the gather form covers both.
- Walking forward instead of reverse topological order: adjoints of a node
  must be complete before they are propagated to its inputs.
- Differentiating the `Select` condition or a `Gather` index. Integers
  carry no adjoint.

## Verify

```
dune build 2>&1 && make unit && dune exec test/unit/test_grad.exe
```

## Tests (already written: `test/staged/unit/test_grad.ml`)

All on the interpreter in F64. `fd` in the test perturbs each input
element by ±1e-6 and reruns the *original* graph; tolerance 1e-5.

- `sum (x²)` → `2x`
- `sum (x*y)` → `y` and `x` (two `wrt`)
- chain: `sum (exp x * y + sin (x*y))`
- `reduce max` → indicator of the argmax
- `reduce_rows add` over `[2;3]` then weighted sum → weights replicated per row
- `scan add` then `sum (w * cumsum x)` → reverse cumulative sums of `w`
- `scan_rows add` over `[2;3]` likewise
- broadcast: `sum (x * broadcast s)` → `d/ds = sum x`, shape `[]`
- reshape passes through: `sum (map f (reshape [6] x))` with `x : [2;3]` has gradient of shape `[2;3]`
- non-scalar output `r = x*y` (shape `[4]`): gradient equals `y` (seeded with ones)
- `wrt` a param the output does not use → zeros of the param's shape
- reduce with an unrecognised operator raises `Not_differentiable` whose message contains `Reduce`
- `Gather` on the path raises `Not_differentiable` (this assertion is *replaced* by T19, which owns the change; see that spec)
- the grad graph's original outputs equal `g`'s outputs value-for-value; output count = originals + |wrt|; names follow `grad_name`
- unknown output name and unknown param name raise `Invalid_argument`
