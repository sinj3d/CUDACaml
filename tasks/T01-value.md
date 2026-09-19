# T01 — `Value`: host-side tensors on Bigarray

## Goal

Implement `lib/ir/value.ml` so that every function in `lib/ir/value.mli`
works. A `Value.t` is the only host tensor type in the project. It must be
backed by a `Bigarray.Array1` because Bigarray storage lives outside the
OCaml heap and is never moved by the GC, which is what later lets the
runtime hand its address to `cudaMemcpy`.

## Files you own

- `lib/ir/value.ml` — replace the stub entirely.

Do not edit `value.mli`, `dtype.ml`, `shape.mli`, or anything else.

## Interfaces

The contract you implement (`lib/ir/value.mli`, verbatim):

```ocaml
type 'a t
type packed = P : _ t -> packed

val create : 'a Dtype.t -> Shape.t -> 'a t
val of_list : 'a Dtype.t -> Shape.t -> 'a list -> 'a t
val dtype : 'a t -> 'a Dtype.t
val shape : 'a t -> Shape.t
val numel : _ t -> int
val byte_size : _ t -> int
val get : 'a t -> int -> 'a
val set : 'a t -> int -> 'a -> unit
val to_list : 'a t -> 'a list

type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw
val raw : 'a t -> 'a raw
```

Types you depend on (`lib/ir/dtype.ml`, already implemented):

```ocaml
type _ t =
  | F32 : float t
  | F64 : float t
  | I32 : int32 t
  | I64 : int64 t
  | Bool : bool t
val size_in_bytes : _ t -> int    (* 4, 8, 4, 8, 1 *)
```

`Shape.numel : Shape.t -> int` (already implemented; `Shape.scalar` has numel 1).

## Implementation

Keep this record (the stub already has it; keep the field names):

```ocaml
type 'a raw = Raw : ('a, _, Bigarray.c_layout) Bigarray.Array1.t -> 'a raw
type 'a t = { dtype : 'a Dtype.t; shape : Shape.t; data : 'a raw }
type packed = P : _ t -> packed
```

`create dtype shape`:
- `n = Shape.numel shape`.
- Match on `dtype` and build the Bigarray with the matching kind. This
  needs a locally abstract type so each branch can return a different
  Bigarray element kind under the same `'a`:
  ```ocaml
  let create : type a. a Dtype.t -> Shape.t -> a t =
   fun dtype shape ->
    let n = Shape.numel shape in
    let data : a raw =
      match dtype with
      | Dtype.F32 ->
          let a = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout n in
          Bigarray.Array1.fill a 0.0;
          Raw a
      | Dtype.F64 -> (* Float64, fill 0.0 *)
      | Dtype.I32 -> (* Int32, fill 0l *)
      | Dtype.I64 -> (* Int64, fill 0L *)
      | Dtype.Bool -> invalid_arg "Value.create: Bool tensors are not storable"
    in
    { dtype; shape; data }
  ```
- **Must zero-fill.** `Bigarray.Array1.create` returns uninitialised memory.

`of_list dtype shape l`: if `List.length l <> Shape.numel shape` raise
`Invalid_argument`. Otherwise `create` then `set` each element with
`List.iteri`.

`get t i` / `set t i v`: `match t.data with Raw a -> Bigarray.Array1.get a i`
(resp. `set`). Bigarray raises `Invalid_argument` on out-of-range indices;
do not add your own check, but do not swallow it either.

`to_list t`: `List.init (numel t) (get t)`.

`numel t = Shape.numel t.shape`; `byte_size t = numel t * Dtype.size_in_bytes t.dtype`.

`raw t = t.data`; `dtype`/`shape` are field accessors.

## Invariants

- Storage is contiguous, C layout, exactly `numel` elements, row-major for
  multi-dimensional shapes (index `i` in `get` is the flat index).
- `raw` returns the *same* storage, not a copy: writes through the
  Bigarray are visible via `get`.
- `F32` values are stored as single precision. `of_list F32 [0.1]` then
  `get` returns `0.100000001490116...`, not `0.1`. Bigarray does this for
  you; do not round manually.

## Failure modes to avoid

- Forgetting `Array1.fill` (uninitialised memory; tests will be flaky).
- Destructuring `Raw a` with `let Raw a = ...` — OCaml rejects existential
  types in `let` patterns. Use `match t.data with Raw a -> ...`.
- Writing `create` without the `type a.` annotation: the match will fail to
  type-check because each branch has a different Bigarray kind.
- Using `Bigarray.Genarray` or `Array` instead of `Array1`: the runtime
  (T07) pattern-matches on `Raw` expecting an `Array1`.
- Returning a `float` Bigarray for `Bool`. `Bool` must raise.

## Verify

```
dune build 2>&1 && dune exec test/unit/test_value.exe
```

Expected last line: `10 tests, 0 failures`.

## Tests (already written: `test/unit/test_value.ml`)

- create zero-fills; numel/byte_size correct
- of_list / get / set / to_list roundtrip on I32
- of_list rejects wrong length
- get out of bounds raises (both ends)
- Bool is not storable
- scalar shape has one element and 4 bytes for F32
- byte_size follows dtype (F64, I64 → 8 per element)
- raw shares storage with get
- 2-D shape is row-major
- F32 rounds 0.1 to single precision
