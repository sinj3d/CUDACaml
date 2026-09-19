# T09 — CLI: `ocaml-cuda list | emit | dot | run | check`

## Goal

`bin/main.ml` is already written against the example registry
(`examples/programs.ml`). This task makes sure it builds, behaves, and
that its output is usable as a debugging tool by the system-test phase.
Expect to fix small compile errors (it was written before the layers
below it existed) and nothing else.

Depends on: T08.

## Files you own

- `bin/main.ml`
- `examples/programs.ml` (only if an example fails to build; do not
  change any example's semantics — T03/T05 tests depend on them)

## Interfaces used

```ocaml
Programs.all  : Programs.t list         (* { name; graph : unit -> Graph.t; inputs : unit -> (string * Value.packed) list } *)
Programs.find : string -> Programs.t option
Backend_cuda.source  : Graph.t -> string
Backend_cuda.compile : Graph.t -> compiled ; Backend_cuda.run
Backend_interp.(compile, run)
Differential.check ~reference ~candidate graph ~inputs : (unit, string) result
Graph.to_dot : Graph.t -> string
```

## Required behaviour

| Command | Output | Exit |
|---|---|---|
| `ocaml-cuda list` | one example name per line, registry order | 0 |
| `ocaml-cuda emit saxpy` | the CUDA C++ source, nothing else, to stdout | 0 |
| `ocaml-cuda dot saxpy` | Graphviz text to stdout | 0 |
| `ocaml-cuda run saxpy` | one line per output: `name [shape] = [first 8 values, ...]` | 0 (needs GPU) |
| `ocaml-cuda check saxpy` | `saxpy: ok` or `saxpy: MISMATCH <reason>` to stderr | 0 / 1 |
| unknown example / bad args | usage to stderr | 2 |

`emit` and `dot` must work **without a GPU** — they must not call
`Device.init`. (`Backend_cuda.source` does not; verify by running on the
no-GPU box.)

## Verify

```
dune build 2>&1
dune exec ocaml-cuda -- list
dune exec ocaml-cuda -- emit chain | grep -c "__global__"      # must print 1
dune exec ocaml-cuda -- emit saxpy | grep -c "__global__"      # must print 2
dune exec ocaml-cuda -- emit saxpy | grep -q 'extern "C"' && echo ok
dune exec ocaml-cuda -- dot fanout | grep -c -- "->"           # > 0
dune exec ocaml-cuda -- nope; echo "exit=$?"                   # exit=2
```
On the GPU machine additionally:
```
for e in $(dune exec ocaml-cuda -- list); do dune exec ocaml-cuda -- check $e || exit 1; done
dune exec ocaml-cuda -- run squares | head -1     # r [64] = [0., 1., 4., 9., 16., 25., 36., 49., ...]
```

## Failure modes to avoid

- Printing anything besides the source on `emit` (people pipe it to
  `nvcc -ptx` to debug).
- Catching exceptions broadly and printing "error": let the real
  exception surface with its message.
