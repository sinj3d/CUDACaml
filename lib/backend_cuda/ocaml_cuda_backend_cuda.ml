open Ocaml_cuda_lower
open Ocaml_cuda_runtime
module Emit = Emit
module Mangle = Mangle
module Executor = Executor

let name = "cuda"

(* A compiled program owns its device buffers (see [Executor]); [release]
   gives them back. *)
type compiled = { exec : Executor.t }

let source graph = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program |> Emit.program

let compile graph =
  let program = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program in
  let module_ = Jit.compile ~name:program.name ~source:(Emit.program program) in
  { exec = Executor.create program module_ }

let run { exec } ~inputs = Executor.run exec ~inputs
let release { exec } = Executor.release exec
let executor { exec } = exec
