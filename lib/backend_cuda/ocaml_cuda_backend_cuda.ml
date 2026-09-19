open Ocaml_cuda_lower
open Ocaml_cuda_runtime
module Emit = Emit
module Mangle = Mangle
module Executor = Executor

let name = "cuda"

type compiled = { program : Kernel_ir.program; module_ : Jit.module_ }

let source graph = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program |> Emit.program

let compile graph =
  let program = graph |> Ocaml_cuda_passes.Pipeline.run |> Lower.program in
  let module_ = Jit.compile ~name:program.name ~source:(Emit.program program) in
  { program; module_ }

let run { program; module_ } ~inputs = Executor.run program module_ ~inputs
