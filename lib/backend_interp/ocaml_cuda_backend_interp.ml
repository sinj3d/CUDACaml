open Ocaml_cuda_ir

let name = "interp"

type compiled = Graph.t

let compile g = g
let run _ ~inputs:_ = failwith "Backend_interp.run: TODO"
