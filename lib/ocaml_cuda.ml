(** Umbrella module. Users [open Ocaml_cuda] and see one flat namespace;
    the layering below is an implementation concern. Layers only depend
    downward (see ARCHITECTURE.md). *)

(* Layer 1: the IR *)
module Uid = Ocaml_cuda_ir.Uid
module Dtype = Ocaml_cuda_ir.Dtype
module Shape = Ocaml_cuda_ir.Shape
module Expr = Ocaml_cuda_ir.Expr
module Tensor = Ocaml_cuda_ir.Tensor
module Value = Ocaml_cuda_ir.Value
module Graph = Ocaml_cuda_ir.Graph
module Dsl = Ocaml_cuda_ir.Dsl

(* Layer 1.5: libraries over the DSL *)
module Rng = Ocaml_cuda_rng.Rng

(* Layer 2: graph -> graph *)
module Pass = Ocaml_cuda_passes.Pass

module Layout = Ocaml_cuda_passes.Layout
module Pipeline = Ocaml_cuda_passes.Pipeline

(* Layer 3: graph -> imperative kernels *)
module Fusion = Ocaml_cuda_lower.Fusion
module Schedule = Ocaml_cuda_lower.Schedule
module Kernel_ir = Ocaml_cuda_lower.Kernel_ir
module Lower = Ocaml_cuda_lower.Lower

(* Layer 4: device (IR-agnostic) *)
module Runtime = Ocaml_cuda_runtime

(* Layer 5: executors *)
module Backend = Ocaml_cuda_backend.Backend
module Differential = Ocaml_cuda_backend.Differential
module Backend_interp = Ocaml_cuda_backend_interp
module Backend_cuda = Ocaml_cuda_backend_cuda
