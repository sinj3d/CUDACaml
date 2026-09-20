(** Umbrella module. Users [open Cudacaml] and see one flat namespace;
    the layering below is an implementation concern. Layers only depend
    downward (see ARCHITECTURE.md). *)

(* Layer 1: the IR *)
module Uid = Cudacaml_ir.Uid
module Dtype = Cudacaml_ir.Dtype
module Shape = Cudacaml_ir.Shape
module Expr = Cudacaml_ir.Expr
module Tensor = Cudacaml_ir.Tensor
module Value = Cudacaml_ir.Value
module Graph = Cudacaml_ir.Graph
module Dsl = Cudacaml_ir.Dsl

(* Layer 1.5: libraries over the DSL *)
module Rng = Cudacaml_rng.Rng

(* Layer 2: symbolic differentiation of element functions *)
module Deriv = Cudacaml_ad.Deriv
module Grad = Cudacaml_ad.Grad

(* Layer 2: graph -> graph *)
module Pass = Cudacaml_passes.Pass

module Layout = Cudacaml_passes.Layout
module Pipeline = Cudacaml_passes.Pipeline

(* Layer 3: graph -> imperative kernels *)
module Fusion = Cudacaml_lower.Fusion
module Schedule = Cudacaml_lower.Schedule
module Kernel_ir = Cudacaml_lower.Kernel_ir
module Lower = Cudacaml_lower.Lower

(* Layer 4: device (IR-agnostic) *)
module Runtime = Cudacaml_runtime

(* Layer 5: executors *)
module Backend = Cudacaml_backend.Backend
module Differential = Cudacaml_backend.Differential
module Backend_interp = Cudacaml_backend_interp
module Backend_cuda = Cudacaml_backend_cuda
