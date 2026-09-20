(** Imperative kernel IR: what a GPU actually runs.

    The gap between [Tensor] and here is the whole compiler. The graph is
    timeless dataflow; this has loops, thread indices, memory spaces and
    synchronisation. [Emit] pretty-prints it without making any decisions,
    so every performance choice is visible in this data structure.

    Hardcaml analogue: [Rtl_ast], the language-neutral form that both the
    Verilog and VHDL printers consume. *)

open Cudacaml_ir

type memspace = Global | Shared

type buffer = { name : string; dtype : Dtype.packed; memspace : memspace; numel : int }

type literal = F of float | I of int64 | B of bool

type expr =
  | Var of string
  | Lit of literal * Dtype.packed
  | Global_thread_id  (** blockIdx.x * blockDim.x + threadIdx.x *)
  | Global_size  (** gridDim.x * blockDim.x *)
  | Local_thread_id  (** threadIdx.x *)
  | Local_thread_id_y  (** threadIdx.y; only a 2-D launch has one *)
  | Block_id  (** blockIdx.x *)
  | Block_id_y  (** blockIdx.y; only a 2-D launch has one *)
  | Block_dim  (** blockDim.x *)
  | Load of { buf : buffer; index : expr }
  | Binop of Dtype.packed * Expr.binop * expr * expr  (** dtype = result type *)
  | Unop of Dtype.packed * Expr.unop * expr
  | Cmp of Expr.cmp * expr * expr
  | Logic of Expr.logic * expr * expr
  | Not of expr
  | Select of expr * expr * expr
  | Cast of Dtype.packed * expr

type stmt =
  | Let of { var : string; dtype : Dtype.packed; value : expr }  (** declares + inits *)
  | Assign of { var : string; value : expr }  (** re-assigns an existing [Let] var *)
  | Store of { buf : buffer; index : expr; value : expr }
  | Atomic_add of { buf : buffer; index : expr; value : expr }
      (** [atomicAdd(&buf[index], value)]: a read-modify-write that no other
          thread can interleave with. The order in which concurrent adds
          land is unspecified, so a float accumulation is not bit-reproducible. *)
  | For of { var : string; lo : expr; hi : expr; step : expr; body : stmt list }
      (** [for (int var = lo; var < hi; var += step)] *)
  | If of { cond : expr; then_ : stmt list; else_ : stmt list }
  | Sync_threads

type kernel = {
  name : string;
  params : buffer list;  (** in order; all [Global] *)
  shared : buffer list;  (** [__shared__] declarations at the top of the body *)
  body : stmt list;
  launch : Schedule.launch;
}

(** The host-side plan. [Executor] runs it top to bottom; it is the only
    thing the runtime layer needs to understand about a program. *)
type host_op =
  | Alloc of buffer
  | Upload of { param : string; into : buffer }
  | Launch of { kernel : string; args : buffer list }
  | Download of { from : buffer; output : string; shape : Shape.t }
  | Free of buffer

type program = { name : string; kernels : kernel list; plan : host_op list }
