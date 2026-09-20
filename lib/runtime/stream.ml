(* Streams. A thin wrapper: the point is that the rest of the project
   never names [Cuda.Stream] directly. *)

type t = Cuda.Stream.t

(* A pure value in cudajit (a null CUstream), so binding it here costs
   nothing and is safe on a machine with no driver. *)
let default = Cuda.Stream.no_stream

(* [non_blocking] means "do not implicitly synchronise with the NULL
   stream". Without it, a [compile_with ~streams:4] whose executor 0 sits
   on the NULL stream would serialise the other three. *)
let create () =
  Device.init ();
  Cuda.Stream.create ~non_blocking:true ()

let synchronize t = Cuda.Stream.synchronize t

(* cudajit keeps the marshalled arguments of every launch alive on the
   stream that ran it, and drops them only when the stream is known to be
   drained -- which it learns from a synchronize or from this query. A
   caller that waits on events alone would otherwise accumulate one such
   record per launch for the life of the process. *)
let is_idle t = Cuda.Stream.is_ready t
let unsafe_stream t = t
