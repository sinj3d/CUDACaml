(* Events. Always timing-enabled: [elapsed_ms] is the only reason this
   layer exists beyond synchronisation, and the cost of the flag is a
   fraction of a microsecond at record time. *)

type t = Cuda.Event.t

let record s =
  Device.init ();
  let e = Cuda.Event.create ~enable_timing:true () in
  Cuda.Event.record e (Stream.unsafe_stream s);
  e

let wait s e = Cuda.Event.wait (Stream.unsafe_stream s) e
let synchronize t = Cuda.Event.synchronize t
let elapsed_ms ~start ~stop = Cuda.Event.elapsed_time ~start ~end_:stop
