(* T23: Longstaff–Schwartz American put on the interpreter backend. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module Lsm = Ocaml_cuda_examples.Lsm
module Bs = Ocaml_cuda_examples.Black_scholes

let compiles = ref 0

module Counting : Backend.S = struct
  let name = "counting-interp"

  type compiled = Backend_interp.compiled

  let compile g =
    incr compiles;
    Backend_interp.compile g

  let run = Backend_interp.run
end

module L = Lsm.Make (Backend_interp)
module LC = Lsm.Make (Counting)

let () =
  let binom = Lsm.binomial_american_put ~steps:500 Bs.market in
  let euro = Lsm.european_put Bs.market in
  C.test "references: binomial American put 6.09, European put 5.5735" (fun () ->
      C.float ~tol:0.02 ~expect:6.09 binom;
      C.float ~tol:1e-3 ~expect:5.5735 euro);
  C.test "LSM price within 0.25 of the binomial value and above the European put" (fun () ->
      let t = L.create ~n_paths:8192 ~n_steps:50 Bs.market in
      let px = L.price t ~seed:11l in
      Printf.printf "    lsm %.4f  binomial %.4f  european %.4f\n%!" px binom euro;
      C.float ~tol:0.25 ~expect:binom px;
      C.bool ~expect:true (px >= euro -. 0.05));
  C.test "deterministic per seed; seeds differ modestly" (fun () ->
      let t = L.create ~n_paths:4096 ~n_steps:20 Bs.market in
      let a = L.price t ~seed:11l and b = L.price t ~seed:11l and c = L.price t ~seed:12l in
      C.float ~tol:0.0 ~expect:a b;
      C.bool ~expect:true (Float.abs (a -. c) < 0.5));
  C.test "create compiles exactly three graphs" (fun () ->
      compiles := 0;
      let t = LC.create ~n_paths:256 ~n_steps:4 Bs.market in
      C.int ~expect:3 !compiles;
      ignore (LC.price t ~seed:1l);
      C.int ~expect:3 !compiles);
  C.run ()
