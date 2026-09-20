(* Black–Scholes Monte Carlo and pathwise Greeks on the interpreter,
   F64, fixed seed. Every expected value is the closed form. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check
module Bs = Ocaml_cuda_examples.Black_scholes
module Programs = Ocaml_cuda_examples.Programs

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let dtype = Dtype.F64
let seed = 7l
let inputs () = Bs.inputs ~dtype Bs.market ~seed
let cf = Bs.analytic Bs.market
let g_name w = Grad.grad_name ~output:"price" ~wrt:w

let () =
  C.test "closed form" (fun () ->
      C.float ~tol:2e-3 ~expect:10.4506 cf.price;
      C.float ~tol:2e-3 ~expect:0.6368 cf.delta;
      C.float ~tol:2e-3 ~expect:37.524 cf.vega;
      C.float ~tol:2e-3 ~expect:53.232 cf.rho);
  C.test "call_mc 65536 paths within 0.25 of the closed form" (fun () ->
      let o = run (Bs.call_mc ~n_paths:65536 ~dtype Bs.market) (inputs ()) in
      C.float ~tol:0.25 ~expect:cf.price (Bs.scalar_out o "price"));
  C.test "call_mc_paths 16384 x 8 within 0.5 of the closed form" (fun () ->
      let o = run (Bs.call_mc_paths ~n_paths:16384 ~n_steps:8 ~dtype Bs.market) (inputs ()) in
      C.float ~tol:0.5 ~expect:cf.price (Bs.scalar_out o "price"));
  C.test "call_greeks: forward price identical, Greeks near closed form" (fun () ->
      let price = Bs.scalar_out (run (Bs.call_mc ~n_paths:65536 ~dtype Bs.market) (inputs ())) "price" in
      let o = run (Bs.call_greeks ~n_paths:65536 ~dtype Bs.market) (inputs ()) in
      C.float ~tol:0.0 ~expect:price (Bs.scalar_out o "price");
      C.float ~tol:0.02 ~expect:cf.delta (Bs.scalar_out o (g_name "s0"));
      C.float ~tol:1.5 ~expect:cf.vega (Bs.scalar_out o (g_name "vol"));
      C.float ~tol:1.5 ~expect:cf.rho (Bs.scalar_out o (g_name "rate")));
  C.test "greeks graph: 4 params, 4 outputs, bounded kernel count" (fun () ->
      let g = Bs.call_greeks ~n_paths:4096 ~dtype Bs.market in
      C.bool ~expect:true
        (List.sort compare (List.map fst (Graph.params g)) = [ "rate"; "s0"; "seed"; "vol" ]);
      C.int ~expect:4 (List.length (Graph.outputs g));
      C.bool ~expect:true (List.length (Lower.program g).kernels < 40));
  C.test "registered examples" (fun () ->
      List.iter (fun n -> C.bool ~expect:true (Option.is_some (Programs.find n))) [ "bs_mc"; "bs_paths"; "bs_greeks" ]);
  C.run ()
