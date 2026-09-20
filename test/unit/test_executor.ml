(* Differential (CPU-only part) and Executor (GPU part, skipped
   without a device). *)
open Cudacaml
module C = Cudacaml_testlib.Check
module Programs = Cudacaml_examples.Programs

(* Wrap the interpreter and corrupt element [i] of the first output. *)
let corrupt i (outs : (string * Value.packed) list) =
  (match outs with
  | (_, Value.P v) :: _ -> (
      match Value.dtype v with
      | Dtype.F32 -> Value.set v i (Value.get v i *. 1.001)
      | Dtype.F64 -> Value.set v i (Value.get v i *. 1.001)
      | Dtype.I32 -> Value.set v i Int32.max_int
      | Dtype.I64 -> Value.set v i Int64.max_int
      | Dtype.Bool -> ())
  | [] -> ());
  outs

module Slight : Backend.S = struct
  let name = "slight"

  type compiled = Backend_interp.compiled

  let compile = Backend_interp.compile
  let run c ~inputs = corrupt 1 (Backend_interp.run c ~inputs)
end

module Reordered : Backend.S = struct
  let name = "reordered"

  type compiled = Backend_interp.compiled

  let compile = Backend_interp.compile
  let run c ~inputs = List.rev (Backend_interp.run c ~inputs)
end

module Missing : Backend.S = struct
  let name = "missing"

  type compiled = Backend_interp.compiled

  let compile = Backend_interp.compile
  let run c ~inputs = List.tl (Backend_interp.run c ~inputs)
end

module Boom : Backend.S = struct
  let name = "boom"

  type compiled = unit

  let compile _ = ()
  let run () ~inputs:_ = failwith "boom"
end

let check ?tolerance candidate (e : Programs.t) =
  Differential.check ?tolerance ~reference:(module Backend_interp) ~candidate (e.graph ()) ~inputs:(e.inputs ())

let expect_ok r = match r with Ok () -> () | Error m -> C.fail "%s" m
let expect_error ?sub r = match r with Ok () -> C.fail "expected a mismatch" | Error m -> Option.iter (fun sub -> C.contains ~sub m) sub

let () =
  C.test "interp agrees with itself on every example" (fun () ->
      List.iter (fun (e : Programs.t) -> expect_ok (check (module Backend_interp) e)) Programs.all);
  C.test "a 0.1% error fails at the default tolerance and names the output" (fun () ->
      expect_error ~sub:"r" (check (module Slight) (Programs.saxpy 16)));
  C.test "the same error passes at tolerance 1e-2" (fun () ->
      expect_ok (check ~tolerance:1e-2 (module Slight) (Programs.saxpy 16)));
  C.test "output order does not matter, only names" (fun () ->
      expect_ok (check (module Reordered) (Programs.saxpy 16)));
  C.test "a missing output is an error" (fun () -> expect_error (check (module Missing) (Programs.saxpy 16)));
  C.test "a candidate that raises yields Error, not an exception" (fun () ->
      expect_error ~sub:"boom" (check (module Boom) (Programs.saxpy 16)));
  if not (Runtime.Device.available ()) then C.skip "no CUDA device: Executor tests"
  else begin
    C.test "executor: saxpy on the GPU matches the interpreter" (fun () ->
        expect_ok (check (module Backend_cuda) (Programs.saxpy 1000)));
    C.test "executor: missing input raises" (fun () ->
        let e = Programs.saxpy 16 in
        let c = Backend_cuda.compile (e.graph ()) in
        C.raises (fun () -> ignore (Backend_cuda.run c ~inputs:[])));
    C.test "executor: compiled program can be run twice with different inputs" (fun () ->
        let e = Programs.sum 100 in
        let c = Backend_cuda.compile (e.graph ()) in
        let mk k = [ ("x", Value.P (Value.of_list Dtype.F32 (Shape.of_dims [ 100 ]) (List.init 100 (fun _ -> k)))) ] in
        let first_f32 : type a. a Dtype.t -> a Value.t -> float =
         fun d v -> match d with Dtype.F32 -> Value.get v 0 | _ -> nan
        in
        let get o =
          match List.assoc "s" o with Value.P v -> first_f32 (Value.dtype v) v
        in
        C.float ~tol:1e-3 ~expect:100.0 (get (Backend_cuda.run c ~inputs:(mk 1.0)));
        C.float ~tol:1e-3 ~expect:200.0 (get (Backend_cuda.run c ~inputs:(mk 2.0))))
  end;
  C.run ()
