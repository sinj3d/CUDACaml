(* T16: Rng. Known-answer vectors pin Philox bit for bit; statistics pin the
   uniform/normal mappings. Everything on the interpreter, deterministic. *)
open Ocaml_cuda
open Dsl
module C = Ocaml_cuda_testlib.Check
module K = Kernel_ir

let run g inputs = Backend_interp.run (Backend_interp.compile g) ~inputs
let vec n = Shape.of_dims [ n ]
let h = Int32.of_string
let seed v = ("seed", Value.P (Value.of_list Dtype.I32 Shape.scalar [ v ]))

let ints o name : int32 list =
  match List.assoc name o with
  | Value.P v -> ( match Value.dtype v with Dtype.I32 -> Value.to_list v | _ -> failwith "not i32")

let floats o name : float list =
  match List.assoc name o with
  | Value.P v -> (
      match Value.dtype v with Dtype.F32 -> Value.to_list v | Dtype.F64 -> Value.to_list v | _ -> failwith "not float")

let hex l = String.concat " " (List.map (fun v -> Printf.sprintf "%08lx" v) l)

(* Evaluate Philox on one fixed (ctr, key) by mapping constants over a
   1-element I32 tensor; the four words come out as four outputs. *)
let kat ~ctr:(c0, c1, c2, c3) ~key:(k0, k1) =
  let k v = const Dtype.I32 v in
  let x = param "x" Dtype.I32 (vec 1) in
  let word sel =
    map
      (fun _ ->
        let w0, w1, w2, w3 = Rng.Philox.round10 ~ctr:(k c0, k c1, k c2, k c3) ~key:(k k0, k k1) in
        sel (w0, w1, w2, w3))
      x
  in
  let g =
    Graph.create ~name:"kat"
      ~outputs:
        [
          ("w0", Tensor.P (word (fun (a, _, _, _) -> a)));
          ("w1", Tensor.P (word (fun (_, b, _, _) -> b)));
          ("w2", Tensor.P (word (fun (_, _, c, _) -> c)));
          ("w3", Tensor.P (word (fun (_, _, _, d) -> d)));
        ]
  in
  let o = run g [ ("x", Value.P (Value.of_list Dtype.I32 (vec 1) [ 0l ])) ] in
  List.map (fun n -> List.hd (ints o n)) [ "w0"; "w1"; "w2"; "w3" ]

let expect_words ~expect got =
  if not (List.for_all2 Int32.equal expect got) then C.fail "expected %s, got %s" (hex expect) (hex got)

let u32_list ~key n =
  let g = Graph.create ~name:"u" ~outputs:[ ("r", Tensor.P (Rng.u32 ~key:(scalar "seed" Dtype.I32) (vec n))) ] in
  ints (run g [ seed key ]) "r"

let uniform_list dtype ~key n =
  let g = Graph.create ~name:"u" ~outputs:[ ("r", Tensor.P (Rng.uniform dtype ~key:(scalar "seed" Dtype.I32) (vec n))) ] in
  floats (run g [ seed key ]) "r"

let normal_list dtype ~key n =
  let g = Graph.create ~name:"n" ~outputs:[ ("r", Tensor.P (Rng.normal dtype ~key:(scalar "seed" Dtype.I32) (vec n))) ] in
  floats (run g [ seed key ]) "r"

let mean l = List.fold_left ( +. ) 0.0 l /. float_of_int (List.length l)

let var l =
  let m = mean l in
  mean (List.map (fun x -> (x -. m) *. (x -. m)) l)

let () =
  (* Random123 kat_vectors, philox4x32 with 10 rounds. See T16 for the
     policy on suspected transcription errors. *)
  C.test "KAT: zero counter, zero key" (fun () ->
      expect_words
        ~expect:[ h "0x6627e8d5"; h "0xe169c58d"; h "0xbc57ac4c"; h "0x9b00dbd8" ]
        (kat ~ctr:(0l, 0l, 0l, 0l) ~key:(0l, 0l)));
  C.test "KAT: all-ones counter and key" (fun () ->
      expect_words
        ~expect:[ h "0x408f276d"; h "0x41c83b0e"; h "0xa20bc7c6"; h "0x6d5451fd" ]
        (kat ~ctr:(-1l, -1l, -1l, -1l) ~key:(-1l, -1l)));
  C.test "KAT: pi digits" (fun () ->
      expect_words
        ~expect:[ h "0xd16cfe09"; h "0x94fdcceb"; h "0x5001e420"; h "0x24126ea1" ]
        (kat
           ~ctr:(h "0x243f6a88", h "0x85a308d3", h "0x13198a2e", h "0x03707344")
           ~key:(h "0xa4093822", h "0x299f31d0")));
  C.test "u32 element 0 with key 0 is word 0 of the zero KAT; first four differ" (fun () ->
      let l = u32_list ~key:0l 4 in
      if not (Int32.equal (List.hd l) (h "0x6627e8d5")) then C.fail "got %s" (hex l);
      C.int ~expect:4 (List.length (List.sort_uniq Int32.compare l)));
  C.test "reproducible for a key; different keys differ" (fun () ->
      let a = u32_list ~key:42l 1024 and b = u32_list ~key:42l 1024 and c = u32_list ~key:43l 1024 in
      C.bool ~expect:true (List.for_all2 Int32.equal a b);
      let differ = List.length (List.filter (fun (x, y) -> not (Int32.equal x y)) (List.combine a c)) in
      C.bool ~expect:true (differ >= 922));
  C.test "uniform F64: open interval, mean, variance, chi-square" (fun () ->
      let n = 65536 in
      let u = uniform_list Dtype.F64 ~key:1l n in
      C.bool ~expect:true (List.for_all (fun x -> x > 0.0 && x < 1.0) u);
      C.float ~tol:0.01 ~expect:0.5 (mean u);
      C.float ~tol:0.005 ~expect:(1.0 /. 12.0) (var u);
      let buckets = Array.make 16 0 in
      List.iter (fun x -> let b = int_of_float (x *. 16.0) in buckets.(b) <- buckets.(b) + 1) u;
      let e = float_of_int n /. 16.0 in
      let chi = Array.fold_left (fun acc o -> acc +. (((float_of_int o -. e) ** 2.0) /. e)) 0.0 buckets in
      if chi >= 45.0 then C.fail "chi-square %.1f" chi);
  C.test "normal F64: moments and tail mass" (fun () ->
      let n = 65536 in
      let z = normal_list Dtype.F64 ~key:2l n in
      C.bool ~expect:true (List.for_all Float.is_finite z);
      C.float ~tol:0.02 ~expect:0.0 (mean z);
      C.float ~tol:0.05 ~expect:1.0 (var z);
      let tail = List.length (List.filter (fun x -> Float.abs x > 1.96) z) in
      C.float ~tol:0.006 ~expect:0.05 (float_of_int tail /. float_of_int n));
  C.test "normal F32: finite, centred" (fun () ->
      let z = normal_list Dtype.F32 ~key:3l 65536 in
      C.bool ~expect:true (List.for_all Float.is_finite z);
      C.float ~tol:0.02 ~expect:0.0 (mean z));
  C.test "uniform graph has one param, the seed, and one upload" (fun () ->
      let g = Graph.create ~name:"u" ~outputs:[ ("r", Tensor.P (Rng.uniform Dtype.F32 ~key:(scalar "seed" Dtype.I32) (vec 16))) ] in
      C.bool ~expect:true (List.map fst (Graph.params g) = [ "seed" ]);
      let plan = (Lower.program g).plan in
      C.int ~expect:1 (List.length (List.filter (function K.Upload _ -> true | _ -> false) plan)));
  C.test "counter is the flat index: [4;8] equals [32]" (fun () ->
      let k () = scalar "seed" Dtype.I32 in
      let g2 = Graph.create ~name:"a" ~outputs:[ ("r", Tensor.P (Rng.u32 ~key:(k ()) (Shape.of_dims [ 4; 8 ]))) ] in
      let g1 = Graph.create ~name:"b" ~outputs:[ ("r", Tensor.P (Rng.u32 ~key:(k ()) (vec 32))) ] in
      let a = ints (run g2 [ seed 9l ]) "r" and b = ints (run g1 [ seed 9l ]) "r" in
      C.bool ~expect:true (List.for_all2 Int32.equal a b));
  C.run ()
