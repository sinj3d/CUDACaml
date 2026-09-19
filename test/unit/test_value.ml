(* T01: Value. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check

let () =
  C.test "create zero-fills and sizes" (fun () ->
      let v = Value.create Dtype.F32 (Shape.of_dims [ 4 ]) in
      C.int ~expect:4 (Value.numel v);
      C.int ~expect:16 (Value.byte_size v);
      C.floats ~expect:[ 0.; 0.; 0.; 0. ] (Value.to_list v));
  C.test "of_list / get / set / to_list roundtrip" (fun () ->
      let v = Value.of_list Dtype.I32 (Shape.of_dims [ 3 ]) [ 1l; 2l; 3l ] in
      C.bool ~expect:true (Value.get v 2 = 3l);
      Value.set v 0 9l;
      C.bool ~expect:true (Value.to_list v = [ 9l; 2l; 3l ]));
  C.test "of_list rejects wrong length" (fun () ->
      C.raises (fun () -> ignore (Value.of_list Dtype.F32 (Shape.of_dims [ 2 ]) [ 1.0 ])));
  C.test "get out of bounds raises" (fun () ->
      let v = Value.create Dtype.F64 (Shape.of_dims [ 2 ]) in
      C.raises (fun () -> ignore (Value.get v 2));
      C.raises (fun () -> ignore (Value.get v (-1))));
  C.test "Bool is not storable" (fun () ->
      C.raises (fun () -> ignore (Value.create Dtype.Bool Shape.scalar)));
  C.test "scalar shape has exactly one element" (fun () ->
      let v = Value.create Dtype.F32 Shape.scalar in
      C.int ~expect:1 (Value.numel v);
      C.int ~expect:4 (Value.byte_size v));
  C.test "byte_size follows dtype" (fun () ->
      C.int ~expect:24 (Value.byte_size (Value.create Dtype.F64 (Shape.of_dims [ 3 ])));
      C.int ~expect:24 (Value.byte_size (Value.create Dtype.I64 (Shape.of_dims [ 3 ]))));
  C.test "raw is a c_layout Bigarray of matching length, sharing storage" (fun () ->
      let v = Value.of_list Dtype.I64 (Shape.of_dims [ 2; 3 ]) [ 1L; 2L; 3L; 4L; 5L; 6L ] in
      match Value.raw v with
      | Value.Raw a ->
          C.int ~expect:6 (Bigarray.Array1.dim a);
          C.bool ~expect:true (Bigarray.Array1.get a 5 = 6L);
          Bigarray.Array1.set a 0 42L;
          C.bool ~expect:true (Value.get v 0 = 42L));
  C.test "2-D shape is row-major flat" (fun () ->
      let v = Value.of_list Dtype.I32 (Shape.of_dims [ 2; 2 ]) [ 1l; 2l; 3l; 4l ] in
      C.bool ~expect:true (Value.get v 2 = 3l);
      C.bool ~expect:true (Shape.equal (Value.shape v) (Shape.of_dims [ 2; 2 ])));
  C.test "f32 storage rounds to single precision" (fun () ->
      let v = Value.of_list Dtype.F32 (Shape.of_dims [ 1 ]) [ 0.1 ] in
      C.bool ~expect:true (Value.get v 0 <> 0.1);
      C.float ~tol:1e-9 ~expect:0.100000001490116 (Value.get v 0));
  C.run ()
