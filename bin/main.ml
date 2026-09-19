(* CLI over the example registry.
     list             names of all examples
     emit  <example>  print generated CUDA C++
     dot   <example>  print the graph in Graphviz form
     run   <example>  execute on the CUDA backend and print outputs
     check <example>  Differential.check interp vs cuda
     info             print the device description *)

open Ocaml_cuda
module Programs = Ocaml_cuda_examples.Programs

let usage () =
  prerr_endline "usage: ocaml-cuda (list | info | emit|dot|run|check <example>)";
  exit 2

let example name =
  match Programs.find name with
  | Some e -> e
  | None ->
      prerr_endline ("unknown example: " ^ name);
      exit 2

let print_value name (Value.P v) =
  let items : string list =
    match Value.dtype v with
    | Dtype.F32 -> List.map string_of_float (Value.to_list v)
    | Dtype.F64 -> List.map string_of_float (Value.to_list v)
    | Dtype.I32 -> List.map Int32.to_string (Value.to_list v)
    | Dtype.I64 -> List.map Int64.to_string (Value.to_list v)
    | Dtype.Bool -> List.map string_of_bool (Value.to_list v)
  in
  let shown = List.filteri (fun i _ -> i < 8) items in
  Printf.printf "%s %s = [%s%s]\n" name (Shape.to_string (Value.shape v)) (String.concat ", " shown)
    (if List.length items > 8 then ", ..." else "")

let () =
  match Array.to_list Sys.argv with
  | [ _; "list" ] -> List.iter (fun (e : Programs.t) -> print_endline e.name) Programs.all
  | [ _; "emit"; name ] -> print_string (Backend_cuda.source ((example name).graph ()))
  | [ _; "dot"; name ] -> print_string (Graph.to_dot ((example name).graph ()))
  | [ _; "run"; name ] ->
      let e = example name in
      let c = Backend_cuda.compile (e.graph ()) in
      List.iter (fun (n, v) -> print_value n v) (Backend_cuda.run c ~inputs:(e.inputs ()))
  | [ _; "info" ] ->
      if not (Runtime.Device.available ()) then begin
        prerr_endline "no CUDA device";
        exit 3
      end;
      print_string (Runtime.Device.info_to_string (Runtime.Device.info ()))
  | [ _; "check"; name ] -> (
      let e = example name in
      match
        Differential.check ~reference:(module Backend_interp) ~candidate:(module Backend_cuda)
          (e.graph ()) ~inputs:(e.inputs ())
      with
      | Ok () -> print_endline (name ^ ": ok")
      | Error msg ->
          prerr_endline (name ^ ": MISMATCH " ^ msg);
          exit 1)
  | _ -> usage ()
