(* T11: Device.info and its rendering. GPU-dependent; skips without one. *)
open Ocaml_cuda
module C = Ocaml_cuda_testlib.Check

let lines s = String.split_on_char '\n' (String.trim s)
let starts_with ~prefix s = String.length s >= String.length prefix && String.sub s 0 (String.length prefix) = prefix

let () =
  if not (Runtime.Device.available ()) then begin
    C.test "info raises without a device" (fun () -> C.raises (fun () -> ignore (Runtime.Device.info ())));
    C.skip "no CUDA device: the remaining info tests need one";
    C.run ()
  end;
  C.test "info fields are plausible" (fun () ->
      let i = Runtime.Device.info () in
      let major, minor = i.compute_capability in
      C.bool ~expect:true (String.length i.name > 0);
      C.bool ~expect:true (major >= 5);
      C.bool ~expect:true (minor >= 0 && minor < 10);
      C.bool ~expect:true (i.multiprocessors >= 1);
      C.bool ~expect:true (i.total_memory_bytes >= 1 lsl 30));
  C.test "info_to_string has the four keys in order" (fun () ->
      let i = Runtime.Device.info () in
      let ls = lines (Runtime.Device.info_to_string i) in
      C.int ~expect:4 (List.length ls);
      C.bool ~expect:true (starts_with ~prefix:"device: " (List.nth ls 0));
      C.bool ~expect:true (starts_with ~prefix:"compute: sm_" (List.nth ls 1));
      C.bool ~expect:true (starts_with ~prefix:"multiprocessors: " (List.nth ls 2));
      C.bool ~expect:true (starts_with ~prefix:"memory_mib: " (List.nth ls 3));
      let major, minor = i.compute_capability in
      C.string ~expect:(Printf.sprintf "compute: sm_%d%d" major minor) (List.nth ls 1);
      C.string ~expect:(Printf.sprintf "multiprocessors: %d" i.multiprocessors) (List.nth ls 2);
      C.string ~expect:(Printf.sprintf "memory_mib: %d" (i.total_memory_bytes / 1_048_576)) (List.nth ls 3));
  C.test "Device.name is unchanged and equals info.name" (fun () ->
      C.string ~expect:(Runtime.Device.info ()).name (Runtime.Device.name ()));
  C.run ()
