(* Minimal test harness. No external dependency so every task can run its
   tests with nothing but dune installed.

   Usage:
     let () =
       Check.test "name" (fun () -> Check.int ~expect:3 (1 + 2));
       Check.run ()
   [Check.run] exits 1 if any test failed. *)

let failures = ref 0
let count = ref 0

let test name f =
  incr count;
  match f () with
  | () -> Printf.printf "  ok    %s\n%!" name
  | exception e ->
      incr failures;
      Printf.printf "  FAIL  %s: %s\n%!" name (Printexc.to_string e)

let fail fmt = Printf.ksprintf failwith fmt

let int ~expect got =
  if expect <> got then fail "expected %d, got %d" expect got

let bool ~expect got =
  if expect <> got then fail "expected %b, got %b" expect got

let string ~expect got =
  if expect <> got then fail "expected %S, got %S" expect got

let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = sub || go (i + 1)) in
  if not (go 0) then fail "expected %S to contain %S" s sub

let float ?(tol = 1e-5) ~expect got =
  if Float.abs (expect -. got) > tol then fail "expected %g, got %g" expect got

let floats ?(tol = 1e-5) ~expect got =
  if List.length expect <> List.length got then
    fail "length: expected %d, got %d" (List.length expect) (List.length got);
  List.iteri
    (fun i (e, g) ->
      if Float.abs (e -. g) > tol then fail "index %d: expected %g, got %g" i e g)
    (List.combine expect got)

let raises f =
  match f () with
  | exception _ -> ()
  | _ -> fail "expected an exception"

let skip reason = Printf.printf "  SKIP  %s\n%!" reason

let run () =
  Printf.printf "%d tests, %d failures\n%!" !count !failures;
  exit (if !failures = 0 then 0 else 1)
