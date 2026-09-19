type t = int

let counter = ref 0

let fresh () =
  incr counter;
  !counter

let to_int t = t
let compare = Int.compare
let equal = Int.equal
let to_string t = "n" ^ Int.to_string t
