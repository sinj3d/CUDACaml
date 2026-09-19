type t = int list

let scalar = []
let of_dims dims = dims
let dims t = t
let rank = List.length
let numel t = List.fold_left ( * ) 1 t
let equal = List.equal Int.equal
let to_string t = "[" ^ String.concat "," (List.map Int.to_string t) ^ "]"
