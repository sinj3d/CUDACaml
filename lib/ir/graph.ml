type t = { name : string; outputs : (string * Tensor.packed) list }

let todo what = failwith ("Graph." ^ what ^ ": TODO")
let create ~name ~outputs = { name; outputs }
let name t = t.name
let outputs t = t.outputs
let params _ = todo "params"
let topological_order _ = todo "topological_order"
let iter _ ~f:_ = todo "iter"
let fold _ ~init:_ ~f:_ = todo "fold"
let fan_out _ _ = todo "fan_out"
let to_dot _ = todo "to_dot"
