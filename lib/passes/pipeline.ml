let default : (module Pass.S) list = [ (module Layout) ]

let run ?(passes = default) graph =
  List.fold_left (fun g (module P : Pass.S) -> P.run g) graph passes
