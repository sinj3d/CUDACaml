(lang dune 3.0)
(name cudacaml)
(generate_opam_files true)

(package
 (name cudacaml)
 (synopsis "An embedded array DSL in OCaml that compiles to CUDA")
 (depends
  (ocaml (>= 5.1))
  dune
  (cudajit (>= 0.7))
  ))
