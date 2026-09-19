(lang dune 3.0)
(name ocaml_cuda)
(generate_opam_files true)

(package
 (name ocaml_cuda)
 (synopsis "An embedded array DSL in OCaml that compiles to CUDA")
 (depends
  (ocaml (>= 5.1))
  dune
  ; (cudajit (>= 0.7)) ; enable when lib/runtime is wired to the real driver API
  ))
