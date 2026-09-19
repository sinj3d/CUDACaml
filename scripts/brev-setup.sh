#!/usr/bin/env bash
#
# brev.dev bootstrap for ocaml-cuda.
#
# Takes a fresh Ubuntu 22.04 / 24.04 GPU box (an A100 on brev.dev is the case
# this was written for) from empty to a checkout where
#
#     make build && make unit
#
# are green, and prints the two commands worth running next, `make system`
# and `make record`.
#
# Idempotent: every step checks first and says what it found, so re-running
# after a failure costs only the step that failed. Nothing here assumes that
# /usr/local/cuda exists.
#
#     bash scripts/brev-setup.sh
#
# Knobs, all optional -- a fork should never need the script edited:
#
#     OCAML_CUDA_REPO          git URL to clone   (default: this project)
#     OCAML_CUDA_DIR           where to clone it  (default: ~/ocaml-cuda)
#     OCAML_CUDA_SWITCH        opam switch name   (default: ocaml-cuda)
#     OCAML_CUDA_OCAML         compiler version   (default: 5.4.0)
#     OCAML_CUDA_CUDA_RUNFILE  12.9 installer URL, for the 13.x fallback
#
set -euo pipefail

REPO_URL="${OCAML_CUDA_REPO:-https://github.com/sjin2/ocaml-cuda}"
REPO_DIR="${OCAML_CUDA_DIR:-$HOME/ocaml-cuda}"
SWITCH="${OCAML_CUDA_SWITCH:-ocaml-cuda}"
OCAML_VERSION="${OCAML_CUDA_OCAML:-5.4.0}"
CUDA_FALLBACK_PREFIX="$HOME/cuda-12.9"
CUDA_RUNFILE_URL="${OCAML_CUDA_CUDA_RUNFILE:-https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run}"

APT_PACKAGES="build-essential git m4 pkg-config libffi-dev opam unzip bubblewrap curl"

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    !! %s\n' "$*" >&2; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    warn "not root and no sudo: the apt step will fail if anything is missing"
  fi
fi

# --------------------------------------------------------------- 1. packages

step "System packages"
missing=""
for pkg in $APT_PACKAGES; do
  if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "ok installed"; then
    info "found $pkg"
  else
    missing="$missing $pkg"
  fi
done
if [ -n "$missing" ]; then
  info "installing:$missing"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y $missing
else
  info "nothing to install"
fi
info "found opam $(opam --version 2>/dev/null || echo '?')"

# ------------------------------------------------------------------- 2. gcc

step "C compiler"
if command -v gcc >/dev/null 2>&1; then
  gcc_version="$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion)"
  gcc_major="${gcc_version%%.*}"
  info "found gcc $gcc_version"
  if [ "${gcc_major:-0}" -ge 14 ]; then
    warn "gcc >= 14 rejects cudajit's generated stubs: it promotes"
    warn "-Wincompatible-pointer-types to an error, and the two offending"
    warn "casts are benign const/typedef mismatches (README.md, 'GCC 14"
    warn "rejects cudajit's generated stubs'). Relax that warning for the"
    warn "stub build, or build the switch against an older compiler:"
    warn "    sudo apt-get install -y gcc-13 g++-13"
    warn "    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130"
    warn "    # then re-run this script, which bakes gcc-13 into the switch"
    export CFLAGS="${CFLAGS:-} -Wno-incompatible-pointer-types"
    info "exported CFLAGS=$CFLAGS for this run"
  fi
else
  warn "no gcc found; the package step above should have installed one"
fi

# ------------------------------------------------------------------ 3. CUDA

# The toolkit version at a prefix, or nothing. Two sources, because a runfile
# install has nvcc while a deb install may only leave version.json behind.
cuda_version_at() {
  prefix="$1"
  if [ -x "$prefix/bin/nvcc" ]; then
    "$prefix/bin/nvcc" --version |
      sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p'
  elif [ -r "$prefix/version.json" ]; then
    tr -d ' \n' < "$prefix/version.json" |
      sed -n 's/.*"cuda":{[^}]*"version":"\([0-9][0-9.]*\)".*/\1/p'
  fi
}

step "CUDA toolkit"
CUDA_PATH="${CUDA_PATH:-}"
cuda_version=""
if [ -n "$CUDA_PATH" ]; then
  info "CUDA_PATH is already set to $CUDA_PATH"
  cuda_version="$(cuda_version_at "$CUDA_PATH")"
elif command -v nvcc >/dev/null 2>&1; then
  CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
  cuda_version="$(cuda_version_at "$CUDA_PATH")"
elif [ -r /usr/local/cuda/version.json ]; then
  CUDA_PATH="/usr/local/cuda"
  cuda_version="$(cuda_version_at "$CUDA_PATH")"
fi

if [ -n "$cuda_version" ]; then
  info "toolkit $cuda_version at $CUDA_PATH"
else
  info "no toolkit found: nvcc is not on PATH and /usr/local/cuda/version.json"
  info "does not exist"
fi

cuda_major="${cuda_version%%.*}"
need_fallback=no
case "$cuda_major" in
13 | 14)
  need_fallback=yes
  info "CUDA $cuda_version is too new: cudajit 0.7.2 does not build against"
  info "13.x headers -- cuCtxCreate maps to cuCtxCreate_v4 and both"
  info "nvrtcCompileProgram and cuStreamGetId changed signatures. See"
  info "README.md, 'Use CUDA 12.x, not 13.x'. The A100 is sm_80, so any 12.x"
  info "will do: the 12.8 floor is a Blackwell fact and does not apply here."
  ;;
"")
  need_fallback=yes
  info "installing 12.9, since there is no toolkit to build against"
  ;;
*)
  info "CUDA $cuda_major.x is supported; using it as is"
  ;;
esac

if [ "$need_fallback" = yes ]; then
  if [ -x "$CUDA_FALLBACK_PREFIX/bin/nvcc" ]; then
    info "found an existing side-by-side install at $CUDA_FALLBACK_PREFIX"
  else
    info "installing CUDA 12.9 side by side into $CUDA_FALLBACK_PREFIX"
    info "(--toolkit only: the driver already on the box is left alone)"
    runfile="${TMPDIR:-/tmp}/cuda_12.9_linux.run"
    if [ -s "$runfile" ]; then
      info "reusing the already-downloaded runfile $runfile"
    else
      info "downloading $CUDA_RUNFILE_URL"
      curl -fL --retry 3 -o "$runfile.part" "$CUDA_RUNFILE_URL"
      mv "$runfile.part" "$runfile"
    fi
    sh "$runfile" --silent --toolkit --toolkitpath="$CUDA_FALLBACK_PREFIX" \
      --installpath="$CUDA_FALLBACK_PREFIX" --no-drm --override
  fi
  CUDA_PATH="$CUDA_FALLBACK_PREFIX"
  cuda_version="$(cuda_version_at "$CUDA_PATH")"
  info "toolkit $cuda_version at $CUDA_PATH"
  printf '\n    export CUDA_PATH=%s\n' "$CUDA_FALLBACK_PREFIX"
  info "(put that in ~/.bashrc; the Makefile also finds this prefix on its own)"
  printf '\n'
fi

if [ -n "$CUDA_PATH" ]; then
  export CUDA_PATH
  export PATH="$CUDA_PATH/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  info "device: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
  warn "no nvidia-smi: there is no driver here and every GPU test will skip"
fi

# ------------------------------------------------------------------ 4. opam

step "opam switch $SWITCH ($OCAML_VERSION)"
if [ -d "$HOME/.opam" ]; then
  info "found $HOME/.opam"
else
  info "opam init --disable-sandboxing (bwrap does not work in a container)"
  opam init --disable-sandboxing --bare -y
fi
eval "$(opam env --safe 2>/dev/null || true)"

if opam switch list --short 2>/dev/null | grep -qx "$SWITCH"; then
  info "found switch $SWITCH"
else
  info "creating switch $SWITCH with OCaml $OCAML_VERSION"
  opam switch create "$SWITCH" "$OCAML_VERSION"
fi
eval "$(opam env --switch="$SWITCH" --set-switch)"
info "using $(ocaml -version 2>/dev/null || echo 'no ocaml on PATH yet')"

step "opam packages"
for pkg in dune cudajit; do
  if opam list --installed --short 2>/dev/null | grep -qx "$pkg"; then
    info "found $pkg $(opam show -f version "$pkg" 2>/dev/null || echo '?')"
  else
    info "installing $pkg"
    opam install -y "$pkg"
  fi
done

# --------------------------------------------------------------- 5. checkout

step "Checkout"
if [ -d "$REPO_DIR/.git" ]; then
  info "found $REPO_DIR; pulling"
  git -C "$REPO_DIR" pull --ff-only
else
  info "cloning $REPO_URL into $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
info "at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# ------------------------------------------------------------------ 6. build

step "make env"
make env
step "make build"
make build
step "make unit"
make unit

step "Done: $REPO_DIR builds and its unit suite passes"
info "next:"
info "    cd $REPO_DIR"
info "    make system     # the gated end-to-end suite (needs the GPU)"
info "    make record     # bench f32 and f64, append bench/results/<card>.md"
if [ "$need_fallback" = yes ]; then
  info ""
  info "remember that the toolkit is not the system one:"
  info "    export CUDA_PATH=$CUDA_FALLBACK_PREFIX"
fi
