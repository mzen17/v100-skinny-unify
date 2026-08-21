#!/usr/bin/env bash
# Bootstrap the SM70 NVFP4 serving stack from a clean checkout.
#
# Installs the pinned 1Cat-vLLM wheel into an existing environment, deploys the
# fork patches over the installed package, and warms the skinny-kernel JIT
# build so the first served request does not pay for it.
#
#   CONDA_ENV_NAME     Conda environment name to target (default: skinny1cat)
#   VLLM_WHEEL         OPTIONAL. Defaults to the pinned 1Cat-vLLM 1.2.2 release
#   VLLM_WHEEL_SHA256  required alongside a custom VLLM_WHEEL
#   PYTHON_VERSION     default 3.12

set -euo pipefail

# Support running from either repo root or scripts/ subdirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/../fork_patches" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "$SCRIPT_DIR/fork_patches" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-skinny1cat}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# ---------------------------------------------------------------- preflight
say "checking prerequisites"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; need an NVIDIA driver"

CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
echo "    compute capability: $CAPS"
case "$CAPS" in
  *7.0*) ;;
  *) die "no SM70 (compute 7.0 / Volta) GPU found — this stack targets V100.
       Found: $CAPS" ;;
esac

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "    GPUs visible: $NGPU"
[ "$NGPU" -ge "${REQUIRE_GPUS:-2}" ] || die "this configuration needs ${REQUIRE_GPUS:-4} GPUs; $NGPU visible
       (set REQUIRE_GPUS=n only if you intend a different topology --
        the published numbers are TP4 on 4x V100-SXM2-16GB)"

NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do
  [ -x "$c" ] && NVCC="$c" && break
done
[ -n "$NVCC" ] || die "nvcc not found. Install the CUDA toolkit or set NVCC=/path/to/nvcc"
echo "    nvcc: $NVCC ($("$NVCC" --version | tail -1))"
export PATH="$(dirname "$NVCC"):$PATH"

PINNED_WHEEL_URL="https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.2.2/1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl"
PINNED_WHEEL_SHA256="8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6"
VLLM_WHEEL="${VLLM_WHEEL:-$PINNED_WHEEL_URL}"
VLLM_WHEEL_SHA256="${VLLM_WHEEL_SHA256:-}"
[ -n "$VLLM_WHEEL_SHA256" ] || \
  { [ "$VLLM_WHEEL" = "$PINNED_WHEEL_URL" ] && VLLM_WHEEL_SHA256="$PINNED_WHEEL_SHA256"; } || true

PINNED_MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
PINNED_MODEL_REVISION="554ebba9b5f1b79dc11246341960360e6ef05ef4"
echo "    checkpoint: $PINNED_MODEL_REPO @ $PINNED_MODEL_REVISION"
echo "      hf download $PINNED_MODEL_REPO --revision $PINNED_MODEL_REVISION"

# ------------------------------------------------------------------- python
command -v conda >/dev/null || die "conda binary not found in PATH"

say "resolving conda environment '$CONDA_ENV_NAME'"
ENV_PREFIX="$(conda info --envs | awk -v env="$CONDA_ENV_NAME" '$1 == env {print $NF; exit}')"

if [ -z "$ENV_PREFIX" ] || [ ! -d "$ENV_PREFIX" ]; then
  die "Conda environment '$CONDA_ENV_NAME' not found. Run: conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION"
fi

PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || die "python binary not found at $PY. Install python: conda install -n $CONDA_ENV_NAME -y python=$PYTHON_VERSION pip"
say "using existing environment: $ENV_PREFIX ($("$PY" --version))"

# Check if 1cat-vllm 1.2.2 is already installed
INSTALLED=$("$PY" - <<'EOF' 2>/dev/null || true
import importlib.metadata as md
for n in ("1cat-vllm", "1cat_vllm"):
    try:
        print(md.distribution(n).version); break
    except Exception:
        pass
else:
    print("")
EOF
)

if [ "$INSTALLED" = "1.2.2" ]; then
  say "1cat-vllm 1.2.2 already installed, skipping download and wheel install"
else
  say "installing pinned vLLM wheel"
  CACHE_DIR="$REPO_ROOT/.cache"
  mkdir -p "$CACHE_DIR"

  WHEEL_LOCAL="$VLLM_WHEEL"
  case "$VLLM_WHEEL" in
    http://*|https://*)
      WHEEL_LOCAL="$CACHE_DIR/$(basename "${VLLM_WHEEL%%\?*}")"
      if [ -f "$WHEEL_LOCAL" ] && [ -n "$VLLM_WHEEL_SHA256" ]; then
        cached_sha=$(sha256sum "$WHEEL_LOCAL" 2>/dev/null | cut -d" " -f1 || shasum -a 256 "$WHEEL_LOCAL" | cut -d" " -f1)
        if [ "$cached_sha" = "$VLLM_WHEEL_SHA256" ]; then
          say "using cached wheel from $WHEEL_LOCAL"
        else
          say "cached wheel digest mismatch, re-downloading"
          curl -fL --retry 3 -o "$WHEEL_LOCAL" "$VLLM_WHEEL" || die "wheel download failed: $VLLM_WHEEL"
        fi
      else
        say "downloading wheel to cache"
        curl -fL --retry 3 -o "$WHEEL_LOCAL" "$VLLM_WHEEL" || die "wheel download failed: $VLLM_WHEEL"
      fi
      ;;
  esac
  [ -f "$WHEEL_LOCAL" ] || die "wheel not found: $WHEEL_LOCAL"

  if [ -z "${VLLM_WHEEL_SHA256:-}" ]; then
    [ "${ALLOW_UNVERIFIED_WHEEL:-0}" = 1 ] || die \
      "no SHA256 for $VLLM_WHEEL
     Pin one:    VLLM_WHEEL_SHA256=<digest> ...
     Or waive:  ALLOW_UNVERIFIED_WHEEL=1 ...  (not reproducible)"
    echo "    WARNING: installing an unverified wheel" >&2
  else
    got=$(sha256sum "$WHEEL_LOCAL" 2>/dev/null | cut -d" " -f1) \
      || got=$(shasum -a 256 "$WHEEL_LOCAL" | cut -d" " -f1)
    [ "$got" = "$VLLM_WHEEL_SHA256" ] || die \
      "wheel digest mismatch
     expected $VLLM_WHEEL_SHA256
     got      $got"
    echo "    wheel sha256 verified"
  fi
  "$PY" -m pip install "$WHEEL_LOCAL"
fi

# Check if tilelang and apache-tvm-ffi are already at 0.1.10
CHECK_DEPS=$("$PY" - <<'EOF' 2>/dev/null || echo "MISSING"
import importlib.metadata as md
try:
    t = md.distribution("tilelang").version
    a = md.distribution("apache-tvm-ffi").version
    print(f"{t}:{a}")
except Exception:
    print("MISSING")
EOF
)

if [ "$CHECK_DEPS" = "0.1.10:0.1.10" ]; then
  say "tilelang 0.1.10 and apache-tvm-ffi 0.1.10 already installed, skipping pip install"
else
  say "pinning tilelang 0.1.10 + apache-tvm-ffi 0.1.10 (required for SM70)"
  "$PY" -m pip install "tilelang==0.1.10" "apache-tvm-ffi==0.1.10"
  echo "    NOTE: pip reported a dependency conflict against the wheel's declared"
  echo "          tilelang/apache-tvm-ffi 0.1.9 pins. That is EXPECTED: 0.1.9 does"
  echo "          not build on SM70. The versions installed above are correct."
fi

"$PY" -c "import tilelang; print('    tilelang', tilelang.__version__)" \
  || die "tilelang failed to import -- the environment is broken.
   This is usually an apache-tvm-ffi version mismatch; expected 0.1.10.
   Check: $PY -m pip show apache-tvm-ffi"

SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$SP/vllm" ] || die "vllm did not install into $SP"
say "site-packages: $SP"

# ---------------------------------------------- fork patches + kernels
# Shared with skinny.sh; see scripts/lib-install.sh.
# shellcheck source=scripts/lib-install.sh
. "$REPO_ROOT/scripts/lib-install.sh"
skinny_install_stack

cat <<EOF

Bootstrap complete.

  conda env    : $CONDA_ENV_NAME ($ENV_PREFIX)
  kernels      : $KERNEL_SRC
  next         : bash scripts/serve-qwen38-native.sh <checkpoint-dir>

Set VLLM_SKINNY_NVFP4_SRC=$KERNEL_SRC in the serving environment
(serve-qwen38-native.sh does this for you).
EOF
