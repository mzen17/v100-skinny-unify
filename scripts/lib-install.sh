#!/usr/bin/env bash
# Shared install steps for the two bootstrap entry points.
#
#   scripts/bootstrap-sm70.sh  creates a fresh venv at ./.venv-sm70
#   skinny.sh                  targets an existing conda environment
#
# They differ only in how they obtain an interpreter. Everything after that --
# deploying the fork patches, rebuilding flash_attn_v100 with the E4M3 KV
# patch, and warming the skinny-kernel JIT -- is identical, and lived in both
# files as a duplicated copy until that became a maintenance hazard: a patch
# applied to one and not the other silently produces a broken install. It lives
# here now, and both entry points source it.
#
# Callers must have these in scope before calling skinny_install_stack:
#   REPO_ROOT  PY  SP  NVCC  and the say()/die() helpers.
# Body deliberately NOT re-indented: it contains heredocs whose terminators
# must stay at column 0.

skinny_install_stack() {
[ -n "${REPO_ROOT:-}" ] || die "skinny_install_stack: REPO_ROOT unset"
[ -n "${PY:-}" ]        || die "skinny_install_stack: PY unset"
[ -n "${SP:-}" ]        || die "skinny_install_stack: SP unset"

# ------------------------------------------------------------- fork patches
# Copy tracked files over the installed package, keeping a .pre_bootstrap
# backup of whatever was there. Never hand-edit the installed file: the
# tracked copy in fork_patches/ is the reviewable source of truth.
say "deploying fork patches"
deploy() {  # $1 = tracked file, $2 = path under site-packages
  local src="$REPO_ROOT/fork_patches/$1" dst="$SP/$2"
  [ -f "$src" ] || die "missing tracked patch: $src"
  [ -f "$dst" ] || die "install path not found (wheel version mismatch?): $dst"
  [ -f "$dst.pre_bootstrap" ] || cp -p "$dst" "$dst.pre_bootstrap"
  cp -p "$src" "$dst"
  echo "    $1 -> $2"
}
deploy gdn_attn.py          vllm/v1/attention/backends/gdn_attn.py
deploy gpu_model_runner.py  vllm/v1/worker/gpu_model_runner.py
deploy marlin.py            vllm/model_executor/kernels/linear/nvfp4/marlin.py
deploy modelopt.py          vllm/model_executor/layers/quantization/modelopt.py
deploy torch_utils.py       vllm/utils/torch_utils.py
deploy attention.py         vllm/model_executor/layers/attention/attention.py
deploy custom_all_reduce.py vllm/distributed/device_communicators/custom_all_reduce.py
deploy flash_attn_v100.py   vllm/v1/attention/backends/flash_attn_v100.py
# sm70_native_round.py is deliberately NOT installed — experimental, inert.

# ------------------------------------------------- flash_attn_v100 (E4M3 KV)
# The stock flash_attn_v100 shipped with the wheel refuses FP8 E4M3 in its XQA
# decode path and its FP8 prefill bridge -- both are fp16/E5M2 only. This
# checkpoint declares E4M3 KV, so without this rebuild every long-context
# request falls onto the slow paths (54k prompt: 148 s TTFT, 22 tok/s decode
# instead of 60 s / 50 tok/s). See kernel_patches/README.md for the full
# rationale, measurements and provenance.
#
# Set SKINNY_SKIP_FA_PATCH=1 to skip; the stack still runs, just without E4M3
# on the fast attention paths (and fork_patches/flash_attn_v100.py's widened
# gates would then hit a kernel that raises, so also serve with FP16 KV).
FA_PATCH="$REPO_ROOT/kernel_patches/0001-flash-attn-v100-e4m3-kv.patch"
FA_UPSTREAM_SHA="7aede2cf010d92815c9d7bff25867b4fa009b6cb"
if [ "${SKINNY_SKIP_FA_PATCH:-0}" != "1" ] && [ -f "$FA_PATCH" ]; then
  say "building patched flash_attn_v100 (E4M3 KV support)"
  FA_SRC="${FA_SRC:-$REPO_ROOT/.cache/1Cat-vLLM}"
  if [ ! -d "$FA_SRC/.git" ]; then
    mkdir -p "$(dirname "$FA_SRC")"
    git clone https://github.com/1CatAI/1Cat-vLLM.git "$FA_SRC" \
      || die "could not clone 1Cat-vLLM for the kernel patch"
  fi
  git -C "$FA_SRC" fetch --all --quiet || true
  git -C "$FA_SRC" checkout --quiet "$FA_UPSTREAM_SHA" \
    || die "1Cat-vLLM does not contain pinned commit $FA_UPSTREAM_SHA"
  git -C "$FA_SRC" checkout --quiet -- . 2>/dev/null || true
  git -C "$FA_SRC" apply "$FA_PATCH" \
    || die "kernel patch did not apply cleanly at $FA_UPSTREAM_SHA"
  ( cd "$FA_SRC/flash-attention-v100" \
    && CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}" \
       TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS="${MAX_JOBS:-6}" \
       "$PY" setup.py build_ext --inplace >/dev/null ) \
    || die "flash_attn_v100 rebuild failed"
  FA_SO=$(ls "$FA_SRC"/flash-attention-v100/flash_attn_v100_cuda*.so 2>/dev/null | head -1)
  [ -n "$FA_SO" ] || die "rebuild produced no flash_attn_v100_cuda .so"
  FA_DST="$SP/flash_attn_v100"
  [ -d "$FA_DST" ] || die "flash_attn_v100 is not installed at $FA_DST"
  [ -f "$FA_DST/flash_attn_v100_cuda.so.pre_e4m3" ] || \
    cp -p "$FA_DST"/flash_attn_v100_cuda*.so "$FA_DST/flash_attn_v100_cuda.so.pre_e4m3"
  cp "$FA_SO" "$FA_DST/$(basename "$FA_SO")"
  # Thread the new is_e4m3 argument through the INSTALLED wrapper. Deliberately
  # not copying upstream's newer flash_attn_interface.py wholesale: it adds
  # _assert_decode_launch_covers_seq_lens, which calls seq_lens.max().item() --
  # a device sync on every decode step, measured at about +10 ms/round.
  "$PY" - "$FA_DST/flash_attn_interface.py" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
if "is_e4m3" in s:
    print("    wrapper already carries is_e4m3"); raise SystemExit
old = """    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:"""
new = """    k_scale: float = 1.0,
    v_scale: float = 1.0,
    is_e4m3: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:"""
assert s.count(old) == 1, f"unexpected wrapper shape ({s.count(old)} matches)"
s = s.replace(old, new)
old2 = """        float(k_scale),
        float(v_scale),
    )
    return key_out, value_out"""
new2 = """        float(k_scale),
        float(v_scale),
        bool(is_e4m3),
    )
    return key_out, value_out"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, "w").write(s)
print("    wrapper: is_e4m3 threaded through")
PYEOF
  # flash_attn_v100_cuda lives INSIDE the package, so import it through the
  # package rather than at top level, and prove the E4M3 plumbing is present.
  "$PY" -c "
import inspect, torch
import flash_attn_v100 as F
from flash_attn_v100 import flash_attn_decode_paged_xqa, fp8_e5m2_paged_kv_to_fp16
assert 'is_e4m3' in inspect.signature(fp8_e5m2_paged_kv_to_fp16).parameters, \
    'wrapper is missing is_e4m3 -- the bridge patch did not land'
print('    flash_attn_v100 rebuilt and installed (E4M3 KV enabled)')" \
    || die "patched flash_attn_v100 failed to import"
else
  say "skipping flash_attn_v100 E4M3 patch (SKINNY_SKIP_FA_PATCH=1 or patch absent)"
fi

# ------------------------------------------------------------------ kernels
KERNEL_SRC="$REPO_ROOT/kernels/skinny_kernels.cu"
[ -f "$KERNEL_SRC" ] || die "kernel source missing: $KERNEL_SRC"
say "warming the skinny-kernel JIT build (first build takes a few minutes)"
# Build through the DEPLOYED shim, not a hand-rolled ext.load(). The server
# loads name="skinny_nvfp4_v11" with -O3 --use_fast_math -lineinfo
# (fork_patches/marlin.py:150). torch.utils.cpp_extension keys its build
# directory on the name, and differing flags force a rebuild anyway, so
# building a differently-named extension here warms nothing and validates
# nothing -- the real nvcc build would then land inside the server's boot
# wait. Calling the shim also proves the fork-patch deploy above landed.
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}" \
VLLM_SKINNY_NVFP4_SRC="$KERNEL_SRC" TORCH_CUDA_ARCH_LIST=7.0 "$PY" - <<'PYEOF'
import sys
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
mod = _get_skinny_ext()
if mod is None:
    sys.exit("skinny extension failed to build (see nvcc output above)")
missing = [f for f in ("gemm_qpn2", "gemm_qpn8", "gemm_qpn8_mt2")
           if not hasattr(mod, f)]
if missing:
    sys.exit(f"kernel built but missing entry points: {missing}")
print("    kernels built and all entry points present")
PYEOF


}
