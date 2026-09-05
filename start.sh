#!/usr/bin/env bash
# Start the v100-skinny server on the configuration in the repo defaults:
# 64k context, E4M3 FP8 KV (patched flash_attn_v100 kernels), k=3 chain-MTP,
# images at 1 MP, tool calling via qwen3_coder.
#
# --open binds 0.0.0.0 -- this server has NO authentication, so anything that
# can reach port 8000 can use the model and read prompts in flight.
#
# Extra args are passed through, so overrides still work, e.g.
#   ~/.start.sh                 # defaults
#   KVDT=auto GMU=0.88 ~/.start.sh    # FP16 KV instead
set -euo pipefail

REPO=/home/mzen17/v100-skinny
CKPT="$REPO/Qwen3.8-27B-NVFP4"

# Pin the interpreter so this works from any shell, not just one with the
# conda env already activated.
export ENV_PREFIX="${ENV_PREFIX:-/home/mzen17/miniconda3/envs/skinny1cat}"

cd "$REPO"
exec bash scripts/serve-qwen38-native.sh --open "$CKPT" "$@"
