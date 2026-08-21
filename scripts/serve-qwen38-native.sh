#!/usr/bin/env bash
# Serve Qwen3.8-27B NVFP4 on 4x V100 (SM70), k=7 chain-MTP, native NVFP4
# lm_head, fp16 KV cache — the configuration the published numbers were
# measured on.
#
#   bash scripts/serve-qwen38-native.sh <checkpoint-dir>
#
# Overridable: ENV_PREFIX K GMU MML MNS MBT PORT DECODE_PARTITION THINKING
#
# By default this uses whatever python is first on PATH -- i.e. your active
# conda env. Set ENV_PREFIX to pin a specific environment instead.
#
# The boot is GATED on OBSERVED EXECUTION, not on configuration strings: the
# script refuses to report success unless the server
# actually served the configuration asked for. Five checks, each of which has
# caught a silently-wrong boot at least once:
#
#   1. GPUs must be free first. Booting over an occupied GPU yields a server
#      that runs at a fraction of its speed instead of failing.
#   2. served speculative depth == requested k.
#   3. lm_head is served by the skinny kernel from the checkpoint's OWN
#      NVFP4 codes -- witnessed by a vocab-shaped GEMM routing to qpn.
#      (VLLM_SKINNY_LMHEAD_NATIVE is inert for ModelOpt checkpoints whose
#      lm_head is already quantized in-checkpoint: that flag pulls native
#      codes from a source shard for checkpoints whose head is UNquantized.
#      Gating on its log line would fail every correct boot here.)
#   4. kv_cache_dtype == auto. A ModelOpt checkpoint may declare
#      kv_cache_quant_algo=FP8, which describes how its WEIGHTS were made;
#      honouring it below SM80 loses the tensor-core decode route and costs
#      +4.82 ms/round. The fork declines it — this verifies the decline.
#   5. decode route == qpn (the skinny tensor-core path), not a scalar
#      fallback.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# --open binds every interface instead of loopback. It is the single explicit
# act that acknowledges this server has NO authentication -- see the HOST block
# below. Anything else is treated as the checkpoint directory.
OPEN=0
CKPT=""
for arg in "$@"; do
  case "$arg" in
    --open) OPEN=1 ;;
    -h|--help)
      echo "usage: $0 [--open] <checkpoint-dir>" >&2
      echo "  --open   bind 0.0.0.0 instead of 127.0.0.1 (UNAUTHENTICATED)" >&2
      exit 0 ;;
    -*) echo "ERROR: unknown option '$arg'" >&2; exit 2 ;;
    *)  [ -z "$CKPT" ] || { echo "ERROR: more than one checkpoint given" >&2; exit 2; }
        CKPT="$arg" ;;
  esac
done
[ -n "$CKPT" ] || { echo "usage: $0 [--open] <checkpoint-dir>" >&2; exit 2; }
[ -f "$CKPT/config.json" ] || { echo "ERROR: no config.json in $CKPT" >&2; exit 2; }
CKPT="$(cd "$CKPT" && pwd)"

# Use the active environment's interpreter. ENV_PREFIX still wins when set,
# so a bootstrap-built prefix keeps working unchanged.
if [ -n "${ENV_PREFIX:-}" ]; then
  PY="$ENV_PREFIX/bin/python"
else
  PY="$(command -v python || command -v python3 || true)"
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || {
  echo "ERROR: no usable python — activate your environment, or set ENV_PREFIX" >&2; exit 2; }
"$PY" -c 'import vllm' 2>/dev/null || {
  echo "ERROR: vllm is not importable from $PY" >&2
  echo "       run scripts/bootstrap-sm70.sh with this environment active" >&2; exit 2; }

# k=3, not 7: with the 64k window below this is long-context serving, where
# k=3 is the documented profile (see "Depth profiles" in the README).
K="${K:-3}"; K1=$((K + 1)); K2=$((K1 * 2))
# Bind to loopback by default. This server has NO authentication: anything that
# can reach the port can use the model, read any prompt in flight, and drive
# the box. Exposing it is a deliberate act, so it needs an explicit HOST and an
# acknowledgement -- and even then it belongs behind a firewall or a proxy that
# terminates auth. vLLM's own security guidance is that its API keys do not
# protect every endpoint.
if [ "$OPEN" = 1 ]; then HOST="${HOST:-0.0.0.0}"; else HOST="${HOST:-127.0.0.1}"; fi
# Probe the address we actually bound. A fixed 127.0.0.1 probe silently times
# out under HOST=::1, and 0.0.0.0 is not a connect address at all.
case "$HOST" in
  ::1)     PROBE="[::1]" ;;
  0.0.0.0) PROBE="127.0.0.1" ;;
  ::)      PROBE="[::1]" ;;
  *)       PROBE="$HOST" ;;
esac
case "$HOST" in
  127.0.0.1|localhost|::1) ;;
  *)
    [ "$OPEN" = 1 ] || [ "${I_UNDERSTAND_THIS_IS_UNAUTHENTICATED:-0}" = 1 ] || {
      echo "REFUSING to bind $HOST: this server is unauthenticated." >&2
      echo "  Keep the default (127.0.0.1) and use an SSH tunnel:" >&2
      echo "    ssh -N -L 8000:127.0.0.1:8000 <user>@<host>" >&2
      echo "  Or, if you really intend to expose it on a trusted network:" >&2
      echo "    $0 --open <checkpoint-dir>" >&2
      exit 2; }
    echo "==> WARNING: binding $HOST with no authentication. Firewall this." >&2 ;;
esac
# GMU 0.88, not the 0.93 of the all-NVFP4 profile: verbatim mixed FP8+NVFP4
# weights alongside an fp16 KV cache do not fit at 0.93 on 16 GB cards.
# KV cache dtype. Default "auto", which this fork resolves to FP16 on SM70 by
# declining the checkpoint's FP8-KV directive (see fork_patches/torch_utils.py).
#
# KVDT=fp8 selects the checkpoint's own E4M3 KV cache. That halves KV bytes per
# token, which is what makes a 64k window fit on a 16 GB card -- and with the
# rebuilt kernel from kernel_patches/ it keeps the tensor-core XQA decode path
# and the FP8 prefill bridge, so it costs nothing in quality (verified
# byte-identical output). WITHOUT that rebuilt kernel it silently falls onto
# scalar_paged decode and the slow prefill: 54k prompt goes 60 s -> 148 s TTFT.
# Gate 5/6 below adapt to the arm; gates 7/8 are asserted either way.
#
# Do NOT use fp8_e5m2 here: it reaches the same fast paths, but this fork forces
# UNIT KV scales for e5m2 on a quantized checkpoint, discarding the calibrated
# ones (measured mean KL 0.23 nats and +25% reasoning-trace length).
KVDT="${KVDT:-fp8}"

# Multimodal. This checkpoint is genuinely multimodal -- 333 vision tensors, a
# 27-layer tower -- and it reads images correctly on Volta. Images are enabled
# by default at a 1-megapixel ceiling.
#
# The ceiling is load-bearing, not cosmetic. The processor's own default is
# 16.7 MP, and vLLM sizes its memory profile against the LARGEST permitted
# image, so leaving it uncapped reserves ~0.8 GiB that never gets used and
# costs more KV than the vision weights themselves:
#
#     images off ............ 129,901 KV tokens   (1.98x concurrency at 64k)
#     images on, 1 MP cap ... 107,666             (1.64x)
#     images on, uncapped ....  72,557            (1.11x)
#
# Long-context speed is unaffected either way (54k prompt: 61.1 s vs 60.0 s
# TTFT). Oversized images are downscaled to the cap, not rejected.
#
# Set MM_LIMIT='{"image":0,"video":0}' for a text-only server; that is the
# configuration the published throughput numbers were measured on.
MM_LIMIT="${MM_LIMIT:-{\"image\":1,\"video\":0\}}"
MM_PIXELS="${MM_PIXELS:-1048576}"
# 0.92, not 0.88: FP8 KV needs the extra budget to hold a 64k window. This
# box's usable range is narrow and non-monotonic -- 0.90 boots and then OOMs
# on the first prefill, 0.93 dies in CUDA-graph capture. Lower it to 0.88 if
# you drop back to FP16 KV (KVDT=auto), which needs far less headroom.
GMU="${GMU:-0.92}"
# 64k. Only reachable with KVDT=fp8; FP16 KV tops out near 40k on a 16 GB
# card. Declared context is free -- see the README.
MML="${MML:-65536}"
MNS="${MNS:-1}"
MBT="${MBT:-4096}"
PORT="${PORT:-8000}"
THINKING="${THINKING:-true}"
# Pin the decode partition size. The default selector switches to 1024 at
# max_model_len >= 32768, and the MTP verify path (which arrives as q>1) has
# no active-partition skip, so a large MML taxes every round for capacity it
# never uses. 256 recovers it. Raise this for genuinely long contexts
# (>32k actual), where 1024 is the default for a reason.
DECODE_PARTITION="${DECODE_PARTITION:-256}"
TP="${TP:-2}"
LOG="${LOG:-$REPO_ROOT/serve.log}"

# ---- 1. never boot over occupied GPUs ------------------------------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${USED:-0}" -ge 200 ]; then
  echo "ABORT: GPUs are occupied (${USED} MiB in use). Finish the teardown first." >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

# NUMA pin is part of the launch config, not an optimisation: all four GPUs
# sit on socket 0 and an unpinned boot re-rolls thread/page placement for a
# +/-3% round-time lottery -- larger than the entire parity margin we report.
# SKINNY_NUMA=0 disables it for experiments.
if [ "${SKINNY_NUMA:-1}" = "1" ] && command -v numactl >/dev/null; then
  NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"
else
  NUMA_PREFIX=""
  [ "${SKINNY_NUMA:-1}" = "1" ] && echo "WARNING: numactl not found; boots will vary ~3% run to run" >&2
fi

# Any exit before READY must not strand a server holding four GPUs.
PIDFILE="${PIDFILE:-$REPO_ROOT/serve.pid}"
export PIDFILE
cleanup_on_fail() {
  [ -s "$PIDFILE" ] || return 0
  local pid; pid=$(cat "$PIDFILE")
  kill -0 "$pid" 2>/dev/null || return 0
  echo "==> tearing down the server (pid $pid) so it does not hold the GPUs" >&2
  kill -TERM "-$(ps -o pgid= "$pid" 2>/dev/null | tr -d " ")" 2>/dev/null \
    || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  rm -f "$PIDFILE"
}
trap 'cleanup_on_fail' INT TERM

rm -f "$LOG"
# +rms_norm_gated (2026-08-20). The SM70 profile sets custom_ops=['none'],
# which disables EVERY CustomOp -- including RMSNormGated. That sent the gated
# RMSNorm in all 48 GDN linear-attention layers through
# `RMSNormGated.forward_static`, whose own docstring calls it "Pure-PyTorch RMS
# normalization": x.float() / pow(2).mean() / rsqrt / silu / .to(orig_dtype) as
# separate eager aten kernels, 387 launches each per capture. Re-enabling just
# this one op routes it to the fused Triton `rmsnorm_fn` in forward_cuda.
# Measured A/B (k=3, fp8 KV, 64k, identical build): ms/round 42.82 -> 38.89,
# 42.50 -> 38.57, 42.51 -> 38.42 (-9.2 to -9.6%); GPU kernels per capture
# 19,096 -> 15,327; the mean/rsqrt/silu kernels drop to exactly zero.
# Correctness: AIME fixture 01 medium 5 seeds 5/5 (all 277), and a
# teacher-forced KL divergence over 346 positions of mean 3.96e-06 nats with
# 100% top-1 agreement -- fp reassociation noise, not a behavioural change.
# Scope: this enables ONE op by name. The blanket 'none' still governs the rest.
# Only pass processor kwargs when images are actually enabled; an empty
# expansion keeps the text-only command line byte-identical to before.
case "$MM_LIMIT" in
  *'"image":0'*) MM_PROC_ARGS="" ;;
  *) MM_PROC_ARGS="--mm-processor-kwargs {\"size\":{\"longest_edge\":$MM_PIXELS,\"shortest_edge\":65536}}" ;;
esac

# Tool calling uses qwen3_coder, not hermes. This model emits Qwen's XML
# tool-call format; the hermes parser does not recognise it, so requests
# still succeed but `tool_calls` comes back empty -- a silent failure.
# QPN_UNIFY=1 skips building the Marlin repack for QPN-eligible layers, saving
# one resident copy of every NVFP4 weight. Required on 16 GB cards -- without
# it the loader holds both the repack and the qpn buffer and OOMs during model
# load. Harmless on larger cards. Set VLLM_SKINNY_QPN_UNIFY=0 to opt out.
echo "==> serving $CKPT  (k=$K, GMU=$GMU, MML=$MML, partition=$DECODE_PARTITION)"
echo "==> interpreter: $PY"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
TORCH_CUDA_ARCH_LIST=7.0 \
VLLM_SM70_NVFP4_TURBOMIND=0 \
VLLM_SM70_QUANT_BACKEND=marlin \
VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 \
VLLM_SKINNY_NVFP4=1 \
VLLM_SKINNY_QPN=1 \
VLLM_SKINNY_QPN2=1 \
VLLM_SKINNY_LMHEAD=1 \
VLLM_SKINNY_LMHEAD_NATIVE=1 \
VLLM_SKINNY_DROP_CT=1 \
VLLM_SKINNY_QPN_UNIFY="${VLLM_SKINNY_QPN_UNIFY:-1}" \
VLLM_SKINNY_NVFP4_SRC="$REPO_ROOT/kernels/skinny_kernels.cu" \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 \
VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1 \
VLLM_SM70_QPN8_MT2=1 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE="$DECODE_PARTITION" \
setsid $NUMA_PREFIX "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" \
  --served-model-name qwen3.8-27b \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size "$TP"  \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MML" \
  --kv-cache-dtype "$KVDT" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "$MBT" \
  --limit-mm-per-prompt "$MM_LIMIT" \
  ${MM_PROC_ARGS} \
  --default-chat-template-kwargs "{\"enable_thinking\":$THINKING}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config "{\"cudagraph_capture_sizes\":[$K1,$K2],\"custom_ops\":[\"none\",\"+rms_norm_gated\"]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\",\"use_local_argmax_reduction\":true}" \
  --host "$HOST" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"

echo "==> waiting for the server (first boot compiles graphs; several minutes)"
UP=0
for i in $(seq 1 600); do
  if curl -sf -o /dev/null --max-time 2 "http://$PROBE:$PORT/v1/models"; then UP=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED — last lines:" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 2
done
[ "$UP" = 1 ] || { echo "TIMEOUT waiting for the server" >&2; tail -30 "$LOG" >&2
                   cleanup_on_fail; exit 1; }

# Warm one request: the native-lm_head line is emitted lazily by the first
# logits call, so gate 3 is not decidable until something has been served.
WARM=$(curl -sf --max-time 300 "http://$PROBE:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say ready."}],"temperature":0,"max_completion_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null) || WARM=""


# ---- boot gate -----------------------------------------------------------
FAIL=0
gate() { if [ "$2" = ok ]; then echo "    PASS  $1"; else echo "    FAIL  $1 — $3"; FAIL=1; fi; }
echo "==> boot gate"

# A 4xx/5xx or malformed body must not reach READY: everything downstream
# reads log lines that only exist because a real completion happened.
if printf '%s' "$WARM" | "$PY" -c 'import json,sys
d=json.load(sys.stdin)
c=d["choices"][0]["message"].get("content")
assert isinstance(c,str), "no string content"
assert d.get("usage",{}).get("completion_tokens",0)>0, "zero completion tokens"' 2>/dev/null; then
  gate "warm request returned a valid completion" ok
else
  gate "warm request returned a valid completion" fail "HTTP error or malformed JSON"
fi

SERVED_K=$(grep -o "num_speculative_tokens[^,}]*" "$LOG" | head -1 | grep -o "[0-9]*$")
[ "${SERVED_K:-}" = "$K" ] && gate "served depth == $K" ok \
  || gate "served depth == $K" fail "log says '${SERVED_K:-<none>}'"

# lm_head is [vocab, hidden]; at TP4 each rank owns vocab/4 rows. Any GEMM
# whose N is far larger than a trunk projection is the vocab shard, and it
# must route to a qpn kernel -- that is lm_head being served from the
# checkpoint's own 4-bit codes rather than repacked.
LMHEAD_ROUTE=$(grep -ohE "route map: M=[0-9]+ N=[0-9]{5,} K=[0-9]+ -> [a-z0-9]+" "$LOG" \
               | grep -oE "\-> [a-z0-9]+$" | sed 's/-> //' | sort -u | tr '\n' ' ')
case "$LMHEAD_ROUTE" in
  *qpn*) gate "lm_head served from checkpoint codes (qpn)" ok ;;
  "")    gate "lm_head served from checkpoint codes (qpn)" fail "no vocab-shaped GEMM observed" ;;
  *)     gate "lm_head served from checkpoint codes (qpn)" fail "routed to: $LMHEAD_ROUTE" ;;
esac

grep -q "falling back to requant pack\|packing from the model's own weights" "$LOG" \
  && gate "no lm_head repack fallback" fail "silent lm_head downgrade" \
  || gate "no lm_head repack fallback" ok

# kv_cache_dtype == auto is NOT sufficient and used to be the whole check.
# `auto` means "take the checkpoint's word for it", and this checkpoint asks
# for fp8_e4m3 -- `auto` is the very path through which the wrong FP8-KV route
# was selected before the loader fix. The load-bearing witness is the loader
# explicitly DECLINING the directive.
KVD=$(grep -o "kv_cache_dtype=[a-z0-9_]*" "$LOG" | head -1 | cut -d= -f2)
if [ "$KVDT" = "auto" ]; then
  [ "${KVD:-auto}" = "auto" ] && gate "kv_cache_dtype == auto" ok \
    || gate "kv_cache_dtype == auto" fail "got '$KVD' — checkpoint KV directive was honoured"

  grep -q "Ignoring the checkpoint's kv_cache quantization directive" "$LOG" \
    && gate "FP16 KV resolved (checkpoint FP8-KV directive declined)" ok \
    || gate "FP16 KV resolved (checkpoint FP8-KV directive declined)" fail \
         "no decline line — KV storage is NOT proven FP16"
else
  # FP8-KV arm: the request was explicit, so assert it was SERVED as asked
  # rather than asserting the decline. The fast-path gates below are NOT
  # relaxed -- an FP8 arm that lost XQA is exactly the failure worth catching,
  # and it is what an unpatched flash_attn_v100 produces.
  [ "${KVD:-}" = "$KVDT" ] && gate "kv_cache_dtype == $KVDT (explicit)" ok \
    || gate "kv_cache_dtype == $KVDT (explicit)" fail "log says '${KVD:-<none>}'"
  case "$KVDT" in
    fp8|fp8_e4m3)
      gate "E4M3 KV keeps the checkpoint's calibrated scales" ok ;;
    *)
      gate "E4M3 KV keeps the checkpoint's calibrated scales" fail \
        "KVDT=$KVDT is not E4M3; e5m2 discards calibrated KV scales" ;;
  esac
fi

# Honouring the FP8-KV directive on SM70 silently drops decode onto the scalar
# paged route and costs 4.82 ms/round. Zero is the only acceptable count.
SCALAR=$(grep -c "scalar_paged" "$LOG" || true)
[ "${SCALAR:-0}" = 0 ] && gate "zero scalar_paged attention calls" ok \
  || gate "zero scalar_paged attention calls" fail "$SCALAR call(s) — decode fell off the tensor-core route"

# The tensor-core decode-attention path announces itself once, on first use.
# Only meaningful when speculation is on; k=0 has no verifier.
if [ "$K" -gt 0 ] 2>/dev/null; then
  grep -q "XQA path active" "$LOG" \
    && gate "XQA tensor-core decode attention active" ok \
    || gate "XQA tensor-core decode attention active" fail "XQA path never announced"
fi

# `route=qpn*` matches a QPN8 line, so on its own it does NOT prove the NVFP4
# trunk/lm_head went through QPN2. NVFP4 dispatch is logged as "-> qpn2" in the
# route map, a different spelling entirely. Prove each side with its own witness.
# An image-enabled boot that silently loaded no vision tower would accept
# requests and then fail on the first image, so prove the encoder initialised.
case "$MM_LIMIT" in
  *'"image":0'*) ;;
  *)
    grep -q "MMEncoderAttention" "$LOG" \
      && gate "vision encoder initialised (images at ${MM_PIXELS}px cap)" ok \
      || gate "vision encoder initialised" fail "no MMEncoderAttention line" ;;
esac

grep -qE "route map: M=[0-9]+ N=[0-9]+ K=[0-9]+ -> qpn2" "$LOG" \
  && gate "QPN2 dispatched (NVFP4 trunk/lm_head)" ok \
  || gate "QPN2 dispatched (NVFP4 trunk/lm_head)" fail "no '-> qpn2' route-map line"

# Route census: every protected FP8 module must be QPN8-eligible at load, and
# QPN8 must actually be dispatched at run time. `eligible=NO` is the failure
# this catches -- a module silently falling back to the reference path.
CENSUS=$(grep -c "QPN8_CENSUS_LOAD" "$LOG" || true)
INELIGIBLE=$(grep -c "QPN8_CENSUS_LOAD.*eligible=NO" "$LOG" || true)
# The launch claim is an EXACT number: 2 protected modules per layer x 64
# layers x 4 ranks = 512. "Any positive count" would pass a boot that silently
# dropped modules, which is the failure this gate exists to catch.
CENSUS_EXPECTED=$((128 * TP))
[ "${CENSUS:-0}" = "$CENSUS_EXPECTED" ] && [ "${INELIGIBLE:-0}" = 0 ] \
  && gate "QPN8 census exactly $CENSUS_EXPECTED, 0 ineligible" ok \
  || gate "QPN8 census exactly $CENSUS_EXPECTED" fail "census=$CENSUS ineligible=$INELIGIBLE"

grep -q "route=qpn8" "$LOG" \
  && gate "QPN8 dispatched at run time" ok \
  || gate "QPN8 dispatched at run time" fail "no qpn8 route observed"

if [ "$FAIL" = 0 ]; then
  echo "==> READY on port $PORT (pid $(cat "$PIDFILE")) — all gates passed"
else
  echo "==> BOOT GATE FAILED — measurements from this server are NOT quotable." >&2
  echo "    stop it with:  kill -TERM \$(cat $PIDFILE)" >&2
  cleanup_on_fail
  exit 1
fi
