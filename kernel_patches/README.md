# Kernel patches for `flash_attn_v100`

Upstream: https://github.com/1CatAI/1Cat-vLLM
Pinned at: `7aede2cf010d92815c9d7bff25867b4fa009b6cb` (2026-08-19)

## 0001-flash-attn-v100-e4m3-kv.patch

Adds FP8 **E4M3** support to the XQA decode path and the FP8 prefill bridge.
Upstream supports only fp16 and E5M2 there, because E5M2 -> fp16 is a pure bit
shift (same exponent width and bias) while E4M3 needs an exponent rebias and
upstream's only E4M3 decoder is branchy, scalar and returns `float`.

Why it matters here: this checkpoint declares E4M3 KV. Without the patch, e4m3
falls off every fast attention path, and the only way to reach them is E5M2 --
which this fork forces to *unit* KV scales, discarding the checkpoint's
calibrated ones (measured: mean KL 0.23 nats, +25% AIME trace length).

Measured on 2x V100 (TP=2, k=3, 64k window), 54,054-token prompt:

| | before | after |
|---|--:|--:|
| TTFT | 148.1 s | 60.0 s |
| prefill | 365 tok/s | 902 tok/s |
| decode | 22.2 tok/s | 50.4 tok/s |

Quality unchanged: decode KLD vs the unpatched e4m3 path over 384 positions
gives 100% top-1 agreement and byte-identical greedy output; mean KL 8.2e-06.
Both decoders were verified against reference implementations, and a negative
control (same bytes labelled E5M2) shows 2,512x the error -- the dispatch
really does distinguish the two FP8 flavours.

### Apply and build

    git clone https://github.com/1CatAI/1Cat-vLLM.git
    cd 1Cat-vLLM && git checkout 7aede2cf010d92815c9d7bff25867b4fa009b6cb
    git apply /path/to/0001-flash-attn-v100-e4m3-kv.patch
    cd flash-attention-v100
    CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=6 \
      python setup.py build_ext --inplace

Then copy ONLY the rebuilt `flash_attn_v100_cuda*.so` over the installed
package, and port the `is_e4m3` argument onto the *installed*
`flash_attn_interface.py`. Do NOT wholesale-replace that wrapper with
upstream's newer one: it adds `_assert_decode_launch_covers_seq_lens`, which
calls `seq_lens.max().item()` -- a device sync on every decode step, measured
at roughly +10 ms/round.

### Also required (vLLM side, not covered by this patch)

Three gates in `vllm/v1/attention/backends/flash_attn_v100.py`:
  - `xqa_kv_supported` .......... general XQA entry
  - `_smallq_decode_xqa_allowed`  speculative-verify path, seq >= 4096 --
    THIS is where the long-context win lives; widening only the first gate
    produces no speedup at all
  - `_should_use_fp8_prefill_bridge` + its call site (pass `is_e4m3=`)
