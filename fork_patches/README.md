# Fork patches (1Cat-vLLM 1.2.2)

**These files are derivative works of [vLLM](https://github.com/vllm-project/vllm),
Copyright contributors to the vLLM project, licensed under Apache-2.0** — not
under this repository's MIT default. Each states its modifications in its
header as Apache-2.0 §4(b) requires. See [`../LICENSE-APACHE-2.0`](../LICENSE-APACHE-2.0)
and [`../NOTICE`](../NOTICE).

They are complete upstream sources carrying local edits, tracked here so the
changes are diffable and reversible outside the installed package. The engine
is installed as a wheel (no source checkout), so the patched files are copied
over the corresponding paths under the environment's `site-packages/`.

**Install them with [`../scripts/bootstrap-sm70.sh`](../scripts/bootstrap-sm70.sh)**,
which resolves the target paths, keeps a `.pre_bootstrap` backup of every file
it replaces, and verifies the kernel extension afterwards. Never hand-edit an
installed file: the tracked copy here is the reviewable source of truth.

| File | Install path (under `site-packages/`) | What we changed |
|---|---|---|
| `marlin.py` | `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | The skinny NVFP4 dispatch: QPN2 geometry winners own decode M 1–8 including `lm_head`, per-shape (split, nacc) table, shape-aware route map; legacy QPN for M 9–16; Marlin above that. |
| `modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | The QPN8 FP8 W8A16 path (`mma.sync.m8n8k4`, incl. the MT=2 two-tile variant), lowers the ModelOpt minimum compute capability from SM89 to SM70, adds route/census logging. **This is what lets a published mixed FP8+NVFP4 checkpoint load on Volta at all.** |
| `flash_attn_v100.py` | `vllm/v1/attention/backends/flash_attn_v100.py` | Admits FP8 **E4M3** KV to the fast attention paths. Upstream gates XQA decode and the FP8 prefill bridge on fp16-or-E5M2 only, so an e4m3 cache -- which is what this checkpoint declares -- silently took `scalar_paged` decode and the slow prefill. Three gates widened: `xqa_kv_supported`, `_smallq_decode_xqa_allowed` (the speculative-verify path above seq 4096, where the long-context win actually lives), and `_should_use_fp8_prefill_bridge` plus its call site. **Requires the rebuilt kernel from `kernel_patches/`** -- without it the extension raises `XQA decode supports fp16 and fp8_e5m2 KV cache only`. Measured at 54k prompt tokens: TTFT 148 s -> 60 s, decode 22.2 -> 50.4 tok/s, output byte-identical. |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | KV-dtype policy: a checkpoint's `kv_cache_quant_algo` describes how its *weights* were made and is no longer honoured as a KV-cache directive below SM80. Without this the verbatim checkpoint silently booted an FP8 KV cache and lost the tensor-core decode route (+4.82 ms/round). |
| `attention.py` | `vllm/model_executor/layers/attention/attention.py` | The same policy on the compressed-tensors re-apply path. |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | Persistent-metadata speculative round, a per-phase GPU profiler, and NVTX phase brackets for per-kernel attribution. Also the prefill guard on the uniform-decode classification (2026-09-04): a prompt of exactly k+1 tokens used to replay the speculative-verify FULL graph with stale metadata and emit garbage. |
| `gdn_attn.py` | `vllm/v1/attention/backends/gdn_attn.py` | Chain-MTP GDN fast metadata build (−1.4 ms/step, byte-identical output). |
| `custom_all_reduce.py` | `vllm/distributed/device_communicators/custom_all_reduce.py` | All-reduce residency instrumentation, default off. Measurement tool, dormant in production. |
| `qwen3_5_mtp.py` | `vllm/model_executor/models/qwen3_5_mtp.py` | Pipeline-parallel support for the MTP drafter (`SupportsPP`, always-embed predictor, drafter-owned embedding when PP>1). Only exercised by the experimental PP=2 prefill mode in `results/prefill_20260904.md`; inert at PP=1. |
| `cuda_communicator.py` | `vllm/distributed/device_communicators/cuda_communicator.py` | Routes small (<=256 KB) fp16 TP=2 all-reduces through `kernels/skinny_ar.cu`, a single-kernel all-reduce over pinned host memory for hosts without GPU P2P. 27 us vs NCCL's 44 us per decode all-reduce, bit-identical. `VLLM_SKINNY_AR=0` disables. |
| `qwen_gdn_linear_attn.py` | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | Routes the GDN `in_proj_ba` projection through the skinny small-N fp16 GEMV for M<=16 (one kernel, ~3 us, instead of cuBLAS wmma + split-K at 13.6 us per GDN layer). |
| `qwen3_5.py` | `vllm/model_executor/models/qwen3_5.py` | The Qwen3.5 GDN subclass's split-projection `forward_cuda` calls `in_proj_ba` directly; routed through the same small-N GEMV helper. |
| `allreduce_rms_fusion.py` | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` | The fork's TP2 all-reduce + Gemma RMSNorm inductor fusion now also arms when `skinny_ar` is present (it required the P2P custom all-reduce). `cuda_communicator.py` serves the fused op with `skinny_ar.all_reduce_gemma_norm`. |

Not installed:

| File | Why |
|---|---|
| `sm70_native_round.py` | Original work (not derived from vLLM), offered under Apache-2.0 so it can combine with the engine. Experimental: built and validated byte-identical, but inert — the captured graph does not persist the drafter's recurrent state across rounds, so served drafts are rejected. |
| `llm_base_proposer.native_round.patch` | The proposer hook that would select the above. Reverted; kept as a diff for development in a proper source checkout. |
