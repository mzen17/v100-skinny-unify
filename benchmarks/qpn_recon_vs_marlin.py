"""qpn_recon (transient dequant+matmul off the qpn buffer) vs the fork's
Marlin fallback, on the M>16 band -- the band VLLM_SKINNY_QPN_UNIFY=1
reroutes away from Marlin entirely. kernel_matched_bench.py explicitly
stops at M=16 ("M>=17 is Marlin by design"); this fills that gap for the
new path. Uses the SHIPPED shim (_qpn_prepack, _qpn_reconstruct_weight)
so the thing measured is the thing that runs.
"""
import os
import sys
import time

import torch

dev = "cuda"
GSCALE = 1.0

from vllm.model_executor.kernels.linear.nvfp4 import marlin as shim  # noqa: E402


def bench(fn, it=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / it


def packed_bytes(k, n):
    return n * (k // 2) + n * (k // 16)


def marlin_arm():
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin,
        is_fp4_marlin_supported)
    if not is_fp4_marlin_supported():
        return None

    class _L(torch.nn.Module):
        pass

    def build(k, n, codes, sbytes):
        lay = _L()
        lay.params_dtype = torch.float16
        lay.weight = torch.nn.Parameter(codes.clone(), requires_grad=False)
        lay.weight_scale = torch.nn.Parameter(
            sbytes.clone().view(torch.float8_e4m3fn), requires_grad=False)
        lay.weight_global_scale = torch.nn.Parameter(
            torch.tensor([GSCALE], device=dev), requires_grad=False)
        lay.input_size_per_partition = k
        lay.output_size_per_partition = n
        prepare_fp4_layer_for_marlin(lay)
        return lay

    def call(lay, x, k, n):
        return apply_fp4_marlin_linear(
            input=x, weight=lay.weight, weight_scale=lay.weight_scale,
            weight_global_scale=lay.weight_global_scale,
            workspace=lay.workspace, size_n=n, size_k=k, bias=None)

    return build, call


# Real per-rank trunk shapes at TP=2 on this box (gate/up_proj, down_proj)
# plus lm_head, same set kernel_matched_bench.py uses.
SHAPES = [(5120, 17408), (8704, 5120), (5120, 62080)]
M_VALUES = [17, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def main():
    ext = shim._get_skinny_ext()
    if ext is None:
        sys.exit("skinny extension failed to build")
    marlin = marlin_arm()
    g = torch.Generator(device="cpu").manual_seed(0)

    def build_shape(k, n):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8,
                              generator=g).to(dev)
        sbytes = torch.randint(0x30, 0x50, (n, k // 16), dtype=torch.uint8,
                               generator=g).to(dev)
        qc, qs = shim._qpn_prepack(codes, sbytes)
        lay = marlin[0](k, n, codes, sbytes) if marlin else None
        return k, n, qc, qs, lay, packed_bytes(k, n)

    shapes = [build_shape(k, n) for k, n in SHAPES]

    print(f"# shapes (k,n) = {SHAPES}")
    print("shape,M,qpn_recon_ms,qpn_recon_GBs,marlin_ms,marlin_GBs,"
          "recon_vs_marlin_speed_ratio")
    for k, n, qc, qs, lay, gb in shapes:
        for M in M_VALUES:
            x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5

            def f_recon():
                w = shim._qpn_reconstruct_weight(qc, qs, n, k, GSCALE,
                                                 torch.float16)
                return torch.nn.functional.linear(x, w)

            t_recon = bench(f_recon)
            gbs_recon = gb / (t_recon * 1e-3) / 1e9

            if lay is not None:
                t_marlin = bench(lambda: marlin[1](lay, x, k, n))
                gbs_marlin = gb / (t_marlin * 1e-3) / 1e9
                ratio = t_marlin / t_recon  # >1 means qpn_recon is faster
                print(f"({k},{n}),{M},{t_recon:.3f},{gbs_recon:.1f},"
                      f"{t_marlin:.3f},{gbs_marlin:.1f},{ratio:.2f}",
                      flush=True)
            else:
                print(f"({k},{n}),{M},{t_recon:.3f},{gbs_recon:.1f},"
                      f"NA,NA,NA", flush=True)
    print("QPN_RECON_BENCH_DONE")


if __name__ == "__main__":
    main()
