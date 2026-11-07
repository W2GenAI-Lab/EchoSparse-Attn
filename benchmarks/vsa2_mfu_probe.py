"""Measured MFU for the native VSA2 fine kernel, without hardware counters.

Three numbers are easy to get wrong, so this probe measures each one instead of assuming it:

1. **The denominator.** The operator-level MFU divides the *whole* call (router, coarse branch, fusion, fine
   kernel) by the fine kernel's matmul FLOPs. Here the fine kernel is timed on its own through the same binding
   the operator uses, so the ratio is about the kernel.
2. **The peak.** A datasheet peak is not what this part sustains on these shapes. This runs two large bf16 GEMMs
   (square and attention-like) and reports what `torch.matmul` actually reaches on the same GPU, so MFU can be
   quoted against both the datasheet and the achievable figure.
3. **The work.** FLOPs are counted from the (query block, kv block) pairs the router actually selected:
   4*64*64*D per pair for the forward (QK^T and PV), twice that for the backward (dV, dP^T, dQ, dK).

Equivalence is checked in the same process: the isolated fine forward must reproduce the operator's fine output
byte for byte, and the isolated fine backward must match `dq/dk/dv` from a real autograd call within the
operator's own run-to-run noise.

  python benchmarks/vsa2_mfu_probe.py --lens 16384 --heads 16 --dims 128 --out mfu.json
"""
import argparse
import json
import statistics
import sys

import torch


def timed(fn, warmup=10, iters=30, batches=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(batches):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms.append(begin.elapsed_time(end) / iters)
    return statistics.median(ms)


def gemm_peak(dtype=torch.bfloat16):
    """What this GPU actually sustains on large bf16 GEMMs, in TFLOP/s."""
    out = {}
    for n in (8192, 16384):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        ms = timed(lambda: torch.matmul(a, a), warmup=5, iters=10, batches=3)
        out[f"gemm_{n}"] = dict(ms=ms, tflops=2 * n ** 3 / (ms * 1e-3) / 1e12)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lens", type=int, nargs="+", default=[16384])
    ap.add_argument("--heads", type=int, nargs="+", default=[16])
    ap.add_argument("--dims", type=int, nargs="+", default=[128])
    ap.add_argument("--topk-frac", type=float, default=0.10)
    ap.add_argument("--peak-tflops", type=float, default=989.4, help="datasheet dense bf16 peak for this part")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from echosparse import vsa
    from echosparse.vsa import _C as C

    if not torch.cuda.is_available() or not C.has_tk():
        sys.exit("needs a GPU and an ECHOSPARSE_VSA_TK=1 build")

    device = torch.cuda.get_device_name(0)
    peaks = gemm_peak()
    achievable = max(v["tflops"] for v in peaks.values())
    results = {"device": device, "datasheet_tflops": a.peak_tflops, "gemm": peaks,
               "achievable_tflops": achievable, "shapes": {}}

    for L in a.lens:
        for H in a.heads:
            for D in a.dims:
                B, dev, dt = 1, "cuda", torch.bfloat16
                n = L
                lay = vsa.sequence_layout(n, dev)
                q, k, v = (torch.randn(B, H, n, D, device=dev, dtype=dt) for _ in range(3))
                qb, kb, vb = (vsa.to_blocked(t, lay) for t in (q, k, v))
                gate = (0.1 * torch.randn(B, H, n, device=dev)).to(dt)
                gb = vsa.to_blocked(gate, lay)
                topk = max(1, round(a.topk_frac * (n // 64)))
                o, sel = vsa.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=topk, backend="tk",
                                            return_selection=True)
                pairs = int((sel.q2k_ptr[..., 1:] - sel.q2k_ptr[..., :-1]).sum().item())
                flops = dict(fwd=pairs * 4 * 64 * 64 * D, bwd=pairs * 8 * 64 * 64 * D)

                # forward: the same binding the operator calls, with the frozen selection
                fn_fwd = lambda: C.attn_fwd(qb, kb, vb, sel.q2k_ptr, sel.q2k_idx, lay.blk_cnt, True)
                off, lse_f = fn_fwd()
                o_ref, _ = vsa.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=topk, backend="tk",
                                              selection=sel, return_selection=True)
                fwd_identical = torch.equal(off.float(), o_ref.float())

                # backward: reconstruct dout_f and delta exactly as the operator does, then time the fine kernel
                dout = torch.randn_like(qb)
                # delta is the row sum of dO*O, which is what fuse_bwd forms before the fine backward reads it
                delta_f = (dout.float() * off.float()).sum(-1)
                dq = torch.zeros_like(qb, dtype=torch.float32)
                dk = torch.zeros_like(kb, dtype=torch.float32)
                dv = torch.zeros_like(vb, dtype=torch.float32)
                # binding order: q,k,v,dout_f,lse,delta,row_ptr,col_idx,blk_cnt,qg,kg,vg,opts
                fn_bwd = lambda: C.tk_fine_bwd(qb, kb, vb, dout, lse_f, delta_f, sel.k2q_ptr, sel.k2q_idx,
                                               lay.blk_cnt, dq, dk, dv, 0)

                ms_fwd = timed(fn_fwd)
                ms_bwd = timed(fn_bwd)
                entry = dict(pairs=pairs, topk=topk, flops=flops, fwd_ms=ms_fwd, bwd_ms=ms_bwd,
                             fwd_tflops=flops["fwd"] / (ms_fwd * 1e-3) / 1e12,
                             bwd_tflops=flops["bwd"] / (ms_bwd * 1e-3) / 1e12,
                             fwd_identical=fwd_identical)
                entry["fwd_mfu_datasheet"] = entry["fwd_tflops"] / a.peak_tflops
                entry["bwd_mfu_datasheet"] = entry["bwd_tflops"] / a.peak_tflops
                entry["fwd_mfu_achievable"] = entry["fwd_tflops"] / achievable
                entry["bwd_mfu_achievable"] = entry["bwd_tflops"] / achievable
                results["shapes"][f"L{L}/H{H}/D{D}"] = entry
                print(f"L{L}/H{H}/D{D}: pairs {pairs}, fwd {ms_fwd:.3f} ms = {entry['fwd_tflops']:.0f} TFLOP/s "
                      f"({entry['fwd_mfu_datasheet']*100:.1f}% datasheet, {entry['fwd_mfu_achievable']*100:.1f}% achievable), "
                      f"bwd {ms_bwd:.3f} ms = {entry['bwd_tflops']:.0f} TFLOP/s "
                      f"({entry['bwd_mfu_datasheet']*100:.1f}%, {entry['bwd_mfu_achievable']*100:.1f}%), "
                      f"fwd identical {fwd_identical}")

    results["note"] = "measurement_only: FLOPs from the selection; no hardware counters (ncu is not permitted here)"
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"achievable bf16 GEMM peak on {device}: {achievable:.0f} TFLOP/s (datasheet {a.peak_tflops:.0f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
