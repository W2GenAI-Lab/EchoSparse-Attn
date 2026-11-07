"""Native VSA2 through the public API: timing and equivalence of one env-toggle arm against the default.

Measurement only. This script reports medians, spreads, ratios and errors, and makes no performance claim.

One arm per run, one JSON per (arm, shape). For each shape the driver starts four processes in ABBA order: default, arm,
arm, default. Every ECHOSPARSE_VSA_* variable is removed from their environment and only the arm's is set. `--arm baseline`
makes all four the default, which is the A/A run. Ratios only ever come from inside one run, because separate runs drift.

Rows, all vsa.vsa2_attention on blocked bf16 tensors with a scalar gate per token and backend="tk" (CUDA events, warmup 10,
30 iterations, 3 batches):
  vsa2.fwd       the forward in a loop, router included, on the same grad-tracking leaves as f_plus_b
  vsa2.bwd       one backward: fresh graph per sample, forward finished by a synchronize before the interval, gradients
                 reset after it. The synchronizes add idle gaps that f_plus_b does not have
  vsa2.f_plus_b  a complete autograd call: forward, backward and gradient reset, no synchronize inside the loop
  b_estimate_ms  median(f_plus_b) - median(fwd): derived, not a timed interval

Equivalence is reported beside the timings and never folded into them. The driver compares the processes' tensors on the CPU:
  forward   o and the selection must be bit-identical: arm against default, and default against default
  backward  dq, dk, dv and dgate are cross-CTA fp32 reduce-adds, not run-to-run deterministic, so byte equality is the wrong
            bar. Each is reported as an error against the default, beside the same error between two default runs
  padding   slots past blk_cnt must be exactly zero. Only the partial shape has any; a full shape reports none to check

Verdict for one shape, worst first:
  FAIL      NaN, a nonzero padded slot, or o or the selection differing from the default
  NONDETERMINISTIC_FORWARD  the default's o or selection differs from itself, so the forward check cannot separate arms
  OVER_TOL  a gradient's rel_l2 against the default exceeds --grad-tol
  ABOVE_AA  within tolerance, but rel_l2, max_abs or n_diff exceed AA_MARGIN times the largest A/A value
  OK        the outputs match, and gradient differences are within what identical runs show

The default is the reference: this says an arm matches the default, not that the default is right, which is what
tests/test_vsa2.py is for. A toggle no code reads is refused, but one that is read and changes nothing still reads OK and neutral.

  <gpu-runner> python benchmarks/vsa2_kernel_bench.py --arm baseline --out-dir results/aa
  <gpu-runner> python benchmarks/vsa2_kernel_bench.py --arm tk_bwd_store --out-dir results/store
  python benchmarks/vsa2_kernel_bench.py --compare results/aa results/store
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "vsa2-kernel-bench/1"
NOTE = "measurement_only: medians, spreads, ratios and errors; not a performance claim"
ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "ECHOSPARSE_VSA_"
ARMS = {"baseline": {}, "tk_bwd_store": {"ECHOSPARSE_VSA_TK_BWD_STORE": "1"},
        "coarse_meta_cache": {"ECHOSPARSE_VSA_COARSE_META_CACHE": "1"}, "fused_cols": {"ECHOSPARSE_VSA_FUSED_COLS": "1"}}
ROWS = ("vsa2.fwd", "vsa2.bwd", "vsa2.f_plus_b")
GRADIENTS = ("dq", "dk", "dv", "dgate")
ORDER = ("default", "arm", "arm", "default")
LENS, HEADS, DIMS = (4096, 8192, 16384, 32768), (8, 16), (64, 128)
PARTIAL_SHAPE = dict(H=16, L=16384, D=128)
SETTINGS = ("warmup", "iters", "batches", "seed", "topk_frac", "grad_tol")
AA_MARGIN = 4.0  # identical runs can differ by a factor of about 4 in n_diff, and each floor has few pairs
RANK = ("OK", "ABOVE_AA", "OVER_TOL", "NONDETERMINISTIC_FORWARD", "FAIL")


def shape_matrix(lens=LENS, heads=HEADS, dims=DIMS, partial=True):
    shapes = [dict(B=1, H=h, L=l, D=d, partial=False) for l in lens for h in heads for d in dims]
    return shapes + ([dict(B=1, partial=True, **PARTIAL_SHAPE)] if partial else [])


def shape_key(s):
    return f"L{s['L']}_H{s['H']}_D{s['D']}" + ("_partial" if s["partial"] else "")


def shape_error(s):
    """Why the preblocked layout cannot hold this shape, or None."""
    if s["L"] <= 0 or s["L"] % 64:
        return f"L must be a positive multiple of 64, got {s['L']}"
    if s["D"] not in (64, 128):
        return f"D must be 64 or 128, got {s['D']}"
    return None


def partial_counts(nb):
    """Valid tokens per block, through the sequence and not only at its tail. blk_cnt is 1..64 by the layout contract."""
    def count(i):
        if i == nb - 1:
            return 9
        return 1 if i % 7 == 3 else 33 if i % 4 == 1 else 17 if i % 16 == 5 else 64
    return [count(i) for i in range(nb)]


def result_name(arm, shape):
    return f"vsa2_kernel__{arm}__{shape_key(shape)}.json"


def child_env(arm, environ=os.environ):
    env = {k: v for k, v in environ.items() if not k.startswith(ENV_PREFIX)}
    env.update(ARMS[arm])
    return env


def missing_toggles(arm, python_source, extension_bytes):
    """Toggle names neither the loaded Python operator nor the built extension contains; such an arm measures the default."""
    return [n for n in ARMS[arm] if n not in python_source and n.encode() not in extension_bytes]


def source_files():
    return sorted(p for pat in ("echosparse/vsa/*.py", "csrc/vsa/**/*") for p in ROOT.glob(pat) if p.is_file())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest():
    table = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files()}
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()


def spread(values):
    return max(values) - min(values)


def time_rows(children):
    """Per-row timing for one shape; children are in ABBA order. Ratios come from inside this run only."""
    default = [c for c, who in zip(children, ORDER) if who == "default"]
    arm = [c for c, who in zip(children, ORDER) if who == "arm"]
    rows = {}
    for name in ROWS:
        base = [x for c in default for x in c["rows"][name]]
        cand = [x for c in arm for x in c["rows"][name]]
        first, last = (statistics.median(c["rows"][name]) for c in default)
        rows[name] = dict(default_median_ms=statistics.median(base), default_spread_ms=spread(base),
                          arm_median_ms=statistics.median(cand), arm_spread_ms=spread(cand),
                          ratio=statistics.median(cand) / statistics.median(base), aa_displacement_same_run=last / first,
                          samples=dict(default=base, arm=cand), measured=True)
    fb, f = rows["vsa2.f_plus_b"], rows["vsa2.fwd"]
    rows["b_estimate_ms"] = dict(measured=False, method="median(f_plus_b) - median(fwd); not a timed interval",
                                 default=fb["default_median_ms"] - f["default_median_ms"],
                                 arm=fb["arm_median_ms"] - f["arm_median_ms"])
    return rows


def worst(samples, key):
    """The larger of a metric over error samples; NaN wins, so it can never read as small."""
    values = [s[key] for s in samples]
    return next((v for v in values if v != v), max(values))


def gradient_verdict(vs_default, floors, grad_tol, margin=AA_MARGIN):
    """OK, ABOVE_AA, OVER_TOL or FAIL for one gradient, from errors against the default and A/A pairs."""
    rel, err, n = (worst(vs_default, k) for k in ("rel_l2", "max_abs", "n_diff"))
    if worst(vs_default, "n_nan") > 0 or rel != rel or err != err:
        return "FAIL"
    if rel > grad_tol:
        return "OVER_TOL"
    if any(x > margin * worst(floors, k) for x, k in ((rel, "rel_l2"), (err, "max_abs"), (n, "n_diff"))):
        return "ABOVE_AA"
    return "OK"


def shape_verdict(forward, backward, padded_nonzero):
    """The worst of the forward, padding and gradient results for one shape; see the module docstring."""
    found = ["OK"] + [g["verdict"] for g in backward.values()]
    if not forward["aa_o_identical"] or not forward["aa_selection_identical"]:
        found.append("NONDETERMINISTIC_FORWARD")
    if not forward["o_identical"] or not forward["selection_identical"] or any(padded_nonzero.values()):
        found.append("FAIL")
    return max(found, key=RANK.index)


def equivalence(children, errors, grad_tol):
    """errors(tensor name, i, j) is the error between children i and j, in ORDER positions."""
    o = [c["digests"]["o"] for c in children]
    sel = [c["digests"]["selection"] for c in children]
    forward = dict(o_sha256=o, o_identical=o[1] == o[0] and o[2] == o[3], aa_o_identical=o[0] == o[3],
                   selection_identical=sel[1] == sel[0] and sel[2] == sel[3], aa_selection_identical=sel[0] == sel[3])
    backward = {}
    for name in GRADIENTS:
        vs_default = [errors(name, 1, 0), errors(name, 2, 3)]
        aa_default, aa_arm = errors(name, 3, 0), errors(name, 2, 1)
        backward[name] = dict(vs_default=vs_default, aa_default=aa_default, aa_arm=aa_arm,
                              verdict=gradient_verdict(vs_default, [aa_default, aa_arm], grad_tol))
    padded = {k: max(c["padded_nonzero"][k] for c in children) for k in children[0]["padded_nonzero"]}
    return dict(forward=forward, backward=backward, padded_slots=children[0]["padded_slots"], padded_nonzero=padded,
                grad_tol=grad_tol, aa_margin=AA_MARGIN, verdict=shape_verdict(forward, backward, padded))


def tensor_errors(a, b):
    """Error of a against b. n_diff counts numeric differences, n_bit_diff also counts a differing sign of zero."""
    import torch

    a, b = a.float(), b.float()
    diff = a - b
    norm = b.double().norm().item()
    return dict(rel_l2=diff.double().norm().item() / (norm or 1.0), max_abs=diff.abs().max().item(), scale=b.abs().max().item(),
                n_diff=int((a != b).sum().item()), n_bit_diff=int(((a != b) | (torch.signbit(a) != torch.signbit(b))).sum().item()),
                n_nan=int(torch.isnan(a).sum().item()), n_elements=a.numel())


def compare_tensors(children, grad_tol):
    """Compare the processes' saved tensors on the CPU. The driver never touches the GPU."""
    import torch

    def errors(name, i, j):
        x = torch.load(children[i]["tensors"], map_location="cpu", mmap=True)[name]
        y = torch.load(children[j]["tensors"], map_location="cpu", mmap=True)[name]
        return tensor_errors(x, y)
    return equivalence(children, errors, grad_tol)


def run_shape(a, shape, scratch):
    """Four processes for one shape, then their comparison. A shape that cannot run says why."""
    work = tempfile.mkdtemp(prefix="vsa2_kernel_", dir=scratch)
    try:
        children = []
        for i, who in enumerate(ORDER):
            spec = dict(shape=shape, seed=a.seed, topk_frac=a.topk_frac, warmup=a.warmup, iters=a.iters, batches=a.batches,
                        arm=a.arm if who == "arm" else "baseline", out=os.path.join(work, f"{i}.json"),
                        tensors=os.path.join(work, f"{i}.pt"))
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)],
                               env=child_env(spec["arm"]), capture_output=True, text=True)
            if not Path(spec["out"]).exists():
                tail = (r.stderr.strip().splitlines() or ["no output"])[-1]
                return dict(status="failed", reason=f"process {i} ({who}) exited {r.returncode}: {tail}")
            child = json.loads(Path(spec["out"]).read_text())
            if child.get("status") != "ok":
                return dict(status=child.get("status"), reason=f"process {i} ({who}): {child.get('reason')}")
            children.append(child)
        if len({json.dumps(c["identity"], sort_keys=True) for c in children}) > 1:
            return dict(status="invalid", reason="processes differ in build, hardware, inputs or shape")
        try:
            eq = compare_tensors(children, a.grad_tol)
        except Exception as e:  # timings without their comparison must not read as a clean result
            return dict(status="failed", reason=f"comparing tensors: {type(e).__name__}: {e}")
        return dict(status="ok", topk=children[0]["topk"], blk_cnt=children[0]["blk_cnt"], run_order=list(ORDER),
                    timing=time_rows(children), equivalence=eq, identity=children[0]["identity"],
                    stale_extension_sources=sorted({n for c in children for n in c["stale_extension_sources"]}))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def child_main(spec):
    out = Path(spec["out"])
    try:
        out.write_text(json.dumps(measure(spec, spec["shape"])))
    except Exception as e:  # a shape that does not run is a result, not a skipped row
        out.write_text(json.dumps(dict(status="failed", reason=f"{type(e).__name__}: {e}")))
    return 0


def measure(spec, shape):
    error = shape_error(shape)
    if error:
        return dict(status="unsupported", reason=error)
    sys.path.insert(0, str(ROOT))
    import torch

    from echosparse import vsa
    from echosparse.vsa import ops

    if not torch.cuda.is_available():
        return dict(status="unsupported", reason="no GPU")
    if torch.cuda.device_count() != 1:
        return dict(status="unsupported", reason=f"needs exactly one visible GPU, got {torch.cuda.device_count()}")
    C = ops.ext()
    missing = missing_toggles(spec["arm"], Path(ops.__file__).read_text(), Path(C.__file__).read_bytes())
    if missing:
        return dict(status="toggle_missing", reason=f"neither echosparse.vsa.ops nor the built extension contains {missing}")
    W, I, N = spec["warmup"], spec["iters"], spec["batches"]
    dev, dt, seed = "cuda", torch.bfloat16, spec["seed"]
    B, H, L, D, nb = shape["B"], shape["H"], shape["L"], shape["D"], shape["L"] // 64
    topk = max(1, round(spec["topk_frac"] * nb))

    def digest(t):
        data = t.float() if t.dtype == torch.bfloat16 else t
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                              + data.detach().cpu().numpy().tobytes()).hexdigest()

    def loop(fn):
        for _ in range(W):
            fn()
        ms = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            for _ in range(I):
                fn()
            t1.record()
            torch.cuda.synchronize()
            ms.append(t0.elapsed_time(t1) / I)
        return ms

    if shape["partial"]:  # sequence_layout only pads the last block, so partial blocks are laid out through the sequence
        counts = partial_counts(nb)
        cnt = torch.tensor(counts, dtype=torch.int32, device=dev)
        valid = (torch.arange(64, device=dev)[None, :] < cnt.to(torch.int64)[:, None]).reshape(L)
        n = sum(counts)
        scatter = valid.nonzero()[:, 0]
        gather = torch.full((L,), -1, dtype=torch.int64, device=dev)
        gather[scatter] = torch.arange(n, device=dev)
        lay = vsa.BlockLayout((n,), (64,), (), gather, scatter, cnt)
    else:
        n, counts = L, [64] * nb
        lay = vsa.sequence_layout(n, dev)
        valid = lay.gather >= 0
    torch.manual_seed(seed)
    q, k, v = (vsa.to_blocked(torch.randn(B, H, n, D, device=dev, dtype=dt), lay).contiguous() for _ in range(3))
    dout = torch.randn_like(q)
    gate = vsa.to_blocked((0.1 * torch.randn(B, H, n, device=dev)).to(dt), lay)
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate)]
    qq, kk, vv, gg = leaves

    def forward(**kw):
        return vsa.vsa2_attention(qq, kk, vv, gg, lay.blk_cnt, R=16, topk=topk, backend="tk", **kw)

    def reset():
        for t in leaves:
            t.grad = None

    def f_plus_b():
        forward().backward(dout)
        reset()

    def direct():
        samples = []
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for i in range(W + I * N):
            o = forward()
            torch.cuda.synchronize()
            begin.record()
            o.backward(dout)
            end.record()
            torch.cuda.synchronize()
            if i >= W:
                samples.append(begin.elapsed_time(end))
            del o
            reset()
        return [statistics.median(samples[i:i + I]) for i in range(0, len(samples), I)]

    rows = {"vsa2.fwd": loop(forward), "vsa2.bwd": direct(), "vsa2.f_plus_b": loop(f_plus_b)}

    o, sel = forward(return_selection=True)
    # Algorithmic matmul FLOPs from the selection actually made, not from the budget: per selected
    # (query block, kv block) pair and head the forward issues QK^T and PV (4 * 64 * 64 * D FLOPs) and the
    # backward issues dV, dP^T, dQ and dK (twice that). MFU is those FLOPs over time divided by the dense bf16
    # peak; it needs no hardware counter, which matters because ncu is not permitted on this machine.
    pairs = int((sel.q2k_ptr[..., 1:] - sel.q2k_ptr[..., :-1]).sum().item())
    flops = dict(fwd=pairs * 4 * 64 * 64 * D, bwd=pairs * 8 * 64 * 64 * D)
    # achieved TFLOP/s and MFU per row: algorithmic matmul FLOPs divided by the measured interval
    peak = float(os.environ.get("VSA2_PEAK_TFLOPS", "989.4"))  # H100 SXM 80GB HBM3, dense bf16
    for name, key, fl in (("vsa2.fwd", "fwd", flops["fwd"]), ("vsa2.bwd", "bwd", flops["bwd"]),
                          ("vsa2.f_plus_b", "both", flops["fwd"] + flops["bwd"])):
        ms = statistics.median(rows[name])
        achieved = fl / (ms * 1e-3) / 1e12 if ms > 0 else 0.0
        rows[name].update(flops=fl, achieved_tflops=achieved, mfu=achieved / peak, peak_tflops=peak)
    o.backward(dout)
    outputs = dict(o=o.detach(), dq=qq.grad, dk=kk.grad, dv=vv.grad, dgate=gg.grad)
    torch.save({name: t.detach().cpu() for name, t in outputs.items()}, spec["tensors"])
    padded = {name: int((t.float()[:, :, ~valid] != 0).sum().item()) for name, t in outputs.items()}
    built = os.path.getmtime(C.__file__)
    stale = sorted(p.name for p in source_files() if "csrc" in p.parts and p.stat().st_mtime > built)
    props = torch.cuda.get_device_properties(0)
    identity = dict(
        build=dict(extension_sha256=sha256_file(C.__file__), source_sha256=source_digest(), harness_sha256=sha256_file(__file__)),
        hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      multi_processor_count=getattr(props, "multi_processor_count", None),
                      uuid_sha256=hashlib.sha256(str(getattr(props, "uuid", "")).encode()).hexdigest()),
        software=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
        shape=dict(shape, NB=nb, tokens=n, R=16, topk=topk, dtype="bf16", backend="tk", gate="scalar per token"),
        timing=dict(warmup=W, iters=I, batches=N, clock="torch.cuda.Event"),
        inputs=dict(seed=seed, **{name: digest(t) for name, t in (("q", q), ("k", k), ("v", v), ("dout", dout), ("gate", gate),
                                                                  ("blk_cnt", lay.blk_cnt))}))
    selection = hashlib.sha256("".join(digest(getattr(sel, f)) for f in ("q2k_ptr", "q2k_idx", "k2q_ptr", "k2q_idx")).encode()).hexdigest()
    return dict(status="ok", topk=topk, rows=rows, digests=dict(o=digest(o), selection=selection), padded_slots=int((~valid).sum().item()),
                padded_nonzero=padded, tensors=spec["tensors"], identity=identity, stale_extension_sources=stale,
                blk_cnt=dict(min=min(counts), max=max(counts), partial_blocks=sum(c < 64 for c in counts), blocks=nb))


# ------------------------------------------------------------ reading results
def compare_error(aa, arm):
    """Why these two results cannot be read together, or None."""
    if aa.get("schema") != SCHEMA or arm.get("schema") != SCHEMA:
        return f"both files must be {SCHEMA} results"
    if aa["arm"] != "baseline":
        return f"the first file must be the A/A run (--arm baseline), got --arm {aa['arm']}"
    if aa["shape"] != arm["shape"]:
        return f"shapes differ: {shape_key(aa['shape'])} and {shape_key(arm['shape'])}"
    for key in SETTINGS:
        if aa["settings"][key] != arm["settings"][key]:
            return f"settings differ in {key}"
    if aa["status"] == "ok" and arm["status"] == "ok":
        for part in ("build", "hardware", "software", "shape", "timing", "inputs"):
            if aa["identity"][part] != arm["identity"][part]:
                return f"runs differ in {part}"
    return None


def pair_results(a, b):
    """(A/A result, arm result) pairs from two files or two --out-dir directories, by shape."""
    def load(path):
        path = Path(path)
        files = sorted(path.glob("vsa2_kernel__*.json")) if path.is_dir() else [path]
        return {shape_key(r["shape"]): r for r in (json.loads(f.read_text()) for f in files)}
    x, y = load(a), load(b)
    if set(x) != set(y):
        raise ValueError(f"shapes differ: {sorted(set(x) ^ set(y))}")
    for name in x:
        error = compare_error(x[name], y[name])
        if error:
            raise ValueError(f"{name}: {error}")
    return [(x[name], y[name]) for name in x]


def floors_for(res, aa, gradient):
    """Every A/A pair available for one gradient: this run's two, and the A/A run's when there is one."""
    g = res["equivalence"]["backward"][gradient]
    floors = [g["aa_default"], g["aa_arm"]]
    if aa and aa["status"] == "ok":
        h = aa["equivalence"]["backward"][gradient]
        floors += [h["aa_default"], h["aa_arm"]]
    return floors


def combined_verdict(res, aa=None):
    """The run's verdict, re-read against the largest floor available."""
    eq = res["equivalence"]
    backward = {n: dict(g, verdict=gradient_verdict(g["vs_default"], floors_for(res, aa, n), eq["grad_tol"]))
                for n, g in eq["backward"].items()}
    return shape_verdict(eq["forward"], backward, eq["padded_nonzero"])


def mfu_line(result):
    """One line per row: achieved TFLOP/s and MFU against the dense bf16 peak, from the selection."""
    out = []
    for name in ("vsa2.fwd", "vsa2.bwd", "vsa2.f_plus_b"):
        r = result["rows"][name]
        out.append(f"{name} {r.get('mfu', float('nan')) * 100:5.1f}% MFU ({r.get('achieved_tflops', 0):6.1f} of {r.get('peak_tflops', 0):.0f} TFLOP/s)")
    return "; ".join(out)


def report(pairs):
    """pairs: (A/A result or None, arm result)."""
    print(NOTE)
    print(f"arm {pairs[0][1]['arm']}, env {pairs[0][1]['env']}; ratio = arm / default inside one run, ABBA per shape")
    has_aa = pairs[0][0] is not None
    print("A/A same = last default / first default in this run" + ("; A/A run = ratio of the --arm baseline run" if has_aa else ""))
    print(f"\n{'shape':26s} {'row':14s} {'default ms':>10s} {'spread':>7s} {'arm ms':>9s} {'spread':>7s} {'ratio':>7s} {'A/A same':>9s}"
          + (f" {'A/A run':>8s}" if has_aa else ""))
    for aa, res in pairs:
        name = shape_key(res["shape"])
        if res["status"] != "ok":
            print(f"{name:26s} {res['status'].upper()}: {res['reason']}")
            continue
        for row in ROWS:
            t = res["timing"][row]
            line = (f"{name:26s} {row:14s} {t['default_median_ms']:10.3f} {t['default_spread_ms']:7.3f} {t['arm_median_ms']:9.3f} "
                    f"{t['arm_spread_ms']:7.3f} {t['ratio']:7.3f} {t['aa_displacement_same_run']:9.3f}")
            if has_aa:
                line += f" {aa['timing'][row]['ratio']:8.3f}" if aa["status"] == "ok" else f" {'n/a':>8s}"
            print(line)
    print("\nMFU = algorithmic matmul FLOPs (from the selection) / time / dense bf16 peak; no counters involved")
    for _aa, res in pairs:
        if res.get("status") == "ok":
            print(f"{shape_key(res['shape']):26s} {mfu_line(res)}")
    print(f"\n{'shape':26s} {'output':6s} {'rel_l2':>10s} {'A/A rel_l2':>10s} {'max_abs':>10s} {'A/A max_abs':>11s} {'n_diff':>9s} "
          f"{'A/A n_diff':>10s} {'of':>10s}  verdict")
    for aa, res in pairs:
        if res["status"] != "ok":
            continue
        name, eq = shape_key(res["shape"]), res["equivalence"]
        f = eq["forward"]
        said = lambda ok: "bit-identical" if ok else "DIFFERS"
        pad = f"padded slots {eq['padded_slots']}, nonzero {sum(eq['padded_nonzero'].values())}" if eq["padded_slots"] else "no padded slots to check"
        print(f"{name:26s} o and selection: o {said(f['o_identical'])}, selection {said(f['selection_identical'])}, "
              f"A/A {said(f['aa_o_identical'] and f['aa_selection_identical'])}; {pad}")
        for g, e in eq["backward"].items():
            floors = floors_for(res, aa, g)
            v = e["vs_default"]
            print(f"{name:26s} {g:6s} {worst(v, 'rel_l2'):10.3e} {worst(floors, 'rel_l2'):10.3e} {worst(v, 'max_abs'):10.3e} "
                  f"{worst(floors, 'max_abs'):11.3e} {worst(v, 'n_diff'):9d} {worst(floors, 'n_diff'):10d} {v[0]['n_elements']:10d}  "
                  f"{gradient_verdict(v, floors, eq['grad_tol'])}")
        print(f"{name:26s} verdict {combined_verdict(res, aa)}")
        if res["stale_extension_sources"]:
            print(f"{name:26s} WARNING extension older than {res['stale_extension_sources']}; the arm may not be in it")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out-dir", help="one JSON per (arm, shape) is written here; existing results are not overwritten")
    ap.add_argument("--compare", nargs=2, metavar=("AA", "ARM"), help="the --arm baseline result, then the arm's: two files or two --out-dir directories")
    ap.add_argument("--lens", type=int, nargs="+", default=list(LENS))
    ap.add_argument("--heads", type=int, nargs="+", default=list(HEADS))
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--no-partial", action="store_true", help="drop the partial-block shape, the only one with padded slots")
    ap.add_argument("--topk-frac", type=float, default=0.10, help="topk = max(1, round(frac * NB))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--grad-tol", type=float, default=1e-3, help="largest gradient rel_l2 against the default")
    ap.add_argument("--scratch", help="directory for each shape's tensors; up to about 2.7 GB at the largest shape")
    a = ap.parse_args()
    if a.child:
        return child_main(json.loads(a.child))
    if a.compare:
        try:
            report(pair_results(*a.compare))
        except ValueError as e:
            sys.exit(f"refusing to compare: {e}")
        return 0
    if not a.arm or not a.out_dir:
        ap.error("--arm and --out-dir are required, or use --compare")
    out_dir = Path(a.out_dir)
    shapes = shape_matrix(a.lens, a.heads, a.dims, not a.no_partial)
    taken = [result_name(a.arm, s) for s in shapes if (out_dir / result_name(a.arm, s)).exists()]
    if taken:
        ap.error(f"{out_dir} already holds {taken}; results are not overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(a, k) for k in SETTINGS}
    stripped = sorted(k for k in os.environ if k.startswith(ENV_PREFIX))
    pairs = []
    for shape in shapes:
        start = time.time()
        res = dict(schema=SCHEMA, note=NOTE, arm=a.arm, env=ARMS[a.arm], env_stripped=stripped, settings=settings, shape=shape,
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **run_shape(a, shape, a.scratch))
        res["elapsed_s"] = time.time() - start
        (out_dir / result_name(a.arm, shape)).write_text(json.dumps(res, indent=1) + "\n")
        pairs.append((None, res))
        if res["status"] == "toggle_missing":
            report(pairs)
            sys.exit(f"--arm {a.arm} has no toggle to measure: {res['reason']}")
    report(pairs)
    print(f"wrote {len(pairs)} results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
