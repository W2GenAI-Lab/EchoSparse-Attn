"""CPU tests of what benchmarks/vsa2_kernel_bench.py lets a result say. No torch, GPU or build."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import vsa2_kernel_bench as b


def err(rel_l2=0.0, max_abs=0.0, n_diff=0, n_nan=0):
    return dict(rel_l2=rel_l2, max_abs=max_abs, scale=1.0, n_diff=n_diff, n_bit_diff=n_diff, n_nan=n_nan, n_elements=100)


def child(o="o", sel="s", pad=0, ms=1.0):
    return dict(digests=dict(o=o, selection=sel), padded_slots=5, padded_nonzero=dict(o=pad, dq=0, dk=0, dv=0, dgate=0),
                rows={r: [ms, ms, ms] for r in b.ROWS})


def test_shape_matrix_and_partial_counts():
    shapes = b.shape_matrix()
    assert len(shapes) == 4 * 2 * 2 + 1 and sum(s["partial"] for s in shapes) == 1
    assert b.shape_key(shapes[-1]) == "L16384_H16_D128_partial"
    counts = b.partial_counts(256)
    assert min(counts) >= 1 and max(counts) == 64 and sum(c < 64 for c in counts) > 10
    assert b.shape_error(dict(L=100, D=64)) and b.shape_error(dict(L=64, D=32)) and b.shape_error(dict(L=4096, D=64)) is None


def test_each_process_sees_only_its_own_toggle():
    env = dict(PATH="x", ECHOSPARSE_VSA_FUSED_COLS="1", ECHOSPARSE_VSA_TK_BWD_OPTS="3", ECHOSPARSE_VSA_BWD_VARIANT="1")
    assert b.child_env("baseline", env) == dict(PATH="x")
    assert b.child_env("tk_bwd_store", env) == dict(PATH="x", ECHOSPARSE_VSA_TK_BWD_STORE="1")
    assert b.ORDER == ("default", "arm", "arm", "default")


def test_a_toggle_nothing_contains_is_reported():
    assert b.missing_toggles("baseline", "", b"") == []
    assert b.missing_toggles("fused_cols", "other", b"other") == ["ECHOSPARSE_VSA_FUSED_COLS"]
    assert b.missing_toggles("fused_cols", "", b"\0ECHOSPARSE_VSA_FUSED_COLS\0") == [], "read by the extension"
    assert b.missing_toggles("tk_bwd_store", 'X = "ECHOSPARSE_VSA_TK_BWD_STORE"', b"") == [], "read by the operator"


def test_ratio_and_displacement_come_from_one_run():
    rows = b.time_rows([child(ms=1.0), child(ms=2.0), child(ms=2.0), child(ms=1.1)])
    t = rows["vsa2.fwd"]
    assert abs(t["ratio"] - 2.0 / 1.05) < 1e-9 and abs(t["aa_displacement_same_run"] - 1.1) < 1e-9
    assert rows["b_estimate_ms"]["measured"] is False


def test_gradient_verdicts():
    aa = [err(1e-5, 0.01, 1000)]
    assert b.gradient_verdict([err(1e-5, 0.01, 1200)], aa, 1e-3) == "OK", "the noise floor is not a difference"
    assert b.gradient_verdict([err(9e-5, 0.01, 1200)], aa, 1e-3) == "ABOVE_AA"
    assert b.gradient_verdict([err(1e-5, 0.2, 1200)], aa, 1e-3) == "ABOVE_AA"
    assert b.gradient_verdict([err(1e-5, 0.01, 9000)], aa, 1e-3) == "ABOVE_AA"
    assert b.gradient_verdict([err(5e-3, 0.2, 9000)], aa, 1e-3) == "OVER_TOL"
    assert b.gradient_verdict([err(float("nan"), 0.0, 1)], aa, 1e-3) == "FAIL"
    assert b.gradient_verdict([err(0.0, 0.0, 0, n_nan=1)], aa, 1e-3) == "FAIL"
    assert b.gradient_verdict([err(1e-9, 1e-9, 1)], [err()], 1e-3) == "ABOVE_AA", "any difference exceeds an exact A/A"


def test_forward_is_held_to_bytes_and_gradients_never_are():
    def run(children, e=err(1e-5, 0.01, 1000)):
        return b.equivalence(children, lambda n, i, j: e, 1e-3)
    same = [child(), child(), child(), child()]
    assert run(same)["verdict"] == "OK", "differing gradients alone must not fail"
    assert run([child(), child(o="x"), child(o="x"), child()])["verdict"] == "FAIL"
    assert run([child(), child(), child(o="x"), child()])["verdict"] == "FAIL"
    assert run([child(), child(sel="x"), child(sel="x"), child()])["verdict"] == "FAIL"
    assert run([child(), child(pad=1), child(), child()])["verdict"] == "FAIL", "padding is where this breaks"
    assert run([child(o="a"), child(o="a"), child(o="b"), child(o="b")])["verdict"] == "NONDETERMINISTIC_FORWARD"
    assert run([child(), child(), child(), child(sel="x")])["verdict"] == "FAIL"


def result(arm="baseline", build="x", shape=None):
    shape = shape or dict(B=1, H=8, L=4096, D=64, partial=False)
    identity = dict(build=build, hardware=1, software=1, shape=1, timing=1, inputs=1)
    settings = dict(warmup=10, iters=30, batches=3, seed=0, topk_frac=0.1, grad_tol=1e-3)
    return dict(schema=b.SCHEMA, arm=arm, env=b.ARMS[arm], settings=settings, shape=shape, status="ok", identity=identity)


def test_compare_refuses_what_cannot_be_read_together():
    assert b.compare_error(result(), result("fused_cols")) is None
    assert "A/A" in b.compare_error(result("fused_cols"), result())
    assert "shapes differ" in b.compare_error(result(), result("fused_cols", shape=dict(B=1, H=8, L=8192, D=64, partial=False)))
    assert "build" in b.compare_error(result(), result("fused_cols", build="y"))
    other = result("fused_cols")
    other["settings"]["seed"] = 1
    assert "seed" in b.compare_error(result(), other)
    assert "must be" in b.compare_error(dict(schema="other"), result())


def test_directories_pair_by_shape_and_refuse_a_missing_one():
    with tempfile.TemporaryDirectory() as d:
        for sub, arm in (("aa", "baseline"), ("x", "fused_cols")):
            os.mkdir(os.path.join(d, sub))
            r = result(arm)
            with open(os.path.join(d, sub, b.result_name(arm, r["shape"])), "w") as f:
                json.dump(r, f)
        assert len(b.pair_results(os.path.join(d, "aa"), os.path.join(d, "x"))) == 1
        extra = result("fused_cols", shape=dict(B=1, H=8, L=8192, D=64, partial=False))
        with open(os.path.join(d, "x", b.result_name("fused_cols", extra["shape"])), "w") as f:
            json.dump(extra, f)
        try:
            b.pair_results(os.path.join(d, "aa"), os.path.join(d, "x"))
        except ValueError as e:
            assert "shapes differ" in str(e)
            return
    raise AssertionError("a missing shape was accepted")


def test_a_verdict_is_reread_against_every_floor():
    e = err(1e-5, 0.01, 1000)
    eq = b.equivalence([child()] * 4, lambda n, i, j: err(1e-3 / 2, 0.5, 90000) if (i, j) in ((1, 0), (2, 3)) else e, 1e-3)
    assert eq["verdict"] == "ABOVE_AA"
    res = dict(status="ok", equivalence=eq)
    wide = dict(status="ok", equivalence=b.equivalence([child()] * 4, lambda n, i, j: err(1e-3, 0.5, 90000), 1e-3))
    assert b.combined_verdict(res) == "ABOVE_AA" and b.combined_verdict(res, wide) == "OK"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all vsa2_kernel_bench CPU tests passed")
