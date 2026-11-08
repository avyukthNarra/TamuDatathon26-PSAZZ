"""W0 acceptance: reproduces B0=30.00 and B3=46.07, cross-checks the scorer against
slot_forecasting.contest.contest_score, checks leakage/NaN handling, determinism, audit guard.
Run from docs/sai:  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
                    nice -n 10 timeout 300 python3 -m harness.selfcheck"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_split, evaluate, diff_ci, run_method, audit_episodes
from harness.baselines import pooled, b3_prototype
from harness.data import CONDITIONS, DATA

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mainFiles"))


def main():
    v = load_split("validation")
    ok = True
    cut = v.cutoff[:, None]
    ticks = np.arange(v.grid.shape[1])[None, :]
    assert np.isnan(v.grid[ticks >= cut]).all(), "grid contains ticks >= cutoff"
    print("grid:", v.grid.shape, "observed fraction", round(float((~np.isnan(v.grid)).mean()), 3),
          "(NaN = unobserved, not 0)")
    a = audit_episodes(v)
    per = {c: sum(1 for e in a if v.condition[v.episode_rows(e)[0]] == c) for c in CONDITIONS}
    print("audit episodes:", len(a), per)
    assert len(a) == 24 and set(per.values()) == {6}

    P0 = pooled(v)
    e0 = evaluate(P0, v, scope="all", who="selfcheck", bootstrap=200)
    print(f"B0 pooled (all 96): {e0['points']:.3f}  (expect 30.00)")
    ok &= abs(e0["points"] - 30.0) < 0.01
    P3 = b3_prototype(v)
    e3 = evaluate(P3, v, scope="all", who="selfcheck", bootstrap=200)
    print(f"B3 prototype (all 96): {e3['points']:.3f}  (expect 46.07)")
    ok &= abs(e3["points"] - 46.07) < 0.01
    from slot_forecasting.contest import contest_score
    from slot_forecasting.io import read_targets
    ts = read_targets(DATA / "validation/targets.csv")
    groups = {q: c for q, c in zip(v.qid, v.condition)}
    ref = {q: tuple(r) for q, r in zip(v.qid, v.ref)}
    off = contest_score({q: tuple(r) for q, r in zip(v.qid, P3)}, ts, groups, ref, max_points=v.max_points,
                        version="experimentation-suite-v3")
    print(f"official contest_score on B3: {off['technical_points']:.6f}  harness: {e3['points']:.6f}")
    ok &= abs(off["technical_points"] - e3["points"]) < 1e-9
    d = evaluate(P3, v, "dev")
    print("B3 dev:", round(d["points"], 3), [round(x, 2) for x in d["ci95"]], d["condition_ratios"])
    print("B3 - B0 on dev:", diff_ci(P3, P0, v, "dev"))
    doc, _ = run_method("SELFCHECK_B3", "claude", "selfcheck copy of B3", {}, b3_prototype, v)
    ok &= doc["determinism_max_diff"] < 1e-8
    try:
        evaluate(P3, v, "audit")
        ok = False
        print("FAIL: audit scope allowed without who")
    except PermissionError:
        print("audit guard OK")
    print("W0 SELFCHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
