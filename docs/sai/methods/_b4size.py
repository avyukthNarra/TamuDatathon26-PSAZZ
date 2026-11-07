"""Driver for A*_B4: _freq size-split engine with B4's win probability (_core lag-7 + pairs, lam .05) and B4 as the base."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _freq as f
import _core as K
from _freq import load_split, smoke_split, evaluate, diff_ci, blend_gain, run_method, read_pred_csv, SAI, class_shares
import numpy as np

B4CFG = K.C(lam=0.05, lags=(7,), pairs=((1, 3), (2, 5), (4, 7)))
_PWC = {}


def _pw_b4(split, qs, xs, e5r):
    k = (split.name, split.n, str(split.qid[0]))
    if k not in _PWC:
        _PWC[k] = K.predict_pw(split, B4CFG)
    return _PWC[k]


f.pw_ridge = _pw_b4


def run_b4(mid, cells, desc):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(k) == "1", k
    v = load_split("validation")
    sm = smoke_split(v)
    pts = [evaluate(f.make(c)(sm), sm, "smoke", bootstrap=0)["points"] for c in cells[:2]]
    print(f"[{mid}] smoke {pts}", flush=True)
    if "--smoke" in sys.argv:
        return
    b4 = read_pred_csv(str(SAI / "preds/B4_validation.csv"), v)
    out, best = [], None
    gpath = SAI / "grids" / f"{mid}.json"
    for p in cells:
        t0 = time.perf_counter()
        ev = evaluate(f.make(p)(v), v, "dev", bootstrap=0)
        out.append({"params": p, "points": ev["points"], "condition_ratios": ev["condition_ratios"], "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[{mid}] {ev['points']:.3f} {json.dumps(p)}", flush=True)
        if best is None or ev["points"] > best[0]:
            best = (ev["points"], p)
        gpath.write_text(json.dumps(out, indent=1, default=float))
    p = best[1]
    d, P = run_method(mid, "claude-w1", desc, p, f.make(p), v)
    bg = blend_gain(P, b4, v)
    d["extra"] = {"vs_B4": diff_ci(P, b4, v, "dev"), "blend_gain_vs_B4": {"points": bg["points"], "ci95": bg["ci95"], "vs_base": bg["vs_base"], "w_mean": bg["w_mean"]},
                  "class_shares_B4": class_shares(b4, v, "dev"), "grid_cells_completed": len(out), "cell0_points": out[0]["points"]}
    (SAI / "results" / f"{mid}.json").write_text(json.dumps(d, indent=1, default=float))
    print(f"[{mid}] BEST {d['points']:.3f} {d['ci95']} cell0 {out[0]['points']:.3f} vsB4 {d['extra']['vs_B4']} blend {bg['vs_base']} w {bg['w_mean']:.2f}")
    print("shares", d["class_error_shares"])
