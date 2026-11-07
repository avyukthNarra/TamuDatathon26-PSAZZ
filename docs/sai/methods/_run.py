"""Grid runner shared by W1 methods: smoke -> full grid -> best cell via harness.run_method -> report."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import (load_split, smoke_split, evaluate, run_method, blend_gain, diff_ci,  # noqa: E402
                     read_pred_csv)

ROOT = HERE.parent


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_grid(mid, desc, cells, predict_fn, extra_cells=(), stop_after=None):
    """cells: list of (label, params-dict). predict_fn(split, params) -> (n,5).
    extra_cells are cross-cell variations (section 5) run and logged but eligible for best too."""
    v = load_split("validation")
    if "--smoke" in sys.argv or "--smoke-only" in sys.argv:
        sm = smoke_split(v)
        t0 = time.time()
        for lab, p in list(cells)[:3]:
            P = predict_fn(sm, p)
            assert np.isfinite(P).all() and np.abs(P.sum(1) - 1).max() < 1e-6
            ev = evaluate(P, sm, "smoke", bootstrap=0)
            print(f"[smoke] {mid} {lab}: {ev['points']:.2f}", flush=True)
        print(f"[smoke] ok {time.time() - t0:.1f}s")
        if "--smoke-only" in sys.argv:
            return
    log = []
    for lab, p in list(cells) + list(extra_cells):
        t0 = time.time()
        try:
            P = predict_fn(v, p)
            ev = evaluate(P, v, "dev", bootstrap=0)
        except Exception as e:  # report, do not hide
            print(f"[cell failed] {lab}: {e!r}")
            log.append({"label": lab, "params": _jsonable(p), "error": repr(e)})
            continue
        log.append({"label": lab, "params": _jsonable(p), "points": ev["points"],
                    "condition_ratios": ev["condition_ratios"], "seconds": time.time() - t0})
        print(f"{mid} {lab}: {ev['points']:.3f}  {time.time() - t0:.1f}s", flush=True)
    (ROOT / "grids" / f"{mid}.json").write_text(json.dumps(log, indent=1))
    ok = [r for r in log if "points" in r]
    best = max(ok, key=lambda r: r["points"])
    allp = dict(list(cells) + list(extra_cells))
    bp = allp[best["label"]]
    doc, P = run_method(mid, "claude-w1", desc + f" | best cell {best['label']}", _jsonable(bp),
                        lambda s: predict_fn(s, bp), v)
    PB = read_pred_csv(ROOT / "preds" / "B3_validation.csv", v)
    d = diff_ci(P, PB, v, "dev")
    bg = blend_gain(P, PB, v)
    res = json.loads((ROOT / "results" / f"{mid}.json").read_text())
    res["best_label"] = best["label"]
    res["n_grid_cells"] = len(ok)
    res["vs_B3"] = d
    res["blend_gain_vs_B3"] = {"points": bg["points"], "ci95": bg["ci95"], "w_mean": bg["w_mean"],
                               "vs_base": bg["vs_base"], "condition_ratios": bg["condition_ratios"]}
    (ROOT / "results" / f"{mid}.json").write_text(json.dumps(res, indent=1))
    extra_lines = []
    l2p = ROOT / "preds" / "L2_validation.csv"
    if mid != "L2" and l2p.is_file():
        PL = read_pred_csv(l2p, v)
        d2 = diff_ci(P, PL, v, "dev")
        bg2 = blend_gain(P, PL, v)
        res["vs_L2"] = d2
        res["blend_gain_vs_L2"] = {"points": bg2["points"], "vs_base": bg2["vs_base"], "w_mean": bg2["w_mean"]}
        (ROOT / "results" / f"{mid}.json").write_text(json.dumps(res, indent=1))
        extra_lines.append(f"vs L2 best ({json.loads((ROOT / 'results' / 'L2.json').read_text())['points']:.2f}): standalone {d2['diff']:+.2f} CI "
                           f"[{d2['ci95'][0]:+.2f}, {d2['ci95'][1]:+.2f}]; blend_gain vs L2 {bg2['vs_base']['diff']:+.2f} CI "
                           f"[{bg2['vs_base']['ci95'][0]:+.2f}, {bg2['vs_base']['ci95'][1]:+.2f}] (w {bg2['w_mean']:.2f})")
    b3 = json.loads((ROOT / "results" / "B3.json").read_text())["condition_ratios"]
    cr = res["condition_ratios"]
    top = sorted(ok, key=lambda r: -r["points"])[:3]
    lines = [
        f"# {mid}",
        f"Method: {desc}",
        f"Grid completed: {len(ok)} cells (grids/{mid}.json); top3: " + "; ".join(f"{r['label']}={r['points']:.2f}" for r in top),
        f"Best cell: {best['label']} params {json.dumps(_jsonable(bp))[:300]}",
        f"Points (dev, 72 ep): {res['points']:.2f} CI [{res['ci95'][0]:.2f}, {res['ci95'][1]:.2f}]; B3 = 45.82",
        "Condition ratios: " + ", ".join(f"{k}={x:.3f} (B3 {b3[k]:.3f})" for k, x in cr.items()),
        "Class error shares: " + ", ".join(f"{k}={x:.2f}" for k, x in res["class_error_shares"].items()),
        f"vs B3 standalone: {d['diff']:+.2f} CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",
        f"blend_gain vs B3: {bg['points']:.2f} pts, vs_base {bg['vs_base']['diff']:+.2f} CI "
        f"[{bg['vs_base']['ci95'][0]:+.2f}, {bg['vs_base']['ci95'][1]:+.2f}], mean w {bg['w_mean']:.2f}",
        *extra_lines,
        f"Runtime {res['runtime_seconds']:.1f}s per split; determinism diff {res['determinism_max_diff']}; stdlib-only: no (numpy)",
        "Note: best cell chosen on dev among the grid cells, so its points are optimistically biased by the selection.",
    ]
    (ROOT / "reports" / f"{mid}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res
