"""Cross-fit check for the priors the pair mixer reads from validation targets.

Mixer v3 reads the two payout-size modes per machine type from validation targets.csv; v4
also reads the dead/live win levels. Scoring those models on validation therefore uses the
answers it is scored on. Here the priors are fitted on one half of the validation episodes
(stratified by condition) and used to predict the other half, so every scored row is
out-of-fold. Prints dev points for the in-sample and cross-fitted versions of each combo.

Usage (from the repo root): python3 docs/sai/combos/crossfit_priors.py
Runs each mixer version in a subprocess, because v3 and v4 share the module name.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "crossfit"
SOURCES = {"v3": ROOT / "mainFiles", "v4": ROOT / "worktrees/_tm/mixer4/mainFiles"}

WORKER = r'''
import csv, json, sys
from pathlib import Path
src, version, fold_file, out = sys.argv[1:5]
sys.path[:0] = [src, "ROOT/mainFiles", "ROOT/mainFiles/combo"]
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting import pair_mixer as pm
import predict as combo
data = Path("ROOT/mainFiles/public")
q, h = load_inputs(data / "train")
labels = read_labels(data / "train/labels.csv", q)
rules = combo.fit_rules(q, h, labels)
vq, vh = load_inputs(data / "validation")
targets = read_targets(data / "validation/targets.csv")
folds = json.loads(Path(fold_file).read_text())
rows = {}
for name, fit_eps in folds.items():
    fit_eps = set(fit_eps)
    t = [x for x in targets if x.episode_id in fit_eps]
    qs = [x for x in vq if x.episode_id in fit_eps]
    kw = {"size_modes": pm.size_modes_from_targets(t, qs)}
    if version == "v4":
        kw["win_prior"] = pm.win_prior_from_targets(t)
    mixer = pm.adapt(q, h, labels, log=lambda *_: None, **kw)
    pred = combo.blend(rules, mixer, vq, vh)
    for x in vq:
        if (name == "all") or (x.episode_id not in fit_eps):
            rows.setdefault(name if name == "all" else "oof", {})[x.query_id] = list(pred[x.query_id])
Path(out).write_text(json.dumps(rows))
'''


def folds():
    """Two halves of the 96 validation episodes, alternating within each condition by episode id."""
    cond = {}
    with open(ROOT / "mainFiles/public/validation/conditions.csv") as f:
        next(f)
        for line in f:
            qid, c = line.strip().split(",")
            cond.setdefault(c, set()).add(qid.rsplit("-m", 1)[0])
    a, b = [], []
    for c in sorted(cond):
        for i, e in enumerate(sorted(cond[c])):
            (a if i % 2 == 0 else b).append(e)
    return {"all": a + b, "A": a, "B": b}


def main():
    OUT.mkdir(exist_ok=True)
    fold_file = OUT / "folds.json"
    fold_file.write_text(json.dumps(folds()))
    worker = OUT / "worker.py"
    worker.write_text(WORKER.replace("ROOT", str(ROOT)))
    for version, src in SOURCES.items():
        subprocess.run([sys.executable, str(worker), str(src), version, str(fold_file),
                        str(OUT / f"{version}.json")], check=True)

    sys.path.insert(0, str(ROOT / "docs/sai"))
    import numpy as np
    from harness.data import load_split
    from harness.results import read_pred_csv, to_csv
    from harness.scoring import evaluate, diff_ci
    v = load_split("validation")
    ref = read_pred_csv(ROOT / "docs/sai/preds/TM_COMBINED3_validation.csv", v)
    report = {}
    for version in SOURCES:
        res = json.loads((OUT / f"{version}.json").read_text())
        for kind, rows in res.items():
            P = np.array([rows[q] for q in v.qid])
            P = P / P.sum(1, keepdims=True)
            name = f"COMBINED_{version}_{'insample' if kind == 'all' else 'crossfit'}"
            to_csv(OUT / f"{name}_validation.csv", v, P)
            r, d = evaluate(P, v, "dev"), diff_ci(P, ref, v, "dev")
            report[name] = {"dev": r["points"], "ci95": r["ci95"], "diff_vs_C3": d,
                            "ratios": r["condition_ratios"]}
            print(f"{name:28s} dev {r['points']:.2f} {np.round(r['ci95'], 2)} "
                  f"vs C3 {d['diff']:+.2f} {np.round(d['ci95'], 2)}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
