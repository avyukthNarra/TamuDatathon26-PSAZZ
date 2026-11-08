"""Score Pranav's mixer6 (814cb60: 100-spin win memory, size_eta 4) alone and in rule blends.

Priors (size modes, per-type win prior) are read from validation targets either in-sample or
cross-fitted (fit on one half of the validation episodes, predict the other half; halves as in
combos/crossfit_priors.py). Mixer and rule forecasts are saved separately, then blended here at
several rule weights and scored on dev against the mixer5 20/80 blend.

Usage (from the repo root): python3 docs/sai/round/score_mixer6.py
Needs worktrees/_tm/mixer6 (git archive 814cb60 | tar -x -C worktrees/_tm/mixer6).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "mixer6"
SRC = ROOT / "worktrees/_tm/mixer6/mainFiles"
RULE_WEIGHTS = (0.0, 0.1, 0.2)

WORKER = r'''
import json, sys
from pathlib import Path
src, fold_file, out = sys.argv[1:4]
sys.path[:0] = [src, "ROOT/mainFiles", "ROOT/mainFiles/combo"]
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting import pair_mixer as pm
import predict as combo
data = Path("ROOT/mainFiles/public")
q, h = load_inputs(data / "train")
labels = read_labels(data / "train/labels.csv", q)
vq, vh = load_inputs(data / "validation")
rule_models = combo.fit_rules(q, h, labels)
rules = [m.predict(vq, vh) for m in rule_models]
rule = {x.query_id: [sum(f[x.query_id][i] for f in rules) / len(rules) for i in range(5)] for x in vq}
targets = read_targets(data / "validation/targets.csv")
folds = json.loads(Path(fold_file).read_text())
mix = {}
for name, fit_eps in folds.items():
    fit_eps = set(fit_eps)
    t = [x for x in targets if x.episode_id in fit_eps]
    qs = [x for x in vq if x.episode_id in fit_eps]
    model = pm.adapt(q, h, labels, log=lambda *_: None, size_modes=pm.size_modes_from_targets(t, qs),
                     win_prior=pm.win_prior_by_type_from_targets(t, qs))
    pred = model.predict(vq, vh)
    kind = "insample" if name == "all" else "crossfit"
    for x in vq:
        if name == "all" or x.episode_id not in fit_eps:
            mix.setdefault(kind, {})[x.query_id] = list(pred[x.query_id])
Path(out).write_text(json.dumps({"rule": rule, "mixer": mix}))
'''


def main():
    sys.path.insert(0, str(ROOT / "docs/sai/combos"))
    from crossfit_priors import folds
    OUT.mkdir(exist_ok=True)
    fold_file = OUT / "folds.json"
    fold_file.write_text(json.dumps(folds()))
    worker = OUT / "worker.py"
    worker.write_text(WORKER.replace("ROOT", str(ROOT)))
    raw = OUT / "raw.json"
    subprocess.run([sys.executable, str(worker), str(SRC), str(fold_file), str(raw)], check=True)
    res = json.loads(raw.read_text())
    worker.unlink()
    raw.unlink()

    sys.path.insert(0, str(ROOT / "docs/sai"))
    import numpy as np
    from harness.data import load_split
    from harness.results import read_pred_csv, to_csv
    from harness.scoring import evaluate, diff_ci
    v = load_split("validation")
    ref = read_pred_csv(Path(__file__).resolve().parent / "mixer5/COMBINED_v5_insample_validation.csv", v)
    R = np.array([res["rule"][q] for q in v.qid])
    report = {}
    for kind, rows in res["mixer"].items():
        M = np.array([rows[q] for q in v.qid])
        for w in RULE_WEIGHTS:
            P = w * R + (1 - w) * M
            P = P / P.sum(1, keepdims=True)
            name = f"MIXER6_rule{int(w * 100):02d}_{kind}"
            to_csv(OUT / f"{name}_validation.csv", v, P)
            r, d = evaluate(P, v, "dev"), diff_ci(P, ref, v, "dev")
            report[name] = {"dev": r["points"], "ci95": r["ci95"], "diff_vs_mixer5_blend": d,
                            "ratios": r["condition_ratios"]}
            print(f"{name:28s} dev {r['points']:.2f} {np.round(r['ci95'], 2)} "
                  f"vs mixer5 20/80 {d['diff']:+.2f} {np.round(d['ci95'], 2)} "
                  f"{ {k: round(x, 3) for k, x in r['condition_ratios'].items()} }")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
