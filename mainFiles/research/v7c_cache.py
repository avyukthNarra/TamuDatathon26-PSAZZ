"""Out-of-fold train forecasts from the vendored v7c pipeline, same rows as runs/research-cache/train.json
(8 episode folds; v7c refit WITHOUT the fold; cutoff + last 10 observed ticks). Run from mainFiles/."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v7c_forecasting.entry_config import USE_ROUND2
from v7c_forecasting.forecast import fit_flip, fit_size, predict_split
from v7c_forecasting.io import load_inputs, read_labels, read_targets
from v7c_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_by_type_from_targets
from v7c_forecasting.schema import Query
from slot_forecasting.pair_mixer import episode_kfolds
pub = Path("public")
tq, th = load_inputs(pub / "train"); labels = read_labels(pub / "train/labels.csv", tq)
tcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "train/conditions.csv"))}
vq, _ = load_inputs(pub / "validation"); vt = read_targets(pub / "validation/targets.csv")
modes, prior = size_modes_from_targets(vt, vq), win_prior_by_type_from_targets(vt, vq, two_means=USE_ROUND2)
rows = json.load(open("runs/research-cache/train.json"))
want = {}
for r in rows: want.setdefault(r["episode"], []).append(r)
out = {}
folds = episode_kfolds(tq, tcond)
for fold in sorted(set(folds.values())):
    held = {e for e, f in folds.items() if f == fold}
    fq = tuple(q for q in tq if q.episode_id not in held); fh = tuple(p for p in th if p.episode_id not in held)
    fl = {q.query_id: labels[q.query_id] for q in fq}
    model = adapt(fq, fh, fl, log=lambda *_: None, size_modes=modes, win_prior=prior, round2=USE_ROUND2)
    size_rule = fit_size(fq, fh, True, fl, True); gate = fit_flip(model, fq, fh, True)
    for back in range(0, 11):
        qs = tuple(Query(r["qid"], r["episode"], r["machine"], r["type"], r["signal"], r["cutoff"])
                   for e in held for r in want.get(e, []) if r["back"] == back)
        if not qs: continue
        lim = {(q.episode_id, q.machine_id): q.cutoff for q in qs}
        hist = tuple(p for p in th if (p.episode_id, p.machine_id) in lim and p.tick < lim[(p.episode_id, p.machine_id)])
        for qid, p in predict_split(model, qs, hist, size_rule, gate).items():
            out[f"{qid}|{back}"] = list(p)
    print(f"fold {fold} done", flush=True)
json.dump(out, open("runs/research-cache/v7c_train.json", "w"))
print("rows", len(out))
