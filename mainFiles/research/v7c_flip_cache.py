"""v7c forecasts WITH and WITHOUT its episode flip swap, for validation (cross-fit halves) and train
(8 held-out folds, cutoff + 10 ticks). Folds run in parallel. Run from mainFiles/."""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("runs/research-cache/v7c_flip.json")

def fit(fit_q, fit_h, labels, targets, vq):
    from v7c_forecasting.entry_config import USE_ROUND2
    from v7c_forecasting.forecast import fit_flip, fit_size
    from v7c_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_by_type_from_targets
    model = adapt(fit_q, fit_h, labels, log=lambda *_: None, size_modes=size_modes_from_targets(targets, vq),
                  win_prior=win_prior_by_type_from_targets(targets, vq, two_means=USE_ROUND2), round2=USE_ROUND2)
    return model, fit_size(fit_q, fit_h, True, labels, True), fit_flip(model, fit_q, fit_h, True)

def both(model, size_rule, gate, qs, hist):
    from v7c_forecasting.forecast import predict_split
    a = predict_split(model, qs, hist, size_rule, gate)
    b = predict_split(model, qs, hist, size_rule, None)
    return {q: (list(a[q]), list(b[q])) for q in a}

def val_job(half):
    from v7c_forecasting.io import load_inputs, read_labels, read_targets
    from slot_forecasting.experiments import episode_folds
    pub = Path("public"); tq, th = load_inputs(pub / "train"); labels = read_labels(pub / "train/labels.csv", tq)
    vq, vh = load_inputs(pub / "validation"); vt = read_targets(pub / "validation/targets.csv")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "validation/conditions.csv"))}
    halves = episode_folds(vt, cond); scored, other = halves[half], halves[1 - half]
    model, size_rule, gate = fit(tq, th, labels, [t for t in vt if t.episode_id in other], vq)
    qs = tuple(q for q in vq if q.episode_id in scored); keys = {(q.episode_id, q.machine_id) for q in qs}
    return {f"{k}|v": v for k, v in both(model, size_rule, gate, qs, tuple(p for p in vh if (p.episode_id, p.machine_id) in keys)).items()}

def train_job(fold):
    from v7c_forecasting.io import load_inputs, read_labels, read_targets
    from v7c_forecasting.schema import Query
    from slot_forecasting.pair_mixer import episode_kfolds
    pub = Path("public"); tq, th = load_inputs(pub / "train"); labels = read_labels(pub / "train/labels.csv", tq)
    vq, _ = load_inputs(pub / "validation"); vt = read_targets(pub / "validation/targets.csv")
    tcond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "train/conditions.csv"))}
    held = {e for e, f in episode_kfolds(tq, tcond).items() if f == fold}
    fq = tuple(q for q in tq if q.episode_id not in held); fh = tuple(p for p in th if p.episode_id not in held)
    model, size_rule, gate = fit(fq, fh, {q.query_id: labels[q.query_id] for q in fq}, vt, vq)
    rows = [r for r in json.load(open("runs/research-cache/train.json")) if r["episode"] in held]
    out = {}
    for back in range(11):
        qs = tuple(Query(r["qid"], r["episode"], r["machine"], r["type"], r["signal"], r["cutoff"]) for r in rows if r["back"] == back)
        if not qs: continue
        lim = {(q.episode_id, q.machine_id): q.cutoff for q in qs}
        hist = tuple(p for p in th if (p.episode_id, p.machine_id) in lim and p.tick < lim[(p.episode_id, p.machine_id)])
        out.update({f"{k}|{back}": v for k, v in both(model, size_rule, gate, qs, hist).items()})
    return out

if __name__ == "__main__":
    with Pool(8) as pool:
        parts = pool.map(val_job, [0, 1]) + pool.map(train_job, range(8))
    merged = {}
    for p in parts: merged.update(p)
    json.dump(merged, open(OUT, "w")); print("rows", len(merged))
