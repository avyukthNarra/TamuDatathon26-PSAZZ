import os, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets

PUB = Path(__file__).resolve().parents[1] / "public"
PAY = (0, 1, 2, 5, 20)
HL = tuple(int(x) for x in os.environ.get("HL", "8,32").split(","))
TYPES = ("steady", "balanced", "jackpot")
TICK = os.environ.get("TICK") == "1"       # add cutoff/tick as a feature
WEIGHT = os.environ.get("WEIGHT") == "1"   # weight late-tick training rows more
LAGRESP = os.environ.get("LAGRESP") == "1" # per-machine lag response features
_CUM = {}
MAXT = 320


def lag_cum(seq):
    """Cumulative counts per lag for win(s) given win/loss at s-lag, using only observed pairs."""
    present = np.zeros(MAXT); win = np.zeros(MAXT)
    for k, v in seq.items():
        present[k] = 1; win[k] = v > 0
    out = []
    for lag in range(1, 9):
        pa = np.zeros(MAXT); wa = np.zeros(MAXT)
        pa[lag:] = present[:-lag]; wa[lag:] = win[:-lag]
        valid = present * pa
        cols = [valid * wa, valid * wa * win, valid * (1 - wa), valid * (1 - wa) * win]
        out.append(np.cumsum(np.stack(cols), 1))
    return np.cumsum(np.stack([present, present * win]), 1), np.stack(out), present, win


def lag_response(seq, t, m=10.0):
    key = id(seq)
    if key not in _CUM:
        _CUM[key] = (seq, lag_cum(seq))
    tot, cum, present, win = _CUM[key][1]
    n = tot[0, t - 1]
    p = tot[1, t - 1] / n if n else 0.4
    diffs, effect = [], 0.0
    for i, lag in enumerate(range(1, 9)):
        n1, w1, n0, w0 = cum[i, :, t - 1]
        d = (w1 + m * p) / (n1 + m) - (w0 + m * p) / (n0 + m)
        diffs.append(d)
        a = win[t - lag] if present[t - lag] else 0.5
        effect += d * (a - 0.5)
    return diffs + [effect]



def machine_seqs(history):
    seqs = collections.defaultdict(dict)
    for h in history:
        seqs[(h.episode_id, h.machine_id)][h.tick] = h.payout
    return seqs


def feats(q, seqs, ep_machines, t):
    seq = seqs[(q.episode_id, q.machine_id)]
    ticks = np.array([k for k in seq if k < t])
    if len(ticks) == 0:
        return None
    pays = np.array([seq[k] for k in ticks])
    onehot = np.stack([(pays == p) for p in PAY], 1).astype(float)
    f = [float(q.machine_type == m) for m in TYPES] + [float(q.signal == s) for s in (-1, 0, 1)]
    f.append(len(ticks))
    for h in HL:
        w = 0.5 ** ((t - 1 - ticks) / h)
        f.extend((w[:, None] * onehot).sum(0) / w.sum())
        f.append(w.sum())
    for lag in range(1, 9):
        v = seq.get(t - lag)
        f.extend([float(v is not None), float(v is not None and v > 0), float(v is not None and v >= 2), float(v is not None and v >= 5)])
    # episode context: other machines' recent win rate / mean payout (ticks < t)
    others = []
    for m in ep_machines[q.episode_id]:
        if m == q.machine_id:
            continue
        s = seqs[(q.episode_id, m)]
        ts = np.array([k for k in s if k < t])
        if len(ts) == 0:
            continue
        w = 0.5 ** ((t - 1 - ts) / 16)
        pv = np.array([s[k] for k in ts])
        others.append(((w * (pv > 0)).sum() / w.sum(), (w * pv).sum() / w.sum()))
    o = np.array(others) if others else np.zeros((1, 2))
    f.extend(o.mean(0))
    if TICK:
        f.append(float(t))
    if LAGRESP:
        f.extend(lag_response(seq, t))
    return f


def build_train(queries, history, stride=2, min_prior=24):
    seqs = machine_seqs(history)
    qmap = {(q.episode_id, q.machine_id): q for q in queries}
    ep_machines = collections.defaultdict(list)
    for (e, m) in seqs:
        ep_machines[e].append(m)
    X, y, T = [], [], []
    for key, seq in seqs.items():
        q = qmap[key]
        ticks = sorted(seq)
        for i, t in enumerate(ticks):
            if i < min_prior or i % stride:
                continue
            f = feats(q, seqs, ep_machines, t)
            if f is None:
                continue
            X.append(f)
            y.append(PAY.index(seq[t])); T.append(t)
    return np.array(X), np.array(y), np.array(T)


def predict_queries(model, queries, history):
    seqs = machine_seqs(history)
    ep_machines = collections.defaultdict(list)
    for (e, m) in seqs:
        ep_machines[e].append(m)
    out = {}
    for q in queries:
        f = feats(q, seqs, ep_machines, q.cutoff)
        out[q.query_id] = f
    ids = list(out)
    P = model.predict_proba(np.array([out[i] for i in ids]))
    full = np.zeros((len(ids), 5))
    full[:, model.classes_] = P
    return {i: tuple(full[k]) for k, i in enumerate(ids)}


if __name__ == "__main__":
    tq, th = load_inputs(PUB / "train")
    vq, vh = load_inputs(PUB / "validation")
    targets = read_targets(PUB / "validation/targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(PUB / "validation/conditions.csv", ("query_id", "condition"))}
    ref = read_submission(PUB / "validation/reference.csv")
    labels = read_labels(PUB / "train/labels.csv", tq)
    X, y, T = build_train(tq, th)
    sw = np.where(T >= 150, 3.0, 1.0) if WEIGHT else None
    print("train", X.shape, np.bincount(y))
    base = FrequencyForecaster("pooled_recent", prior_strength=10, half_life=16).fit(tq, th, labels).predict(vq, vh)
    tuning, audit = episode_folds(targets, groups)
    def rep(name, p):
        out = [f"{name:26s}"]
        for label, eps in (("all", set(tuning) | set(audit)), ("tune", tuning), ("audit", audit)):
            a = subset_score(p, targets, groups, ref, eps, 60)
            out.append(f"{label} {a['technical_points']:.3f}")
        a = subset_score(p, targets, groups, ref, set(tuning) | set(audit), 60)
        out.append(" ".join(f"{c[:5]}={v['loss_ratio']:.3f}" for c, v in a["by_condition"].items()))
        print("  ".join(out))
    rep("pooled_recent(10,16)", base)
    m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200, min_samples_leaf=80, max_depth=4, l2_regularization=5.0, random_state=0).fit(X, y, sample_weight=sw)
    rep(f"gb tick={TICK} w={WEIGHT} lr={LAGRESP}", predict_queries(m, vq, vh))
