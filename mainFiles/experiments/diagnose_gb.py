"""Where does the GB model's error vs the true probabilities come from?"""
import os, sys, collections
os.environ.setdefault("HL", "4,16,64,128")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import gb_proto as g
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets

tq, th = load_inputs(g.PUB / "train"); vq, vh = load_inputs(g.PUB / "validation")
labels = read_labels(g.PUB / "train/labels.csv", tq)
T = {t.query_id: t for t in read_targets(g.PUB / "validation/targets.csv")}
cond = {r["query_id"]: r["condition"] for r in read_rows(g.PUB / "validation/conditions.csv", ("query_id", "condition"))}
ref = read_submission(g.PUB / "validation/reference.csv")
X, y = g.build_train(tq, th)
m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200, min_samples_leaf=80, max_depth=4, l2_regularization=5.0, random_state=0).fit(X, y)
gb = g.predict_queries(m, vq, vh)
Q = {q.query_id: q for q in vq}
ids = sorted(Q)
P = np.array([gb[i] for i in ids]); R = np.array([ref[i] for i in ids])
TR = np.array([T[i].probabilities for i in ids])
err = ((P - TR) ** 2).sum(1); rerr = ((R - TR) ** 2).sum(1)
print("mean sq err  gb %.4f  ref %.4f  (ratio %.3f)" % (err.mean(), rerr.mean(), err.mean() / rerr.mean()))
print("\nper-class share of gb error:", np.round(((P - TR) ** 2).sum(0) / ((P - TR) ** 2).sum(), 3))
print("per-class corr(pred,true):  ", [round(float(np.corrcoef(P[:, k], TR[:, k])[0, 1]), 3) for k in range(5)])
print("\nTrue-prob spread (std across queries):", np.round(TR.std(0), 3), " gb pred std:", np.round(P.std(0), 3))
# floor: predicting the global mean of true probs for every query
print("floor if we only knew the global true mean:", round(float(((TR - TR.mean(0)) ** 2).sum(1).mean()), 4))
for name, key in (("machine_type", lambda i: Q[i].machine_type), ("signal", lambda i: Q[i].signal), ("condition", lambda i: cond[i])):
    print(f"\nby {name}:")
    for v in sorted({key(i) for i in ids}, key=str):
        s = np.array([key(i) == v for i in ids])
        print(f"  {str(v):14s} n={s.sum():4d} gb {err[s].mean():.4f} ref {rerr[s].mean():.4f} ratio {err[s].mean()/rerr[s].mean():.3f}")
# does error depend on how many past obs the machine has?
hist_n = collections.Counter(); seqs = g.machine_seqs(vh)
n = np.array([len([k for k in seqs[(Q[i].episode_id, Q[i].machine_id)] if k < Q[i].cutoff]) for i in ids])
for lo, hi in ((0, 70), (70, 150), (150, 300)):
    s = (n >= lo) & (n < hi)
    if s.sum(): print(f"history {lo}-{hi}: n={s.sum()} gb {err[s].mean():.4f} ref {rerr[s].mean():.4f}")
# bias: mean pred vs mean truth by type for p_0, p_20
print("\nmean pred vs true by type (p0, p20):")
for t in g.TYPES:
    s = np.array([Q[i].machine_type == t for i in ids])
    print(f"  {t:9s} p0 {P[s,0].mean():.3f}/{TR[s,0].mean():.3f}  p20 {P[s,4].mean():.4f}/{TR[s,4].mean():.4f}")
# episodes: error spread (some episodes may dominate)
ep = collections.defaultdict(list)
for i, e in zip(ids, err): ep[Q[i].episode_id].append(e)
em = sorted((np.mean(v), k[-6:], cond[next(i for i in ids if Q[i].episode_id == k)]) for k, v in ep.items())
print("\nworst 5 episodes:", [(round(float(a), 3), b, c) for a, b, c in em[-5:]])
print("best 3 episodes:", [(round(float(a), 3), b, c) for a, b, c in em[:3]])
