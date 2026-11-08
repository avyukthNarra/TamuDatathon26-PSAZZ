"""Pair frequencies and per-machine levels from ALL histories (train, validation, test), using only
ticks 8..95 of long histories, which precede any drift (starts >= cutoff-96 >= 96) or shared flip (~0.66*cutoff).
Per machine: for each of the 28 lag pairs, max log-likelihood of a two-rate parity model; best pair and its
margin over the runner-up (nats). Confident machines (margin >= 3) give pair frequencies and level estimates.
"""
import itertools, json, collections, numpy as np
from common import *
PAIRS = list(itertools.combinations(range(1, 9), 2))
def ll2(w, n):
    p = np.clip(w / np.maximum(n, 1e-9), 1e-6, 1 - 1e-6); return (w * np.log(p) + (n - w) * np.log(1 - p)).sum()
out = {}; lev = collections.defaultdict(list)
for split in ("train", "validation", "test"):
    d = load(split); cnt = collections.Counter(); conf = 0; tot = 0
    for q in d["Q"]:
        h = d["H"][q["machine_id"]]
        if len([t for t in h if t < 96]) < 90: continue  # long histories only
        tot += 1; W = {t: int(h[t] > 0) for t in range(96) if t in h}
        res = []
        for a, b in PAIRS:
            w = np.zeros(2); n = np.zeros(2)
            for t in range(8, 96):
                k = int(W[t - a] == W[t - b]); w[k] += W[t]; n[k] += 1
            res.append((ll2(w, n), (a, b), w / n))
        res.sort(key=lambda r: -r[0])
        if res[0][0] - res[1][0] >= 3:
            conf += 1; cnt[res[0][1]] += 1
            r = res[0][2]; lev[q["machine_type"]].append((int(q["signal"]), r.min(), r.max()))
    out[split] = {"long_machines": tot, "confident": conf, "pairs": {f"{a},{b}": c for (a, b), c in cnt.most_common()}}
    print(split, "long", tot, "confident", conf, "top pairs", cnt.most_common(6), "other pairs total", sum(c for p, c in cnt.items() if p not in [(1, 3), (2, 5), (4, 7)]))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
print("levels from confident machines (all splits, pre-change ticks): logit mean (sd) by signal -1/0/1")
out["levels"] = {}
for ty, L in lev.items():
    L = np.array(L); row = {}
    for k, name in ((1, "dead"), (2, "live")):
        row[name] = {str(s): [round(float(lg(L[L[:, 0] == s, k]).mean()), 2), round(float(lg(L[L[:, 0] == s, k]).std()), 2), int((L[:, 0] == s).sum())] for s in (-1, 0, 1)}
    out["levels"][ty] = row; print(ty, row)
json.dump(out, open("priors/pairs_all_histories.json", "w"), indent=1)
