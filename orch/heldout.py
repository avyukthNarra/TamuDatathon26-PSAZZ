"""Held-out-tick model selection: score any kit-compatible method on thousands of real next-spin labels.

make: copy a public/ folder, replacing test/ with pseudo-episodes built from a source split: for each
episode and k in 1..K, a copy of the episode truncated at cutoff-k (all 8 machines share the pseudo
cutoff, so no peer leakage); queries are the machines whose tick cutoff-k is observed; labels = that payout.
score: 5-class Brier vs observed labels (expected Brier = squared error vs true probs + a constant, so it
ranks models like the contest metric), with episode-bootstrap CI of the difference between two methods.
Caveat: pseudo cutoffs are up to K ticks earlier than real ones, so schedules anchored to the cutoff shift by k.
"""
import argparse, csv, shutil, collections, numpy as np
from pathlib import Path
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]; CLS = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}

def make(public, out, source, K):
    public, out = Path(public), Path(out)
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(public, out, ignore=shutil.ignore_patterns("test"))
    (out / "test").mkdir()
    Q = list(csv.DictReader(open(public / source / "queries.csv")))
    H = collections.defaultdict(list)
    for r in csv.DictReader(open(public / source / "history.csv")): H[r["episode_id"]].append(r)
    qrows, hrows, lrows = [], [], []
    for k in range(1, K + 1):
        eps = {}
        for q in Q: eps.setdefault(q["episode_id"], []).append(q)
        for e, qs in eps.items():
            c = int(qs[0]["cutoff"]) - k; pe = f"{e}-h{k}"
            obs = {}
            for r in H[e]:
                t = int(r["tick"])
                if t < c: hrows.append({"episode_id": pe, "machine_id": f"{r['machine_id']}-h{k}", "tick": t, "payout": r["payout"]})
                elif t == c: obs[r["machine_id"]] = r["payout"]
            for q in qs:
                if q["machine_id"] in obs:
                    qid = f"{q['query_id']}-h{k}"
                    qrows.append({"query_id": qid, "episode_id": pe, "machine_id": f"{q['machine_id']}-h{k}", "machine_type": q["machine_type"], "signal": q["signal"], "cutoff": c})
                    lrows.append({"query_id": qid, "outcome": obs[q["machine_id"]]})
    def w(name, rows, cols):
        with open(out / "test" / name, "w", newline="") as f:
            wr = csv.DictWriter(f, cols); wr.writeheader(); wr.writerows(rows)
    w("queries.csv", qrows, ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
    w("history.csv", hrows, ["episode_id", "machine_id", "tick", "payout"])
    w("sample_submission.csv", [{"query_id": r["query_id"], **{c: 0.2 for c in PCOLS}} for r in qrows], ["query_id"] + PCOLS)
    w("heldout_labels.csv", lrows, ["query_id", "outcome"])
    print(f"{out}: {len(qrows)} held-out queries from {source}, K={K}")

def brier(sub, labels):
    P = {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(sub))}
    L = {r["query_id"]: CLS[int(r["outcome"])] for r in csv.DictReader(open(labels))}
    per_ep = collections.defaultdict(list)
    for q, y in L.items():
        e = np.zeros(5); e[y] = 1; per_ep[q.rsplit("-m", 1)[0]].append(((P[q] - e) ** 2).sum())
    return {e: np.mean(v) for e, v in per_ep.items()}

def score(subs, labels, reps=2000, seed=7):
    res = [brier(s, labels) for s in subs]; eps = sorted(res[0])
    for s, r in zip(subs, res): print(f"{np.mean([r[e] for e in eps]):.5f}  {s}")
    if len(subs) > 1:
        rng = np.random.default_rng(seed); d = np.array([res[1][e] - res[0][e] for e in eps])
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
        print(f"second - first: {d.mean():+.5f}  95% CI [{np.quantile(boots, .025):+.5f}, {np.quantile(boots, .975):+.5f}]  (negative = second better)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    m = sp.add_parser("make"); m.add_argument("public"); m.add_argument("out"); m.add_argument("--source", default="validation"); m.add_argument("--K", type=int, default=4)
    s = sp.add_parser("score"); s.add_argument("labels"); s.add_argument("subs", nargs="+")
    a = ap.parse_args()
    make(a.public, a.out, a.source, a.K) if a.cmd == "make" else score(a.subs, a.labels)
