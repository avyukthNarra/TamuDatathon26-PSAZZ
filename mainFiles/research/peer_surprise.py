import sys, random, math
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
    import peer_fixed as F
stack = P.stack
# peer surprise: peers' recent win share minus the peers' own current forecasts (same episode, same cutoff)
for rows in (P.VAL, P.TRAIN):
    groups = defaultdict(list)
    for r in rows: groups[(r["episode"], r["cutoff"])].append(r)
    for r in rows:
        peers = [o for o in groups[(r["episode"], r["cutoff"])] if o is not r]
        sh = r["x"]["peers 1-5"]
        r["x"]["peer surprise"] = None if sh is None or not peers else sh - sum(o["f"] for o in peers) / len(peers)
        r["x"]["peer mean forecast"] = sum(o["f"] for o in peers) / len(peers) if peers else None
for k in ("peer surprise", "peer mean forecast"):
    vals = [r["x"][k] for r in P.TRAIN if r["x"][k] is not None]; P.MEANS[k] = sum(vals) / len(vals)
for names in (["peer surprise"], ["peers 1-5", "peer mean forecast"]):
    b = P.fit(P.TRAIN, names); print(names, "train-fitted coef", [round(x, 2) for x in b])
    for scale in (0.5, 1.0):
        bb = [x * scale for x in b]
        s = P.val_score(names, bb)
        per = {}
        for which in ("cutoff", "window"):
            per_q = defaultdict(list)
            for r in P.TRAIN:
                if which == "cutoff" and r["back"]: continue
                per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, P.corrected(r, names, bb)), r["outcome"]) - stack.brier(stack.as_probs(r, r["f"]), r["outcome"]))
            dd = defaultdict(lambda: defaultdict(list))
            for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
            per[which] = sum(sum(sum(x) / len(x) for x in dd[c].values()) / len(dd[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        print(f"   x{scale}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) | train cutoff {per['cutoff']:+.4f} window {per['window']:+.4f}", flush=True)
