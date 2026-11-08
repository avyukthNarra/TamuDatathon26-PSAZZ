"""Score v13 variants on both checks (run from mainFiles/ after research/v7c_cache.py)."""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import size_hmm_eval as SE            # v11 win, size-HMM splits SV / ST on cached rows
P, stack, lg, v11 = SE.P, SE.stack, SE.lg, SE.v11
from slot_forecasting.combo import V7C
cache = Path("runs/research-cache/v7c_val.json")
if cache.exists():
    V7V = json.load(open(cache))
else:
    V7V = {}
    for scored, other in zip(stack.HALVES, reversed(stack.HALVES)):
        m = V7C(Path("public"), True, prior_episodes=other, log=lambda *_: None)
        V7V.update({k: list(v) for k, v in m.predict_ids(Path("public"), "validation", {r["qid"] for r in P.VAL if r["episode"] in scored}).items()})
    json.dump(V7V, open(cache, "w"))
V7T = json.load(open("runs/research-cache/v7c_train.json"))
for r in P.VAL: r["v7c"] = V7V[r["qid"]]
for r in P.TRAIN: r["v7c"] = V7T[f"{r['qid']}|{r['back']}"]
def flat(r):
    sh = r["x"]["peers 1-5"]; return 0.0 if sh is None else sh - 0.4187
def make(ww=0.5, sw=1.0, extra_peer=0.0, hmm_size=True, own_sizes=None):
    def f(r, sizes_ours):
        a, b = v11(r), 1 - r["v7c"][0]
        win = 1 / (1 + math.exp(-((1 - ww) * lg(a) + ww * lg(b) + extra_peer * flat(r))))
        sb = [x / b for x in r["v7c"][1:]]
        so = sizes_ours if sizes_ours is not None else [x / (1 - r["mixer"][0]) for x in r["mixer"][1:]]
        size = [(1 - sw) * x + sw * y for x, y in zip(so, sb)]
        raw = [max(x, 1e-4) for x in [1 - win] + [win * s for s in size]]; t = sum(raw)
        return [x / t for x in raw]
    return f
BASE = make()
def evaluate(label, fn):
    preds = {r["qid"]: fn(r, SE.SV.get(id(r))) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0) for e in (stack.ALL_EP, *stack.HALVES)]
    out = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in P.TRAIN:
            if which == "cutoff" and r["back"]: continue
            so = SE.ST.get(id(r))
            per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(fn(r, so), r["outcome"]) - stack.brier(BASE(r, so), r["outcome"]))
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        def score(sel=None):
            return sum(sum(sum(dd[c][x]) / len(dd[c][x]) for x in (dd[c] if sel is None else sel[c])) / len(dd[c] if sel is None else sel[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
        rng = random.Random(0); b = sorted(score({c: [rng.choice(list(e)) for _ in e] for c, e in dd.items()}) for _ in range(1000))
        out[which] = (score(), b[25], b[974])
    bc = s[0]["by_condition"]
    print(f"{label:44s} val {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) " + " ".join(f"{c[:5]} {bc[c]['loss_ratio']:.3f}" for c in sorted(bc)) + f" | train vs v13 cutoff {out['cutoff'][0]:+.4f} [{out['cutoff'][1]:+.4f},{out['cutoff'][2]:+.4f}] window {out['window'][0]:+.4f} [{out['window'][1]:+.4f},{out['window'][2]:+.4f}]", flush=True)
if __name__ == "__main__":
    for label, fn in json.loads(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "--methods" else "[]") or []:
        pass
