import sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v13_eval as E
from slot_forecasting.size_rule_hmm import modes_from_targets
from slot_forecasting.io import load_inputs, read_targets
vq, _ = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
modes = modes_from_targets(vt, vq)
for cond in ("stable", "short_history", "drift", "shared_change"):
    rows = [r for r in E.P.VAL if r["condition"] == cond and 1 - r["truth"][0] > 0.05 and r["type"] in modes]
    wrong = flip_like = 0; errs = []
    for r in rows:
        A, B = modes[r["type"]]
        t = [x / (1 - r["truth"][0]) for x in r["truth"][1:]]
        tb = 1.0 if sum((a - b) ** 2 for a, b in zip(t, B)) < sum((a - b) ** 2 for a, b in zip(t, A)) else 0.0
        p = [x / (1 - r["v7c"][0]) for x in r["v7c"][1:]]
        pb = min(max(sum((x - a) * (b - a) for x, a, b in zip(p, A, B)) / sum((b - a) ** 2 for a, b in zip(A, B)), 0), 1)
        errs.append(abs(pb - tb))
        wrong += abs(pb - tb) > 0.5
    errs.sort()
    print(f"{cond:14s} n {len(rows):3d} | v7c P(big mode) vs truth: mean |err| {sum(errs)/len(errs):.3f}, confidently wrong (>0.5) {wrong} ({wrong/len(rows):.1%}), median {errs[len(errs)//2]:.3f}")
by_ep = defaultdict(list)
for r in E.P.VAL:
    if r["condition"] == "shared_change" and 1 - r["truth"][0] > 0.05 and r["type"] in modes:
        A, B = modes[r["type"]]
        t = [x / (1 - r["truth"][0]) for x in r["truth"][1:]]
        tb = 1.0 if sum((a - b) ** 2 for a, b in zip(t, B)) < sum((a - b) ** 2 for a, b in zip(t, A)) else 0.0
        p = [x / (1 - r["v7c"][0]) for x in r["v7c"][1:]]
        pb = min(max(sum((x - a) * (b - a) for x, a, b in zip(p, A, B)) / sum((b - a) ** 2 for a, b in zip(A, B)), 0), 1)
        by_ep[r["episode"]].append(abs(pb - tb) > 0.5)
bad = sorted(((sum(v), len(v)) for v in by_ep.values()), reverse=True)
print("shared_change episodes: confidently-wrong machines per episode (worst first):", bad[:12])
