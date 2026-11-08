import sys, math
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import peer_correction as P
stack = P.stack
names = ["peers 1-5"]
# 1) the strength each split prefers, in log-odds and in probability space
for b in (0.0, 0.25, 0.5, 0.75, 1.0, 1.75):
    s = P.val_score(names, [b])
    print(f"log-odds coef {b:4}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) " + " ".join(f"{c[:5]} {s[0]['by_condition'][c]['loss_ratio']:.3f}" for c in sorted(s[0]['by_condition'])))
# 2) validation residual vs peer share BY CONDITION and by forecast level
for cond in ("stable", "short_history", "drift", "shared_change"):
    v = P.R.slope(P.VAL, lambda r: P.share(r, range(1, 6)), lambda r: r["condition"] == cond)
    t = P.R.slope(P.TRAIN, lambda r: P.share(r, range(1, 6)), lambda r: r["condition"] == cond)
    print(f"{cond:14s} slope validation {v[0]:+.3f} (z {v[0]/v[1]:+.1f}) | train {t[0]:+.3f} (z {t[0]/t[1]:+.1f})")
for lo, hi in ((0, .2), (.2, .8), (.8, 1.01)):
    v = P.R.slope(P.VAL, lambda r: P.share(r, range(1, 6)), lambda r: lo <= r["f"] < hi)
    t = P.R.slope(P.TRAIN, lambda r: P.share(r, range(1, 6)), lambda r: lo <= r["f"] < hi)
    print(f"forecast {lo}-{hi}: slope validation {v[0]:+.3f} (z {v[0]/v[1]:+.1f}) | train {t[0]:+.3f} (z {t[0]/t[1]:+.1f})")
print("mean peers 1-5 share: validation %.3f train %.3f" % (sum(r['x']['peers 1-5'] or 0 for r in P.VAL) / len(P.VAL), P.MEANS['peers 1-5']))
