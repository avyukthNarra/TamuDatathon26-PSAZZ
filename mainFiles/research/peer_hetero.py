import sys, math
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
def slope(rows, mask):
    xs, ys = [], []
    for r in rows:
        if not mask(r) or r["x"]["peers 1-5"] is None: continue
        xs.append(r["x"]["peers 1-5"]); ys.append(r["res"])
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n; sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    se = math.sqrt(sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys)) / (n - 2) / sxx)
    return f"{b:+.3f} (z {b/se:+.1f}, n {n})"
for label, mask in [("ALL", lambda r: True)] + [(c, lambda r, c=c: r["condition"] == c) for c in ("stable", "short_history", "drift", "shared_change")] \
        + [(t, lambda r, t=t: r["type"] == t) for t in ("steady", "balanced", "jackpot")] \
        + [("forecast < 0.2", lambda r: r["f"] < .2), ("forecast 0.2-0.8", lambda r: .2 <= r["f"] <= .8), ("forecast > 0.8", lambda r: r["f"] > .8)]:
    print(f"{label:18s} validation {slope(P.VAL, mask):28s} train {slope(P.TRAIN, mask)}")
