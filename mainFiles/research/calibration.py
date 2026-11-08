import sys, math
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import residuals2 as R2
bins = [0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.01]
for name, rows, y in (("validation (truth)", R2.VAL, lambda r: 1 - r["truth"][0]), ("train (outcomes)", R2.TRAIN, lambda r: 1.0 if r["outcome"] else 0.0)):
    print(f"\n{name}: mean forecast -> mean truth/outcome per forecast bin (v10 forecast)")
    for lo, hi in zip(bins, bins[1:]):
        g = [r for r in rows if lo <= r["f2"] < hi]
        if len(g) < 10: continue
        mf, my = sum(r["f2"] for r in g) / len(g), sum(y(r) for r in g) / len(g)
        se = math.sqrt(max(my * (1 - my), 1e-4) / len(g)) if "train" in name else math.sqrt(sum((y(r) - my) ** 2 for r in g) / len(g) / len(g))
        print(f"  {lo:.2f}-{hi:.2f}: n {len(g):5d} forecast {mf:.3f} actual {my:.3f} diff {my - mf:+.3f} (z {(my - mf) / se:+.1f})")
