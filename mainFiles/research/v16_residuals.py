import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import flip_transfer as FT
    import v14_residuals as R
V16 = FT.v(1.0, 0.3)
for split, rows in (("validation", FT.E.P.VAL), ("train", FT.E.P.TRAIN)):
    for r in rows:
        p = V16(r, None); w = 1 - p[0]; pbig = (p[3] + p[4]) / w
        if split == "validation":
            t = r["truth"]; r["wres"] = (1 - t[0]) - w
            r["sres"] = (t[3] + t[4]) / (1 - t[0]) - pbig if 1 - t[0] > 0.05 else None
        else:
            r["wres"] = (1.0 if r["outcome"] else 0.0) - w
            r["sres"] = ((1.0 if r["outcome"] >= 3 else 0.0) - pbig) if r["outcome"] > 0 else None
        r["flip_on"] = 1.0 if abs(r["flip_d"]) > 1e-9 else 0.0
R.SIG["flip swap active"] = lambda r: r["flip_on"]
R.SIG["flip effect size"] = lambda r: r["flip_d"]
for key, label in (("wres", "WIN"), ("sres", "SIZE")):
    print(f"=== v16 {label}: |z| > 2 on both splits, same sign")
    for name, fn in R.SIG.items():
        v, t = R.slope(FT.E.P.VAL, fn, key), R.slope(FT.E.P.TRAIN, fn, key)
        if v and t and v[0] * t[0] > 0 and abs(v[1]) > 2 and abs(t[1]) > 2:
            print(f"  {name:26s} validation {v[0]:+.4f} (z {v[1]:+.1f}) | train {t[0]:+.4f} (z {t[1]:+.1f})")
