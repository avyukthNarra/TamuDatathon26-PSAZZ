import sys, math
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import residuals2 as R2
VAL, TRAIN = R2.VAL, R2.TRAIN
def idx(m): return int(m.rsplit("m", 1)[-1])
def make(off):
    def f(r):
        hist = R2.P.R.HIST["validation" if r in VAL_SET else "train"]
        n = 8
        other = f"{r['episode']}-m{(idx(r['machine']) + off) % n:03d}"
        s = hist.get((r["episode"], other))
        if not s: return None
        v = [s[r["cutoff"] - k] > 0 for k in range(1, 6) if r["cutoff"] - k in s]
        return sum(v) / len(v) if v else None
    return f
VAL_SET = set(map(id, VAL))
VAL_SET = type("S", (), {"__contains__": lambda self, r: id(r) in _ids})()
_ids = set(map(id, VAL))
print("machine ids look like:", VAL[0]["machine"])
for off in range(1, 8):
    v, t = R2.slope(VAL, make(off), "res"), R2.slope(TRAIN, make(off), "res")
    print(f"peer at id offset +{off}: validation {v[0]:+.3f} (z {v[1]:+.1f}) | train {t[0]:+.3f} (z {t[1]:+.1f})")
