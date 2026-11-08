import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import subcell as S
S.E.BASE = S.v()
for K, a in ((8, 1.0), (32, 1.0), (16, 0.75), (16, 1.25)):
    S.E.evaluate(f"jackpot sub-cell K {K} alpha {a}", S.v(a, K, {"jackpot"}))
