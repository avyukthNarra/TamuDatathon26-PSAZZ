import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import continuous_scales as C
E, stack = C.E, C.stack
on, off = C.f_of(C.BASE), C.f_of(dict(C.BASE, peer=0.0))
for name, half in zip(("tuning", "audit"), stack.HALVES):
    err = lambda f: sum(sum((a - b) ** 2 for a, b in zip(f(r, None), r["truth"])) for r in E.P.VAL if r["episode"] in half) / sum(1 for r in E.P.VAL if r["episode"] in half)
    print(f"{name} half: error with extra peer {err(on):.6f} | without {err(off):.6f} -> this half would choose {'OFF' if err(off) < err(on) else 'ON'}")
E.BASE = on
E.evaluate("v18 settings (peer on)", on)
E.evaluate("extra peer off", off)
E.evaluate("extra peer 0.25 (middle)", C.f_of(dict(C.BASE, peer=0.25)))
