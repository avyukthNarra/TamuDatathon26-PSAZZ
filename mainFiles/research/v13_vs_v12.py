import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import v13_eval as E
E.evaluate("v12 (our win, size-rule HMM sizes)", E.make(ww=0.0, sw=0.0))
E.evaluate("v7c alone", E.make(ww=1.0, sw=1.0))
E.evaluate("candidate v14: peers +0.5, win 0.4 on v7c", E.make(ww=0.4, extra_peer=0.5))
