import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import v13_eval as E
E.evaluate("v13 (win 50/50, v7c sizes)", E.make())
for ww in (0.3, 0.4, 0.6, 0.7):
    E.evaluate(f"win weight on v7c {ww}", E.make(ww=ww))
for sw in (0.7, 0.85):
    E.evaluate(f"sizes: {sw} v7c + {1-sw:.2f} size-rule HMM", E.make(sw=sw))
for c in (0.25, 0.5):
    E.evaluate(f"extra peer term {c} (v7c has none)", E.make(extra_peer=c))
E.evaluate("extra peer 0.5 + win 0.4 on v7c", E.make(ww=0.4, extra_peer=0.5))
