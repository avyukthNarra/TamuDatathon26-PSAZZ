"""Write results/B0.json and B3.json (dev scope) plus their validation prediction CSVs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_split, run_method
from harness.baselines import pooled, b3_prototype

v = load_split("validation")
run_method("B0", "claude", "starter pooled (strength 20), frozen reference", {"method": "pooled"}, pooled, v, stdlib_only=True)
run_method("B3", "claude", "pair-model prototype (lags 1-5,7 + pairs, ridge 20, holdout half-life)",
           {"lags": [1, 2, 3, 4, 5, 7], "pairs": [[1, 3], [2, 5], [4, 7]], "lam": 20}, b3_prototype, v)
