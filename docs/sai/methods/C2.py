"""C2 = composition, no new fitted parameters: win probability from M9b_B4 (peer-gated pooled changepoint memory on B4),
payout-size shares from C1 (A9 severity ridge + alpha-64 size prior). P = [1-pw, pw * shares]. Reads the saved dev
prediction CSVs of the two members (so it is exactly their chosen cells; selection optimism of both carries over)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_split, read_pred_csv, run_method

HERE = Path(__file__).resolve().parents[1]


def predict(split):
    pw_src = read_pred_csv(HERE / "preds/M9b_B4_validation.csv", split)
    sh_src = read_pred_csv(HERE / "preds/C1_validation.csv", split)
    pw = 1 - pw_src[:, 0]
    sh = sh_src[:, 1:] / sh_src[:, 1:].sum(1, keepdims=True)
    return np.column_stack([1 - pw, pw[:, None] * sh])


if __name__ == "__main__":
    v = load_split("validation")
    run_method("C2", "claude", "win prob from M9b_B4, size shares from C1 (A9 severity + alpha 64)",
               {"pw": "M9b_B4", "shares": "C1"}, predict, v)
