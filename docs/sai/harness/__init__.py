"""House Edge shared harness. Import this; do not reimplement scoring.

    from harness import load_split, evaluate, run_method, audit_split
"""
from .data import load_split, Split, PAY, PC, CONDITIONS, DATA
from .scoring import evaluate, diff_ci, class_shares, audit_episodes, smoke_split
from .stack import blend_gain
from .results import run_method, write_result, check_determinism, cached, to_csv, read_pred_csv
