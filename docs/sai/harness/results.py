"""Result writer (schema in MASTER_TEST_PLAN section 2.4), determinism check, disk cache."""
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np

from .data import CACHE, PC
from .scoring import evaluate

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
PREDS = HERE.parent / "preds"


def cached(key, builder):
    """Compute once, reuse from docs/sai/cache/ (gitignored). Key must encode everything that changes the value."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    value = builder()
    path.write_bytes(pickle.dumps(value, protocol=4))
    return value


def to_csv(path, split, P):
    P = np.asarray(P, float)
    P = P / P.sum(1, keepdims=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id," + ",".join(PC) + "\n")
        for q, row in zip(split.qid, P):
            f.write(q + "," + ",".join(f"{v:.17g}" for v in row) + "\n")


def read_pred_csv(path, split):
    import csv
    pos = {q: i for i, q in enumerate(split.qid)}
    P = np.full((split.n, 5), np.nan)
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            P[pos[r["query_id"]]] = [float(r[k]) for k in PC]
    if np.isnan(P).any():
        raise ValueError("prediction CSV missing queries")
    return P


def check_determinism(fn, split, tol=1e-8):
    """Run twice; return (max abs diff, first output)."""
    a, b = np.asarray(fn(split), float), np.asarray(fn(split), float)
    return float(np.abs(a - b).max()), a


def write_result(mid, owner, description, params, split, P, runtime, seed=0, stdlib_only=False,
                 determinism=None, scope="dev", extra=None, bootstrap=1000):
    ev = evaluate(P, split, scope=scope, who=mid if scope != "dev" else None, bootstrap=bootstrap)
    doc = {"id": mid, "owner": owner, "description": description, "params": params,
           "scope": scope, "points": ev["points"], "ci95": ev.get("ci95"),
           "condition_ratios": ev["condition_ratios"], "R": ev["R"],
           "class_error_shares": ev["class_error_shares"], "runtime_seconds": runtime,
           "seed": seed, "stdlib_only": stdlib_only, "determinism_max_diff": determinism,
           "python": platform.python_version(), "n_episodes": ev["n_episodes"]}
    if extra:
        doc["extra"] = extra
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{mid}.json").write_text(json.dumps(doc, indent=1))
    return doc


def run_method(mid, owner, description, params, predict_fn, split, seed=0, stdlib_only=False,
               check_det=True, scope="dev"):
    """predict_fn(split) -> (n,5). Times the first run, checks determinism with a second, saves
    results/<id>.json and preds/<id>_validation.csv (all validation queries, submission format)."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(k) != "1":
            raise RuntimeError(f"export {k}=1 before running (shared-laptop guardrail)")
    t0 = time.perf_counter()
    P = np.asarray(predict_fn(split), float)
    runtime = time.perf_counter() - t0
    det = None
    if check_det:
        det = float(np.abs(P - np.asarray(predict_fn(split), float)).max())
        if det > 1e-8:
            print(f"WARNING {mid}: determinism diff {det:.2e} > 1e-8")
    to_csv(PREDS / f"{mid}_validation.csv", split, P)
    doc = write_result(mid, owner, description, params, split, P, runtime, seed, stdlib_only, det, scope)
    ci = doc["ci95"]
    print(f"{mid}: {doc['points']:.3f} pts" + (f" CI [{ci[0]:.2f},{ci[1]:.2f}]" if ci else "")
          + "  ratios " + " ".join(f"{k[:5]}={v:.3f}" for k, v in doc["condition_ratios"].items())
          + f"  {runtime:.1f}s det={det}")
    return doc, P
