"""Score candidate models on the kit's fixed TUNING episodes only (audit stays fresh).

Usage:
    python exp_eval.py e1_default                 # tuning only
    python exp_eval.py e1_default --audit         # ALSO score audit: log this in _coord/RESULTS.md

Candidates are registered in CANDIDATES below as name -> zero-arg factory.
Uses the same split (`episode_folds`) and scorer (`subset_score`) as the kit's
`select_model`, so numbers are comparable with run_experiments.py.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.pair_logistic import PairLogisticForecaster
from slot_forecasting.rounds import round_metadata

CANDIDATES = {
    "pooled_recent": lambda: FrequencyForecaster("pooled_recent"),
    "e1_default": lambda: PairLogisticForecaster(),
    "e1_nopairs": lambda: PairLogisticForecaster(pairs=False),
    "e1_hl16": lambda: PairLogisticForecaster(half_life=16),
    "e1_hl64": lambda: PairLogisticForecaster(half_life=64),
    "e1_C1": lambda: PairLogisticForecaster(C=1.0),
    "e1_C10": lambda: PairLogisticForecaster(C=10.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "public"))
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--json", help="write per-candidate results here")
    args = ap.parse_args()
    public = Path(args.data)
    max_points = round_metadata(public)["max_points"]
    tq, th = load_inputs(public / "train")
    vq, vh = load_inputs(public / "validation")
    labels = read_labels(public / "train/labels.csv", tq)
    targets = read_targets(public / "validation/targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(public / "validation/conditions.csv", ("query_id", "condition"))}
    reference = read_submission(public / "validation/reference.csv")
    tuning, audit = episode_folds(targets, groups)
    results = {}
    for name in args.names:
        start = perf_counter()
        model = CANDIDATES[name]().fit(tq, th, labels)
        pred = model.predict(vq, vh)
        row = {"tuning": subset_score(pred, targets, groups, reference, tuning, max_points)}
        if args.audit:
            row["audit"] = subset_score(pred, targets, groups, reference, audit, max_points)
        row["seconds"] = round(perf_counter() - start, 1)
        results[name] = row
        for split in ("tuning", "audit"):
            if split in row:
                r = row[split]
                conds = " ".join(f"{c}={v['loss_ratio']:.3f}" for c, v in r["by_condition"].items())
                print(f"{name:24s} {split:6s} points={r['technical_points']:.3f}  {conds}  ({row['seconds']}s)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
