"""Train-only sysid win model.

Fits the five shared-effect win models, picks one by the training BIC, then
picks a reversal width by training-history marginal likelihood. That is the
selection behind sysid_train_only.csv. The blend reads only P(win), so the
separate size catalog is not fit.

This member always learns its rules from train. PAIRS_OVERRIDE is ignored;
replace this script to drop in another rule library.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import tempfile
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENTRY))
sys.path.insert(0, str(ENTRY / "vendor" / "sysid"))

import numpy as np  # noqa: E402
from blend import write_predictions  # noqa: E402
from slot_forecasting.io import read_labels  # noqa: E402
from sysid_bayes import complete_data, condition_map, train_win  # noqa: E402
from sysid_dynamic import dynamic_predict  # noqa: E402
from sysid_study import load_machines  # noqa: E402

WIN_MODELS = (
    ("no_common", False),
    ("basic", True),
    ("peers_only", "peers_only"),
    ("all_peers", "all_peers"),
    ("strongest_peer", "strongest_peer"),
)
REGIMES = (
    ("dyn0", 0.0, 0.0),
    ("dyn01", 0.1, 0.1),
    ("dyn02", 0.2, 0.2),
    ("dyn01shared0", 0.1, 0.0),
)


def resolve_data(path):
    data = Path(path)
    if (data / "public" / "train").is_dir():
        return data / "public"
    return data


def fit_choice(data, work):
    train, queries, _ = load_machines(data, "train")
    labels = read_labels(data / "train" / "labels.csv", queries)
    conditions = condition_map(data, "train")
    models = {}
    for name, shared in WIN_MODELS:
        models[name] = train_win(train, labels, conditions, work, shared_features=shared)
        print("sysid fit", name, models[name]["stable_train_log_marginal"], flush=True)
    n_win = sum(len(complete_data(machine, labels)[1]) for machine in train
                if conditions[machine.query.query_id] == "stable")
    win_bic = {
        name: -2 * model["stable_train_log_marginal"]
        + (6 + len(model["commonids"]) + len(model["ruleids"]) - 1) * np.log(n_win)
        for name, model in models.items()
    }
    selected_win = min(win_bic, key=win_bic.get)
    likelihood = {}
    for name, width, shared_width in REGIMES:
        _, reports = dynamic_predict(models[selected_win], train, width=width, shared_width=shared_width)
        likelihood[name] = sum(row["log_evidence"] for row in reports.values())
        print("sysid regime", name, likelihood[name], flush=True)
    selected_regime = max(likelihood, key=lambda name: likelihood[name])
    width, shared_width = {name: (width, shared) for name, width, shared in REGIMES}[selected_regime]
    print(f"sysid selected {selected_win}/{selected_regime}", flush=True)
    return models[selected_win], width, shared_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    args = parser.parse_args()
    data = resolve_data(args.data)
    split = "validation" if args.protocol == "crossfit" else args.split
    with tempfile.TemporaryDirectory(prefix="sysid-") as work:
        model, width, shared_width = fit_choice(data, Path(work))
        machines, _, _ = load_machines(data, split)
        win, _ = dynamic_predict(model, machines, width=width, shared_width=shared_width)
    rows = {}
    for query_id, p_win in win.items():
        p_win = float(p_win)
        rows[query_id] = [1.0 - p_win, *(p_win * 0.25 for _ in range(4))]
    write_predictions(args.out, rows)
    print(f"sysid wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
