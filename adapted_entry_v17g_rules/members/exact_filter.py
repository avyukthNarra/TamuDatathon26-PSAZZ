"""Exact filter win probability.

The blend uses only P(win). Level grids are fit on train. When validation
targets exist, the logit-normal level prior is replaced by the same two-means
fit used in orch/priors/levels_from_truth.py (rounded to 3 decimals).

crossfit: each alphabetical episode half is predicted with the other half's
level prior. submission: one prior from every validation target that is present.
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

ENTRY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENTRY))
sys.path.insert(0, str(ENTRY / "vendor" / "exact"))

from blend import write_predictions  # noqa: E402
from members.hooks import PAIRS_OVERRIDE  # noqa: E402
from model import ExactModel, PAIRS, TYPES, fit_levels, fit_schedule, load_machines  # noqa: E402


def resolve_data(path):
    data = Path(path)
    if (data / "public" / "train").is_dir():
        return data / "public"
    return data


def apply_pairs():
    import model
    if PAIRS_OVERRIDE:
        model.PAIRS = np.array(PAIRS_OVERRIDE, dtype=int)
        # Rebind the name this module imported.
        global PAIRS
        PAIRS = model.PAIRS


def alpha_halves(machines, conditions):
    episode_condition = {}
    for machine in machines:
        episode_condition[machine.episode_id] = conditions[machine.query_id]
    first, second = set(), set()
    for condition in sorted(set(episode_condition.values())):
        episodes = sorted(episode for episode, name in episode_condition.items() if name == condition)
        for index, episode in enumerate(episodes):
            (first if index % 2 == 0 else second).add(episode)
    return first, second


def read_pwin(path):
    out = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            out[row["query_id"]] = 1.0 - float(row["p_0"])
    return out


def level_prior(machines, pwin, episode_ids=None):
    """Two-means logit regression, matching levels_from_truth.py including rounding."""
    out = {}
    for kind in TYPES:
        chosen = [
            machine for machine in machines
            if machine.kind == kind
            and machine.query_id in pwin
            and (episode_ids is None or machine.episode_id in episode_ids)
        ]
        if len(chosen) < 4:
            raise ValueError(f"{kind}: not enough validation targets ({len(chosen)}) for a level prior")
        z = np.array([
            np.log(np.clip(pwin[machine.query_id], 1e-6, 1 - 1e-6)
                   / (1 - np.clip(pwin[machine.query_id], 1e-6, 1 - 1e-6)))
            for machine in chosen
        ])
        signal = np.array([int(machine.signal) for machine in chosen], dtype=float)
        centers = np.array([np.quantile(z, 0.2), np.quantile(z, 0.8)])
        for _ in range(50):
            labels = np.abs(z[:, None] - centers[None]).argmin(1)
            centers = np.array([z[labels == k].mean() for k in range(2)])
        if centers[0] > centers[1]:
            centers = centers[::-1]
            labels = 1 - labels
        spec = {}
        for cluster, name in enumerate(("dead", "live")):
            mask = labels == cluster
            design = np.column_stack([np.ones(int(mask.sum())), signal[mask]])
            coef = np.linalg.lstsq(design, z[mask], rcond=None)[0]
            residual = z[mask] - design @ coef
            ddof = 2 if int(mask.sum()) > 2 else 1
            spec[name] = {
                "a": round(float(coef[0]), 3),
                "b": round(float(coef[1]), 3),
                "sd": round(float(residual.std(ddof=ddof)), 3),
            }
        out[kind] = spec
    return out


def apply_level_prior(params, prior):
    updated = copy.deepcopy(params)
    for kind in TYPES:
        row = prior[kind]
        par = updated["levels"][kind]
        par["mean"] = [row["dead"]["a"], row["live"]["a"]]
        par["slope"] = [row["dead"]["b"], row["live"]["b"]]
        corr = np.array(par["cov"])
        rho = corr[0, 1] / np.sqrt(corr[0, 0] * corr[1, 1])
        sd0, sd1 = row["dead"]["sd"], row["live"]["sd"]
        par["cov"] = [[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]]
    return updated


def train_params(data):
    import model

    apply_pairs()
    machines, conditions, *_ = load_machines(data, "train", labels=True)
    if not PAIRS_OVERRIDE:
        model.PAIRS = model.learn_pairs(machines)
    pairs = [(int(a), int(b)) for a, b in model.PAIRS]
    levels, pair_prior = fit_levels(machines, conditions)
    extension = model.learn_extensions(machines)
    schedule = fit_schedule(machines, conditions, extra_rules=extension["extra_rules"])
    return {
        "levels": levels,
        "pair_prior": pair_prior.tolist(),
        "schedule": schedule,
        "condition_prior": [1 / 3, 1 / 3, 1 / 3],
        "pairs": pairs,
        "family_mass": extension["family_mass"],
        "extra_rules": extension["extra_rules"],
    }


def as_rows(win):
    rows = {}
    for query_id, p_win in win.items():
        p_win = float(p_win)
        rows[query_id] = [1.0 - p_win, *(p_win * 0.25 for _ in range(4))]
    return rows


def predict_machines(params, machines):
    model = ExactModel(params)
    win, _ = model.predict_win(machines, verbose=False)
    return win


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    args = parser.parse_args()
    data = resolve_data(args.data)
    params = train_params(data)
    print("exact: train levels and schedule fit", flush=True)
    if args.protocol == "crossfit":
        machines, conditions, *_ = load_machines(data, "validation")
        pwin = read_pwin(data / "validation" / "targets.csv")
        first, second = alpha_halves(machines, conditions)
        win = {}
        for scored, prior_ids in ((first, second), (second, first)):
            fold_params = apply_level_prior(params, level_prior(machines, pwin, prior_ids))
            fold_machines = [machine for machine in machines if machine.episode_id in scored]
            win.update(predict_machines(fold_params, fold_machines))
            print(f"exact crossfit half done, {len(win)} queries", flush=True)
    else:
        targets = data / "validation" / "targets.csv"
        predict_on, *_ = load_machines(data, args.split)
        if targets.exists():
            reference_machines, *_ = load_machines(data, "validation")
            params = apply_level_prior(params, level_prior(reference_machines, read_pwin(targets)))
            print("exact: level prior from validation targets", flush=True)
        else:
            print("exact: no validation targets, using the train level prior", flush=True)
        win = predict_machines(params, predict_on)
    rows = as_rows(win)
    write_predictions(args.out, rows)
    print(f"exact wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
