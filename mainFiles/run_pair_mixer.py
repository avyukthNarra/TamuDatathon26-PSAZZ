"""Score the PairMixer on validation, write a test submission, and build a verified entry ZIP.

Usage (from mainFiles/):  python run_pair_mixer.py --out runs/pair-mixer
Writes to <out>/: validation_predictions.csv, submission.csv, validation_score.json,
entry/ (README.md, reproduce.json, predict.py, slot_forecasting/), entry.zip.
Then re-runs entry/predict.py from a fresh process and checks every probability
matches submission.csv within 1e-8 (the organizers' reproduction tolerance).
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from slot_forecasting.contest import score_contest_files
from slot_forecasting.entry_config import USE_FLIP, USE_ROUND2, USE_SIZE, USE_SIZE_GATE
from slot_forecasting.forecast import fit_flip, fit_size, predict_split
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_json, write_submission
from slot_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_by_type_from_targets
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.starter import public_directory
from slot_forecasting.submission_contract import package_code

HERE = Path(__file__).resolve().parent
MODULES = ("__init__.py", "schema.py", "io.py", "pair_mixer.py", "size_rule.py",
           "size_gate.py", "cell_mode_gate.py", "episode_flip.py", "forecast.py", "entry_config.py")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--out", default="runs/pair-mixer")
parser.add_argument("--cv-val", action="store_true", help="also score validation halves with opposite-half priors")
parser.add_argument("--score-only", action="store_true", help="skip entry zip and the fresh-process rerun")
parser.add_argument("--size", action=argparse.BooleanOptionalAction, default=None,
                    help="train-learned size atoms (default: entry_config.USE_SIZE)")
parser.add_argument("--size-gate", action=argparse.BooleanOptionalAction, default=None,
                    help="train-only gate against the v6 size head (default: entry_config.USE_SIZE_GATE)")
parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=None,
                    help="episode parity swap (default: entry_config.USE_FLIP)")
parser.add_argument("--round2", action=argparse.BooleanOptionalAction, default=None,
                    help="windowed pairs and 2-means prior (default: entry_config.USE_ROUND2)")
args = parser.parse_args()
use_size = USE_SIZE if args.size is None else args.size
use_size_gate = USE_SIZE_GATE if args.size_gate is None else args.size_gate
use_flip = USE_FLIP if args.flip is None else args.flip
use_round2 = USE_ROUND2 if args.round2 is None else args.round2
print(f"features: size={use_size} size_gate={use_size_gate} flip={use_flip} round2={use_round2}", flush=True)

started = perf_counter()
public = public_directory(args.data)
out = Path(args.out).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
train_queries, train_history = load_inputs(public / "train")
val_targets = read_targets(public / "validation/targets.csv")
val_queries, val_history = load_inputs(public / "validation")
train_labels = read_labels(public / "train/labels.csv", train_queries)
size_modes = size_modes_from_targets(val_targets, val_queries)
win_prior = win_prior_by_type_from_targets(val_targets, val_queries, two_means=use_round2)
model = adapt(train_queries, train_history, train_labels, size_modes=size_modes, win_prior=win_prior,
              round2=use_round2)
cell_mode_gate = getattr(model, "cell_mode_gate", None)
if cell_mode_gate is not None:
    write_json(out / "cell_mode_gate.json", cell_mode_gate)
    print(f"cell-mode train gate: {cell_mode_gate}", flush=True)
size_rule = fit_size(train_queries, train_history, use_size, train_labels, use_size_gate)
gate = fit_flip(model, train_queries, train_history, use_flip)
if size_rule is not None:
    print(f"size lags: {size_rule.lags}  train EM loglik {size_rule.loglik:.2f}")
    print(f"size head: {size_rule.gate_choice}  static prior {size_rule.static_prior:.4f}")
    if size_rule.gate_diagnostics is not None:
        diag = size_rule.gate_diagnostics
        print(f"size gate: episodes={diag['n_episodes']} wins={diag['n_wins']} "
              f"mean dBrier={diag['mean_delta_brier_candidate_minus_v6']} "
              f"upper95={diag['upper_ci']}")
    for lag in range(1, 9):
        print(f"  lag {lag} posterior mass {size_rule.mass[lag]:.4f}")
    write_json(out / "size_rule.json", {
        "lags": list(size_rule.lags),
        "mass_share": {str(lag): size_rule.mass[lag] for lag in range(1, 9)},
        "atoms": {kind: [list(atom) for atom in pair] for kind, pair in sorted(size_rule.atoms.items())},
        "static_prior": size_rule.static_prior,
        "pair_candidates": [list(pair) for pair in size_rule.pair_candidates],
        "gate_choice": size_rule.gate_choice,
        "gate_diagnostics": size_rule.gate_diagnostics,
        "train_loglik": size_rule.loglik,
        "em_restarts": [{"seed": seed, "train_loglik": total} for seed, total in size_rule.restarts],
    })
if gate is not None:
    if gate.left is None:
        print("flip gate: no train gap; no episode is swapped")
    else:
        print(f"flip gate {gate.threshold:.4f}  gap {gate.left:.2f} .. {gate.right:.2f}  width {gate.width:.2f}")
files = {"validation": out / "validation_predictions.csv", "test": out / "submission.csv"}
split_detect = {}
for split, path in files.items():
    queries, history = load_inputs(public / split)
    swaps = None
    if gate is not None:
        swaps, detect = gate.swaps(queries, history, model.pairs)
        split_detect[split] = {"n_episodes": len(detect), "n_swapped": len(swaps),
                               "swapped": {eid: tau for eid, tau in sorted(swaps.items())}}
        print(f"{split}: swapped {len(swaps)} / {len(detect)} episodes", flush=True)
    write_submission(path, predict_split(model, queries, history, size_rule, gate, swaps=swaps))
if gate is not None:
    write_json(out / "flip_gate.json", {
        "threshold": gate.threshold, "width": gate.width, "left": gate.left, "right": gate.right,
        "splits": split_detect,
    })

# --- validation: official contest score + win-chance vs payout-size error split ---
event = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
val = public / "validation"
report = score_contest_files(files["validation"], val / "targets.csv", val / "queries.csv",
                             val / "conditions.csv", val / "reference.csv", **event)
predictions = read_submission(files["validation"])
conditions = dict(line.strip().split(",") for line in (val / "conditions.csv").read_text().splitlines()[1:])
split_error = defaultdict(lambda: [0.0, 0.0, 0])
for target in read_targets(val / "targets.csv"):
    p, q = predictions[target.query_id], target.probabilities
    p_win, q_win = 1 - p[0], 1 - q[0]
    # error left if the win chance were exactly right (keeps our size split) = size part
    fixed_win = [q[0]] + [x / p_win * q_win for x in p[1:]]
    total = sum((a - b) ** 2 for a, b in zip(p, q))
    size_part = sum((a - b) ** 2 for a, b in zip(fixed_win, q))
    row = split_error[conditions[target.query_id]]
    row[0] += total - size_part
    row[1] += size_part
    row[2] += 1
write_json(out / "validation_score.json", report)
print(f"{'condition':<15}{'error':>8}{'baseline':>10}{'ratio':>8}{'win part':>10}{'size part':>11}")
for name, row in report["by_condition"].items():
    win, size, n = split_error[name]
    print(f"{name:<15}{row['error']:>8.4f}{row['reference_error']:>10.4f}{row['loss_ratio']:>8.3f}"
          f"{win / n:>10.4f}{size / n:>11.4f}")
print(f"R = {report['mean_loss_ratio']:.4f}  ->  {report['technical_points']:.2f} / {report['max_points']:g} points")
if args.cv_val:
    from slot_forecasting.experiments import episode_folds, subset_score
    from slot_forecasting.io import read_rows
    val_groups = {r["query_id"]: r["condition"] for r in read_rows(val / "conditions.csv", ("query_id", "condition"))}
    tuning, audit = episode_folds(val_targets, val_groups)
    crossfit = {}
    for scored, prior_episode_ids in ((tuning, audit), (audit, tuning)):
        prior_targets = tuple(t for t in val_targets if t.episode_id in prior_episode_ids)
        fold_model = adapt(train_queries, train_history, train_labels,
                           size_modes=size_modes_from_targets(prior_targets, val_queries),
                           win_prior=win_prior_by_type_from_targets(prior_targets, val_queries, two_means=use_round2),
                           round2=use_round2, log=lambda *_: None)
        fold_queries = tuple(q for q in val_queries if q.episode_id in scored)
        fold_keys = {(q.episode_id, q.machine_id) for q in fold_queries}
        fold_history = tuple(p for p in val_history if (p.episode_id, p.machine_id) in fold_keys)
        fold_swaps = None
        fold_gate = gate
        if use_flip and fold_model.pairs != model.pairs:
            fold_gate = fit_flip(fold_model, train_queries, train_history, True)
            fold_swaps, _detect = fold_gate.swaps(fold_queries, fold_history, fold_model.pairs)
        elif use_flip:
            fold_swaps = split_detect["validation"]["swapped"]
        crossfit.update(predict_split(fold_model, fold_queries, fold_history, size_rule, fold_gate,
                                      swaps=fold_swaps))
    write_submission(out / "crossfit_validation_predictions.csv", crossfit)
    halves = [subset_score(crossfit, val_targets, val_groups,
                           read_submission(val / "reference.csv"), episodes,
                           event["max_points"]) for episodes in (tuning, audit)]
    crossfit_report = {"tuning_using_audit_priors": halves[0], "audit_using_tuning_priors": halves[1]}
    write_json(out / "crossfit_validation_score.json", crossfit_report)
    print(f"Cross-fitted validation: tuning {halves[0]['technical_points']:.2f} / {event['max_points']:g}; "
          f"audit {halves[1]['technical_points']:.2f} / {event['max_points']:g}")
if args.score_only:
    print(f"score-only ({perf_counter() - started:.0f}s); entry zip skipped")
    raise SystemExit(0)

# --- entry folder + ZIP ---
entry = out / "entry"
if entry.exists():
    shutil.rmtree(entry)
(entry / "slot_forecasting").mkdir(parents=True)
for name in MODULES:
    shutil.copyfile(HERE / "slot_forecasting" / name, entry / "slot_forecasting" / name)
# The packaged switches are the ones this run scored, so the fresh rerun matches.
(entry / "slot_forecasting" / "entry_config.py").write_text(
    f"USE_SIZE = {use_size}\nUSE_SIZE_GATE = {use_size_gate}\n"
    f"USE_FLIP = {use_flip}\nUSE_ROUND2 = {use_round2}\n",
    encoding="utf-8")
for name in ("predict.py", "reproduce.json"):
    shutil.copyfile(HERE / "entry_template" / name, entry / name)
(entry / "priors.json").write_text(json.dumps({"size_modes": size_modes, "win_prior": win_prior}),
                                  encoding="utf-8")  # fallback if /data lacks validation targets
metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
sha = hashlib.sha256(files["test"].read_bytes()).hexdigest()
commit = "local build"
(entry / "README.md").write_text(f"""# House Edge entry: lag-pair mixer

- Round: {metadata.get('round', 'main')}
- Season ID: {metadata.get('season_id', 'see public/metadata.json')}
- Forecast SHA-256: {sha}
- Code version: {commit} (built by mainFiles/run_pair_mixer.py)
- Environment: `python` image, Python standard library only (tested on Python {platform.python_version()})
- Validation: {report['technical_points']:.2f} / {report['max_points']:g} points (R = {report['mean_loss_ratio']:.4f})

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

- What it does: on `/data/train` it rediscovers which lag pairs drive wins and
  picks match/differ (2-cell) vs 4-cell experts by Brier on `train/labels.csv`
  at the cutoff. The win probability is a per-machine online mixture of those
  lag-pair experts. Each expert tracks, with exponential forgetting, how often
  the machine won when its results at two earlier lags matched vs differed;
  experts are weighted by their recent squared error. The dead/live win-chance
  levels are read from `/data/validation/targets.csv` when that file is present
  (copy in `priors.json`). Given a win, the payout-size mix is a posterior
  mixture of two atoms per machine type. Both the atoms and the lag set are
  learned from `/data/train` histories at run time: EM for the atoms, then every
  lag in 1..8 whose share of per-machine posterior mass exceeds the uniform
  share (all eight lags if none do). A machine's own wins before the cutoff
  update its posterior over static and lag/polarity states. The candidate size
  head is enabled only when a two-fold, whole-episode cross-fit gate on train
  history beats the PairMixer's own size head by the fixed Brier criterion.
  The gate is recomputed for every supplied kit and uses no validation targets.
  A missing lag tick is marginalized over the atom states. Probabilities are
  floored at 1e-4 and renormalized. Only ticks before each cutoff are read.
- Model files: none. The command refits atoms, lags, and the pair mixture from
  `/data/train`. `priors.json` is only the win-chance fallback when validation
  targets are absent.
- Seeds/settings: deterministic. Atom EM uses fixed seeds 0..5. Pair-mixer
  settings are the `PairMixer` defaults in `slot_forecasting/pair_mixer.py`.
  This build: size={use_size}, size_gate={use_size_gate}, flip={use_flip}, round2={use_round2}.
  {f"Size lags kept: {size_rule.lags}; static prior: {size_rule.static_prior:.4f}; selected head: {size_rule.gate_choice}." if size_rule is not None else "Size candidate is disabled; the pair mixer's own size head is used."}
  {("Flip gate is the midpoint of the widest gap in the train log-Bayes-factor list; "
    "an episode above it exchanges agree and differ counts at that episode's MAP tick.")
    if use_flip else "No episode flip."}
  {("Round-two discovery: windowed pair votes, all 28 pairs if the search is empty, "
    "machine-level size key if no lag has positive gain, and 2-means levels for a "
    "type that does not split at 0.5.")
    if use_round2 else "Pair discovery is the v6 vote."}
- Runtime: about {perf_counter() - started:.0f} s for train + validation + test on one CPU core.
""", encoding="utf-8")
zip_path = out / "entry.zip"
if zip_path.exists():
    zip_path.unlink()
package_code(entry, zip_path)

# --- reproduction check from a fresh process, run from inside the entry folder ---
repro = out / "repro_check.csv"
subprocess.run([sys.executable, "predict.py", "--data", str(public), "--out", str(repro)],
               cwd=entry, check=True, capture_output=True)
original = read_submission(files["test"])
again = read_submission(repro)
worst = max(abs(a - b) for key in original for a, b in zip(original[key], again[key]))
if set(original) != set(again) or worst > 1e-8:
    raise SystemExit(f"Reproduction FAILED: max probability difference {worst:g}")
print(f"Entry: {zip_path}  (fresh rerun max diff {worst:g}; limit 1e-8)")
print(f"Submission: {files['test']}  sha256 {sha}  ({perf_counter() - started:.0f}s total)")
