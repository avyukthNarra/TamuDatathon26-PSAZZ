"""Build v13 (slot_forecasting/combo.py: v12 + vendored v7c) and score it on validation, write a test submission, and build a verified entry ZIP.

Usage (from mainFiles/):  python run_strongest.py --out runs/strongest-v13 --cv-val
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
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_json, write_submission
from slot_forecasting.combo import V7C, combine, gate_terms
from slot_forecasting.ensemble import build_ensemble
from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.starter import public_directory
from slot_forecasting.submission_contract import package_code

HERE = Path(__file__).resolve().parent
MODULES = ("__init__.py", "schema.py", "io.py", "pair_mixer.py", "rule_hmm.py", "changepoint.py", "ensemble.py", "peers.py", "plateau.py", "size_rule_hmm.py", "strongest.py", "combo.py", "subcell.py")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE / "public"))
parser.add_argument("--out", default="runs/strongest-v18")
parser.add_argument("--cv-val", action="store_true", help="also score validation halves with opposite-half priors")
parser.add_argument("--win-weight", type=float, default=0.3, help="share of v7c's win chance (v13: 0.5, v15: 0.4)")
parser.add_argument("--flip-transfer", type=float, default=1.0, help="v7c flip-swap effect added to our win (v15: 0)")
parser.add_argument("--subcell", action=argparse.BooleanOptionalAction, default=True, help="v17 jackpot sub-cell correction")
parser.add_argument("--gates", action=argparse.BooleanOptionalAction, default=True,
                    help="v18: keep each add-on only if this round's train outcomes support it (re-run by the entry)")
parser.add_argument("--extra-peer", type=float, default=0.5, help="extra peer term on the blend (v13: 0)")
parser.add_argument("--signal-coef", type=float, default=0.05, help="log-odds per signal step (v14: 0)")
parser.add_argument("--variant", default="strongest", choices=("strongest",),
                    help="ensemble variant (slot_forecasting/ensemble.py); saved into the entry")
args = parser.parse_args()

started = perf_counter()
public = public_directory(args.data)
out = Path(args.out).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
train_queries, train_history = load_inputs(public / "train")
val_targets = read_targets(public / "validation/targets.csv")
val_queries, val_history = load_inputs(public / "validation")
train_labels = read_labels(public / "train/labels.csv", train_queries)
size_modes = size_modes_from_targets(val_targets, val_queries)
win_prior = win_prior_by_type_from_targets(val_targets, val_queries)
model = build_ensemble(train_queries, train_history, train_labels, size_modes, win_prior, variant=args.variant)
v7c = V7C(public, targets_available=True)
from slot_forecasting.peers import train_center
center = train_center(train_queries, train_history)
print(f"blend: win weight on v7c {args.win_weight}, extra peer {args.extra_peer} (centre {center:.4f})")
cfg = {"win_weight": args.win_weight, "extra_peer": args.extra_peer, "center": center, "signal_coef": args.signal_coef,
       "flip_transfer": args.flip_transfer, "subcell": args.subcell}
if args.gates:
    cfg = gate_terms(model, v7c, train_queries, train_history, train_labels, cfg)
print(f"settings after gates: {cfg}")
files = {"validation": out / "validation_predictions.csv", "test": out / "submission.csv"}
for split, path in files.items():
    queries, history = load_inputs(public / split)
    write_submission(path, combine(model.predict(queries, history), v7c.predict_ids(public, split, None),
                                   cfg["win_weight"], cfg["extra_peer"], queries, history, cfg["center"], cfg["signal_coef"],
                                   v7c.predict_ids(public, split, None, flip=False), cfg["flip_transfer"], cfg["subcell"]))

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
        fold_model = build_ensemble(train_queries, train_history, train_labels,
                                    size_modes_from_targets(prior_targets, val_queries),
                                    win_prior_by_type_from_targets(prior_targets, val_queries), log=lambda *_: None,
                                    variant=args.variant)
        fold_queries = tuple(q for q in val_queries if q.episode_id in scored)
        fold_keys = {(q.episode_id, q.machine_id) for q in fold_queries}
        fold_history = tuple(p for p in val_history if (p.episode_id, p.machine_id) in fold_keys)
        fold_v7c = V7C(public, targets_available=True, prior_episodes=prior_episode_ids, log=lambda *_: None)
        keep = {q.query_id for q in fold_queries}
        crossfit.update(combine(fold_model.predict(fold_queries, fold_history),
                                fold_v7c.predict_ids(public, "validation", keep),
                                cfg["win_weight"], cfg["extra_peer"], fold_queries, fold_history, cfg["center"], cfg["signal_coef"],
                                fold_v7c.predict_ids(public, "validation", keep, flip=False), cfg["flip_transfer"], cfg["subcell"]))
    write_submission(out / "crossfit_validation_predictions.csv", crossfit)
    halves = [subset_score(crossfit, val_targets, val_groups,
                           read_submission(val / "reference.csv"), episodes,
                           event["max_points"]) for episodes in (tuning, audit)]
    crossfit_report = {"tuning_using_audit_priors": halves[0], "audit_using_tuning_priors": halves[1]}
    write_json(out / "crossfit_validation_score.json", crossfit_report)
    print(f"Cross-fitted validation: tuning {halves[0]['technical_points']:.2f} / {event['max_points']:g}; "
          f"audit {halves[1]['technical_points']:.2f} / {event['max_points']:g}")

# --- entry folder + ZIP ---
entry = out / "entry"
if entry.exists():
    shutil.rmtree(entry)
(entry / "slot_forecasting").mkdir(parents=True)
for name in MODULES:
    shutil.copyfile(HERE / "slot_forecasting" / name, entry / "slot_forecasting" / name)
for name in ("predict.py", "reproduce.json"):
    shutil.copyfile(HERE / "entry_template_strongest" / name, entry / name)
shutil.copytree(HERE / "v7c_forecasting", entry / "v7c_forecasting", ignore=shutil.ignore_patterns("__pycache__"))
(entry / "combo.json").write_text(json.dumps({"win_weight": args.win_weight, "extra_peer": args.extra_peer,
                                           "center": center, "signal_coef": args.signal_coef,
                                           "flip_transfer": args.flip_transfer, "subcell": args.subcell,
                                           "gates": args.gates}), encoding="utf-8")
(entry / "v7c_priors.json").write_text(json.dumps({"size_modes": v7c.size_modes, "win_prior": v7c.win_prior}),
                                      encoding="utf-8")  # v7c's fallback if /data lacks validation targets
(entry / "variant.json").write_text(json.dumps({"variant": args.variant}), encoding="utf-8")
(entry / "priors.json").write_text(json.dumps({"size_modes": size_modes, "win_prior": win_prior}),
                                  encoding="utf-8")  # fallback if /data lacks validation targets
metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
sha = hashlib.sha256(files["test"].read_bytes()).hexdigest()
commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
(entry / "README.md").write_text(f"""# House Edge entry: v13 (our v12 + Avyukth's v7c)

- Round: {metadata.get('round', 'main')}
- Season ID: {metadata.get('season_id', 'see public/metadata.json')}
- Forecast SHA-256: {sha}
- Code version: git {commit} (built by mainFiles/run_strongest.py)
- Environment: `python` image, Python standard library only, no extra dependencies
  (tested on Python {platform.python_version()}; needs 3.11+)
- Validation: {report['technical_points']:.2f} / {report['max_points']:g} points (R = {report['mean_loss_ratio']:.4f})

## Recreate the forecast

Run from the archive root:

```sh
python predict.py --data /data --out /output/submission.csv
```

- Data used: `/data/train` (history, queries, labels), `/data/validation` (queries and
  `targets.csv`), `/data/test` (history, queries). If `validation/targets.csv` is missing,
  the same priors are read from `priors.json` in this archive. Writes only `--out`.
- Training details (all refit at run time, about 2 minutes):
  1. On train history: find the lag pairs whose match/differ drives wins, and the lags
     whose exact payout drives payout size.
  2. From validation targets: the payout-size modes per machine type, and a win-chance
     prior with dead / live / middle levels per machine type and signal.
  3. Pick 2-cell (match/differ) vs 4-cell win experts by Brier on `train/labels.csv`.
- Prediction: two full models of each machine are fitted from the data. Ours (v12): a
  log-odds average of lag-pair expert mixtures and a hidden-rule HMM, corrected for how the
  episode's other machines just did, with payout sizes from a size-rule HMM. Avyukth's v7c
  (vendored in `v7c_forecasting/`): its own lag-pair mixer with an episode flip swap and a
  train-learned payout-size rule. Final win chance = log-odds blend of the two (settings in combo.json) plus an extra peer term; payout
  sizes given a win = v7c's. Only ticks before each cutoff are read.
- Model files: `priors.json` and `v7c_priors.json` (fallback copies of the validation-derived priors).
- Seeds/settings: deterministic, no randomness; settings are the defaults in
  `slot_forecasting/pair_mixer.py`, `rule_hmm.py`, `changepoint.py` and `ensemble.py`.
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
