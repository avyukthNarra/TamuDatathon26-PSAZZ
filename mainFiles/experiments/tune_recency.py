"""Grid-search half_life (H) and prior_strength (alpha) for the pooled_recent baseline.

Selects on half the validation episodes per condition, then reports the other half once.
Writes nothing but a JSON report; use run_starter.py / my_model.py to export predictions.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_json
from slot_forecasting.rounds import round_metadata

HALF_LIVES = (4, 8, 12, 16, 24, 32, 48, 64, 128)
STRENGTHS = (2, 5, 10, 20, 40, 80)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "public"))
parser.add_argument("--out", default="tune_recency.json")
args = parser.parse_args()

public = Path(args.data).resolve()
if Path(args.out).exists():
    raise SystemExit("Choose a new --out file.")
max_points = round_metadata(public)["max_points"]
train_q, train_h = load_inputs(public / "train")
labels = read_labels(public / "train/labels.csv", train_q)
val_q, val_h = load_inputs(public / "validation")
targets = read_targets(public / "validation/targets.csv")
groups = {r["query_id"]: r["condition"] for r in read_rows(public / "validation/conditions.csv", ("query_id", "condition"))}
reference = read_submission(public / "validation/reference.csv")
tuning, audit = episode_folds(targets, groups)

rows = []
for half in HALF_LIVES:
    for strength in STRENGTHS:
        model = FrequencyForecaster("pooled_recent", prior_strength=strength, half_life=half)
        model.fit(train_q, train_h, labels)
        preds = model.predict(val_q, val_h)
        score = subset_score(preds, targets, groups, reference, tuning, max_points)
        rows.append({"half_life": half, "prior_strength": strength, "preds": preds, "tuning": score})

rows.sort(key=lambda r: r["tuning"]["mean_loss_ratio"])
best = rows[0]
audit_score = subset_score(best["preds"], targets, groups, reference, audit, max_points)

print(f"{'H':>4} {'alpha':>6} {'tuning pts':>11}")
for r in rows[:8]:
    print(f"{r['half_life']:>4} {r['prior_strength']:>6} {r['tuning']['technical_points']:>11.3f}")
print(f"\nbest: H={best['half_life']} alpha={best['prior_strength']}")
print(f"audit (scored once, after choosing): {audit_score['technical_points']:.3f} / {max_points}")
for cond, v in audit_score["by_condition"].items():
    print(f"  {cond}: loss ratio {v['loss_ratio']:.3f}")

write_json(args.out, {
    "best": {"half_life": best["half_life"], "prior_strength": best["prior_strength"]},
    "grid": [{k: r[k] for k in ("half_life", "prior_strength", "tuning")} for r in rows],
    "audit": audit_score,
})
