"""Score a validation prediction CSV on every slice the team quotes, with the kit's own scorer.

    python3 score_slices.py runs/combined-v4/validation_predictions.csv

Slices:
  all 96       every validation episode (the honest estimate for test)
  kit tune/audit  the two 48-episode halves from slot_forecasting.experiments.episode_folds
  Sai audit 24 / Sai dev 72  Sai's frozen split (sai_audit_episodes.json, copied from
               sai/combo-top10 docs/sai/harness/audit_main.json): first 6 episodes per condition
               by sha256('audit-v1|' + episode_id)
"""
import json
import sys
from pathlib import Path

from eval_lag import Bench
from slot_forecasting.io import read_submission
from slot_forecasting.schema import CONDITIONS

HERE = Path(__file__).resolve().parent


def main(paths):
    bench = Bench()
    sai = set(json.loads((HERE / "sai_audit_episodes.json").read_text())["episodes"])
    every = {t.episode_id for t in bench.targets}
    slices = [("all 96", None), ("kit tune 48", bench.tune), ("kit audit 48", bench.audit),
              ("Sai audit 24", sai), ("Sai dev 72", every - sai)]
    for path in paths:
        predictions = read_submission(path, {q.query_id for q in bench.val_q})
        print(path)
        for name, episodes in slices:
            result = bench.score(predictions, episodes)
            ratios = "  ".join(f"{c[:6]} {result['by_condition'][c]['loss_ratio']:.3f}" for c in CONDITIONS)
            print(f"  {name:13} {result['technical_points']:6.2f} / {result['max_points']:g}   {ratios}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
