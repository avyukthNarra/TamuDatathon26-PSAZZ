"""Score model settings on validation with the kit's own contest scorer.

Prints points on ALL validation episodes plus the two halves (tune / audit) the kit's
experiment runner uses, and the four condition ratios. Lower ratio = better, 1.0 = reference.
"""
from pathlib import Path

from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS

from lag_model import LagForecaster

DATA = Path(__file__).resolve().parent / "public"


class Bench:
    def __init__(self, data=DATA):
        data = Path(data)
        self.max_points = round_metadata(data)["max_points"]
        self.train_q, self.train_h = load_inputs(data / "train")
        self.labels = read_labels(data / "train/labels.csv", self.train_q)
        self.val_q, self.val_h = load_inputs(data / "validation")
        v = data / "validation"
        self.targets = read_targets(v / "targets.csv")
        self.groups = {r["query_id"]: r["condition"] for r in read_rows(v / "conditions.csv", ("query_id", "condition"))}
        self.reference = read_submission(v / "reference.csv", {q.query_id for q in self.val_q})
        self.tune, self.audit = episode_folds(self.targets, self.groups)

    def score(self, predictions, episodes=None):
        ts = [t for t in self.targets if episodes is None or t.episode_id in episodes]
        ids = {t.query_id for t in ts}
        return contest_score({k: predictions[k] for k in ids}, ts, {k: self.groups[k] for k in ids},
                             {k: self.reference[k] for k in ids}, max_points=self.max_points, version=VERSION)

    def run(self, model):
        model.fit(self.train_q, self.train_h, self.labels)
        preds = model.predict(self.val_q, self.val_h)
        full = self.score(preds)
        return {"all": full["technical_points"],
                "tune": self.score(preds, self.tune)["technical_points"],
                "audit": self.score(preds, self.audit)["technical_points"],
                **{c: full["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}, preds


def show(name, r):
    print(f"{name:42} all {r['all']:6.2f}  tune {r['tune']:6.2f}  audit {r['audit']:6.2f}  | "
          + "  ".join(f"{c[:6]} {r[c]:.3f}" for c in CONDITIONS), flush=True)


if __name__ == "__main__":
    bench = Bench()
    show("pooled_recent (backup)", bench.run(LagForecaster(lags=()))[0])
    show("lag model v1 (defaults)", bench.run(LagForecaster())[0])
