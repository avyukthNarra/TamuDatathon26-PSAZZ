"""House Edge entry: train-only PairMixer priors by default.

Usage: python predict.py --data /data --out /output/submission.csv
The default ``--prior-source train`` estimates priors from /data/train only. The
explicit ``--prior-source validation`` option is retained for historical diagnostics.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from r0_model import make_model  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_from_targets  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prior-source", choices=("train", "validation"), default="train",
                        help="train is safe for submission; validation is diagnostic only")
    args = parser.parse_args(argv)
    data = Path(args.data)
    if (data / "public").is_dir():
        data = data / "public"
    train_queries, train_history = load_inputs(data / "train")
    labels_path = data / "train" / "labels.csv"
    labels = read_labels(labels_path, train_queries) if labels_path.exists() else None

    provided_priors = None
    if args.prior_source == "validation":
        targets_path = data / "validation" / "targets.csv"
        saved = HERE / "priors.json"
        if targets_path.exists():
            targets = read_targets(targets_path)
            size_modes = size_modes_from_targets(targets, load_inputs(data / "validation")[0])
            win_prior = win_prior_from_targets(targets)
            provided_priors = {"size_modes": size_modes, "win_prior": win_prior}
        elif saved.exists():
            provided_priors = json.loads(saved.read_text(encoding="utf-8"))
        else:
            provided_priors = {"size_modes": None, "win_prior": None}

    model = make_model(prior_source=args.prior_source, provided_priors=provided_priors)
    model.fit(train_queries, train_history, labels)
    queries, history = load_inputs(data / args.split)
    write_submission(args.out, model.predict(queries, history))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
