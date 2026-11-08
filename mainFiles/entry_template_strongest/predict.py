"""House Edge entry v19: our win model blended with Avyukth's v7c (vendored in v7c_forecasting/).

Usage: python predict.py --data /data --out /output/submission.csv
Everything is re-learned from <data>: lag pairs, size lags and modes, win priors per type and
signal (from validation/targets.csv, else priors.json / v7c_priors.json saved at build time),
v7c's size rule and flip gate. Each add-on (jackpot sub-cell, flip transfer, peer term, signal
term) is checked against this round's validation targets when the entry is BUILT (run_strongest.py
calls `predict.py --decide-gates`) and the decided settings are saved in combo.json, so the forecast
is identical with or without validation answers mounted. v7c runs in a second process (2 CPUs).
Standard library only, deterministic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slot_forecasting.combo import V7C, combine, gate_terms, gate_terms_truth, run_in_parallel  # noqa: E402
from slot_forecasting.ensemble import build_ensemble  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_targets, write_submission  # noqa: E402
from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets  # noqa: E402


def forecast(data, split, log=print, decide_gates=False):
    """The v19 forecast for <data>/<split>; also used by run_strongest.py so both agree exactly."""
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "combo.json").read_text(encoding="utf-8"))
    train_queries, train_history = load_inputs(data / "train")
    labels_path = data / "train" / "labels.csv"
    labels = read_labels(labels_path, train_queries) if labels_path.exists() else None
    targets_path = data / "validation" / "targets.csv"
    targets = read_targets(targets_path) if targets_path.exists() else None
    if targets is not None:
        validation_queries, validation_history = load_inputs(data / "validation")
        size_modes = size_modes_from_targets(targets, validation_queries)
        win_prior = win_prior_by_type_from_targets(targets, validation_queries)
    else:
        priors = json.loads((here / "priors.json").read_text(encoding="utf-8"))
        size_modes, win_prior = priors["size_modes"], priors["win_prior"]
    v7c_saved = json.loads((here / "v7c_priors.json").read_text(encoding="utf-8"))
    gate_on_truth = decide_gates and targets is not None
    gate_on_train = decide_gates and targets is None and labels is not None

    def v7c_job():
        v7c = V7C(data, targets_available=targets is not None, saved_priors=v7c_saved, log=lambda *_: None)
        out = {"split": v7c.predict_ids(data, split, None), "split_nf": v7c.predict_ids(data, split, None, flip=False)}
        if gate_on_truth:
            out["val"] = v7c.predict_ids(data, "validation", None)
            out["val_nf"] = v7c.predict_ids(data, "validation", None, flip=False)
        return out

    v7c_result = None if gate_on_train else run_in_parallel(v7c_job)   # second CPU
    model = build_ensemble(train_queries, train_history, labels, size_modes, win_prior, variant="strongest", log=log)
    queries, history = load_inputs(data / split)
    ours = model.predict(queries, history)
    if gate_on_truth:
        ours_val = model.predict(validation_queries, validation_history)
        r = v7c_result()
        cfg = gate_terms_truth(ours_val, r["val"], r["val_nf"], validation_queries, validation_history, targets, cfg, log=log)
    elif gate_on_train:  # no targets: fall back to the train-outcome gates (needs the fitted v7c here)
        v7c = V7C(data, targets_available=False, saved_priors=v7c_saved, log=lambda *_: None)
        cfg = gate_terms(model, v7c, train_queries, train_history, labels, cfg, log=log)
        r = {"split": v7c.predict_ids(data, split, None), "split_nf": v7c.predict_ids(data, split, None, flip=False)}
    else:
        r = v7c_result()
    log(f"settings: {json.dumps(cfg)}")
    if decide_gates:
        return cfg
    return combine(ours, r["split"], cfg["win_weight"], cfg["extra_peer"], queries, history, cfg["center"],
                   cfg.get("signal_coef", 0.0), r["split_nf"], cfg.get("flip_transfer", 0.0), cfg.get("subcell", False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--decide-gates", metavar="JSON", help="build step: decide the add-on gates on this round's "
                        "validation targets (else train outcomes) and write the settings to JSON")
    args = parser.parse_args()
    data = Path(args.data)
    if (data / "public").is_dir():
        data = data / "public"
    if args.decide_gates:
        cfg = forecast(data, "validation", decide_gates=True)
        Path(args.decide_gates).write_text(json.dumps(dict(cfg, gates=False)), encoding="utf-8")
        print(f"wrote {args.decide_gates}")
        return
    write_submission(args.out, forecast(data, args.split))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
