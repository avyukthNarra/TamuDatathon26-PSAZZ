"""Pranav's v19 as a blend member (win chance; its own size split is ignored by the blend).

v19 = our pair mixer + hidden-rule HMM + peer / signal / plateau terms, blended with Avyukth's v7c
(flip-swap transfer, jackpot sub-cell), add-ons gated on validation at build time. It lives
unchanged in vendor/v19/ (the built entry folder from pranav/super, run_strongest.py).

submission: runs vendor/v19/predict.py in its own process (priors from validation targets when
present, else its saved priors.json / v7c_priors.json; identical output either way).
crossfit: each validation half is predicted with priors fit on the OTHER half (sha256 episode
folds, the same split as run_strongest.py --cv-val and this entry's other members).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1]
V19 = ENTRY / "vendor" / "v19"


def resolve_data(path):
    data = Path(path)
    return data / "public" if (data / "public" / "train").is_dir() else data


def crossfit(data, out):
    sys.path.insert(0, str(V19))
    from slot_forecasting.combo import V7C, combine
    from slot_forecasting.ensemble import build_ensemble
    from slot_forecasting.experiments import episode_folds
    from slot_forecasting.io import load_inputs, read_labels, read_rows, read_targets, write_submission
    from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets
    cfg = json.loads((V19 / "combo.json").read_text(encoding="utf-8"))
    tq, th = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", tq)
    vq, vh = load_inputs(data / "validation")
    vt = read_targets(data / "validation" / "targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(data / "validation" / "conditions.csv", ("query_id", "condition"))}
    tuning, audit = episode_folds(vt, groups)
    result = {}
    for scored, other in ((tuning, audit), (audit, tuning)):
        prior = tuple(t for t in vt if t.episode_id in other)
        model = build_ensemble(tq, th, labels, size_modes_from_targets(prior, vq), win_prior_by_type_from_targets(prior, vq),
                               log=lambda *_: None, variant="strongest")
        qs = tuple(q for q in vq if q.episode_id in scored)
        keys = {(q.episode_id, q.machine_id) for q in qs}
        hs = tuple(p for p in vh if (p.episode_id, p.machine_id) in keys)
        v7c = V7C(data, targets_available=True, prior_episodes=other, log=lambda *_: None)
        keep = {q.query_id for q in qs}
        result.update(combine(model.predict(qs, hs), v7c.predict_ids(data, "validation", keep), cfg["win_weight"],
                              cfg["extra_peer"], qs, hs, cfg["center"], cfg["signal_coef"],
                              v7c.predict_ids(data, "validation", keep, flip=False), cfg["flip_transfer"], cfg["subcell"]))
    write_submission(out, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol", choices=("submission", "crossfit"), default="submission")
    args = parser.parse_args()
    data = resolve_data(args.data)
    if args.protocol == "crossfit":
        crossfit(data, Path(args.out))
    else:
        subprocess.run([sys.executable, "predict.py", "--data", str(data), "--out", str(Path(args.out).resolve()),
                        "--split", args.split], cwd=V19, check=True)
    print(f"pranav_v19 wrote {args.out}")


if __name__ == "__main__":
    main()
