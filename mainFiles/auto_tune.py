"""Re-tune the blend on whatever round's data you point it at, and keep it only if it holds up.

    python3 auto_tune.py --data public --out runs/tune-main
    python3 auto_tune.py --data <round-two kit>/public --out runs/tune-round2

What it does:
  1. Fits every candidate model on <data>/train:
       rule       our rule model (all 37 lag rules, rules + payout-size rules)
       mixer      team pair mixer v3 (pairs rediscovered on train; size modes + win prior from validation)
       mixer_all  pair mixer over all 28 lag pairs (safer if round two spreads rules out)
       hmm        our hidden-rule HMM (hmm_model.py), payout sizes from the mixer
  2. Predicts <data>/validation with each, and scores every blend on a 0.1 grid.
  3. Picks the blend with the best score on the TUNE half of validation episodes, then checks it on
     the AUDIT half. It only replaces the current blend if it gains at least --min-gain (0.15)
     points on the tune half AND the audit half agrees (not worse by more than --tolerance).
     Smaller gains are noise at 12 episodes per condition per half.
  4. Writes <out>/auto_tune.json (every blend's scores) and, if accepted, blend_config.json, which
     predict.py reads. Then build the entry with make_entry.py as usual.

Each run is one "attempt". Run it again after changing a model, or on a new round; it only ever
keeps a change that is better on episodes it didn't choose on.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from slot_forecasting.contest import contest_score
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets, write_json
from slot_forecasting.pair_mixer import ALL_PAIRS, WIN_PRIOR_ETA, PairMixer, adapt
from slot_forecasting.rounds import VERSION, round_metadata
from slot_forecasting.schema import CONDITIONS

import predict

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "blend_config.json"
DATA = None  # set in main()
TRAIN = None  # (train queries, train history), set in main()


def candidates(queries, history, labels, log):
    """Name -> fitted model. Each must have predict(queries, history)."""
    data = DATA
    priors = predict.mixer_priors(data)
    rules = predict.rule_model().fit(queries, history, labels)
    mixer = adapt(queries, history, labels, log=log, **priors)
    extra = {"win_prior": priors["win_prior"], "eta": WIN_PRIOR_ETA} if priors["win_prior"] else {}
    mixer_all = PairMixer(pairs=ALL_PAIRS, cell_mode=mixer.cell_mode, size_modes=priors["size_modes"],
                          **extra).fit(queries, history)
    return {"rule": rules, "mixer": mixer, "mixer_all": mixer_all}


def hmm_predictions(queries, history, size_from, data):
    """Our hidden-rule HMM's forecast, with payout sizes borrowed from size_from (as in predict.py)."""
    model = predict.HiddenRuleHMM(predict.choose_pairs(*TRAIN), levels=predict.hmm_levels(data))
    return model.predict(queries, history, size_from=size_from)


def grid(names, step=0.1):
    """All weight vectors on a grid that sum to 1."""
    n = round(1 / step)
    for combo in itertools.product(range(n + 1), repeat=len(names)):
        if sum(combo) == n:
            yield {name: c / n for name, c in zip(names, combo) if c}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "public")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="max audit-half points a new blend may lose vs the current config")
    parser.add_argument("--min-gain", type=float, default=0.15,
                        help="tune-half points a new blend must gain over the current one (smaller = noise)")
    parser.add_argument("--dry-run", action="store_true", help="never write blend_config.json")
    args = parser.parse_args()
    data = args.data.resolve()
    if (data / "public").is_dir():
        data = data / "public"
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; pick a new --out so earlier attempts are kept.")
    out.mkdir(parents=True)
    global DATA
    DATA = data
    meta = round_metadata(data)
    log = lambda *a: print(*a, flush=True)
    log(f"round={meta['round']} season={meta['season_id']} max_points={meta['max_points']}")

    start = time.time()
    tq, th = load_inputs(data / "train")
    labels = read_labels(data / "train" / "labels.csv", tq)
    global TRAIN
    TRAIN = (tq, th)
    models = candidates(tq, th, labels, log)
    vq, vh = load_inputs(data / "validation")
    preds = {}
    for name, model in models.items():
        t0 = time.time()
        preds[name] = model.predict(vq, vh)
        log(f"  predicted validation with {name} ({time.time() - t0:.0f}s)")
    preds["hmm"] = hmm_predictions(vq, vh, preds["mixer"], data)
    log("  predicted validation with hmm")

    v = data / "validation"
    targets = read_targets(v / "targets.csv")
    groups = {r["query_id"]: r["condition"] for r in read_rows(v / "conditions.csv", ("query_id", "condition"))}
    reference = read_submission(v / "reference.csv", {q.query_id for q in vq})
    tune, audit = episode_folds(targets, groups)
    ids = sorted(groups)

    def score(weights, episodes=None):
        p = {k: tuple(sum(w * preds[n][k][i] for n, w in weights.items()) for i in range(5)) for k in ids}
        ts = [t for t in targets if episodes is None or t.episode_id in episodes]
        keep = {t.query_id for t in ts}
        return contest_score({k: p[k] for k in keep}, ts, {k: groups[k] for k in keep},
                             {k: reference[k] for k in keep}, max_points=meta["max_points"], version=VERSION)

    rows = []
    for weights in grid(list(preds)):
        full = score(weights)
        rows.append({"weights": weights, "all": full["technical_points"],
                     "tune": score(weights, tune)["technical_points"],
                     "audit": score(weights, audit)["technical_points"],
                     **{c: full["by_condition"][c]["loss_ratio"] for c in CONDITIONS}})
    rows.sort(key=lambda r: -r["tune"])

    current = json.loads(CONFIG.read_text()) if CONFIG.exists() else {"weights": dict(predict.DEFAULT_WEIGHTS)}
    cur = next((r for r in rows if r["weights"] == current["weights"]), None)
    if cur is None:  # current config not on the grid (or uses an unknown model): score it directly
        w = {n: x for n, x in current["weights"].items() if n in preds}
        full = score(w)
        cur = {"weights": w, "all": full["technical_points"], "tune": score(w, tune)["technical_points"],
               "audit": score(w, audit)["technical_points"],
               **{c: full["by_condition"][c]["loss_ratio"] for c in CONDITIONS}}

    def line(tag, r):
        w = " + ".join(f"{x:.1f} {n}" for n, x in r["weights"].items())
        return (f"{tag:9} {w:34} all {r['all']:6.2f}  tune {r['tune']:6.2f}  audit {r['audit']:6.2f} | "
                + "  ".join(f"{c[:6]} {r[c]:.3f}" for c in CONDITIONS))

    log("\nsingle models:")
    for name in preds:
        log(line("", next(r for r in rows if r["weights"] == {name: 1.0})))
    log("\ntop blends by TUNE half:")
    for r in rows[:5]:
        log(line("", r))
    best = rows[0]
    log("\n" + line("current", cur))
    log(line("best", best))

    gain = best["tune"] - cur["tune"]
    confirmed = best["audit"] >= cur["audit"] - args.tolerance
    accepted = best["weights"] != cur["weights"] and gain >= args.min_gain and confirmed
    if accepted:
        verdict = f"ACCEPTED: +{gain:.2f} on tune half, audit half agrees"
    elif best["weights"] == cur["weights"]:
        verdict = "KEPT CURRENT: already the best blend"
    elif gain < args.min_gain:
        verdict = f"KEPT CURRENT: best is only +{gain:.2f} on the tune half (< {args.min_gain}, i.e. noise)"
    else:
        verdict = "KEPT CURRENT: audit half doesn't confirm the tune-half gain"
    log(f"\n{verdict}  ({time.time() - start:.0f}s)")
    write_json(out / "auto_tune.json", {"round": meta["round"], "season_id": meta["season_id"],
                                        "current": cur, "best": best, "accepted": accepted, "all": rows})
    if accepted and not args.dry_run:
        CONFIG.write_text(json.dumps({"weights": best["weights"], "chosen_on": meta["round"],
                                      "tune": round(best["tune"], 3), "audit": round(best["audit"], 3)},
                                     indent=2) + "\n")
        log(f"wrote {CONFIG.name}; now run: python3 make_entry.py --data {args.data} --out runs/<name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
