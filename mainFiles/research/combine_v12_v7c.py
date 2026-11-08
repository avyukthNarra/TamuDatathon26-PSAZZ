import csv, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.experiments import episode_folds, subset_score
from slot_forecasting.io import read_submission, read_targets
W = "/private/tmp/claude-501/-Users-pranavkhanna-TamuDatathon26-PSAZZ/8348320b-7442-4757-b167-1103cebd0291/scratchpad/wt-v7c/runs-v7c"
vt = read_targets("public/validation/targets.csv"); ref = read_submission("public/validation/reference.csv")
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open("public/validation/conditions.csv"))}
H = episode_folds(vt, cond); ALL = {t.episode_id for t in vt}
def lg(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def combine(a, b, ww, sw):
    out = {}
    for k in a:
        wa, wb = 1 - a[k][0], 1 - b[k][0]
        w = 1 / (1 + math.exp(-((1 - ww) * lg(wa) + ww * lg(wb))))
        sa, sb = [x / wa for x in a[k][1:]], [x / wb for x in b[k][1:]]
        size = [(1 - sw) * x + sw * y for x, y in zip(sa, sb)]
        raw = [max(x, 1e-4) for x in [1 - w] + [w * s for s in size]]; t = sum(raw); out[k] = [x / t for x in raw]
    return out
for tag, fa, fb in (("cross-fit", "runs/strongest-v12/crossfit_validation_predictions.csv", W + "/crossfit_validation_predictions.csv"),
                    ("full", "runs/strongest-v12/validation_predictions.csv", W + "/validation_predictions.csv")):
    a, b = read_submission(fa), read_submission(fb)
    print(f"--- {tag} (win weight on v7c / size weight on v7c)")
    for ww, sw in ((0, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 1), (0.5, 0.5), (0.3, 1), (0.3, 0.7)):
        p = combine(a, b, ww, sw)
        s = [subset_score(p, vt, cond, ref, e, 60.0) for e in (ALL, *H)]
        bc = s[0]["by_condition"]
        print(f"  win {ww:.1f} / size {sw:.1f}: {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) " + " ".join(f"{c[:5]} {bc[c]['loss_ratio']:.3f}" for c in sorted(bc)))
