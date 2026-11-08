
import csv, json, sys
from pathlib import Path
src, version, fold_file, out = sys.argv[1:5]
sys.path[:0] = [src, "/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/mainFiles", "/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/mainFiles/combo"]
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting import pair_mixer as pm
import predict as combo
data = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/mainFiles/public")
q, h = load_inputs(data / "train")
labels = read_labels(data / "train/labels.csv", q)
rules = combo.fit_rules(q, h, labels)
vq, vh = load_inputs(data / "validation")
targets = read_targets(data / "validation/targets.csv")
folds = json.loads(Path(fold_file).read_text())
rows = {}
for name, fit_eps in folds.items():
    fit_eps = set(fit_eps)
    t = [x for x in targets if x.episode_id in fit_eps]
    qs = [x for x in vq if x.episode_id in fit_eps]
    kw = {"size_modes": pm.size_modes_from_targets(t, qs)}
    kw["win_prior"] = pm.win_prior_by_type_from_targets(t, qs)
    mixer = pm.adapt(q, h, labels, log=lambda *_: None, **kw)
    pred = combo.blend(rules, mixer, vq, vh)
    for x in vq:
        if (name == "all") or (x.episode_id not in fit_eps):
            rows.setdefault(name if name == "all" else "oof", {})[x.query_id] = list(pred[x.query_id])
Path(out).write_text(json.dumps(rows))
