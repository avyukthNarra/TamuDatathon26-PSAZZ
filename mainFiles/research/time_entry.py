import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slot_forecasting.io import load_inputs, read_labels, read_targets
from slot_forecasting.pair_mixer import size_modes_from_targets, win_prior_by_type_from_targets
from slot_forecasting.ensemble import build_ensemble
from slot_forecasting.combo import V7C, combine, gate_terms
from slot_forecasting.peers import train_center
T = time.time(); lap = lambda s: print(f"{s:28s} {time.time() - T:6.1f}s", flush=True)
data = Path("public"); tq, th = load_inputs(data / "train"); labels = read_labels(data / "train/labels.csv", tq)
vt = read_targets(data / "validation/targets.csv"); vq = load_inputs(data / "validation")[0]
lap("load")
model = build_ensemble(tq, th, labels, size_modes_from_targets(vt, vq), win_prior_by_type_from_targets(vt, vq), log=lambda *_: None, variant="strongest"); lap("fit ours")
v7c = V7C(data, True, log=lambda *_: None); lap("fit v7c")
cfg = {"win_weight": 0.3, "extra_peer": 0.5, "center": train_center(tq, th), "signal_coef": 0.05, "flip_transfer": 1.0, "subcell": True}
cfg = gate_terms(model, v7c, tq, th, labels, cfg, log=lambda *_: None); lap("gates")
q, h = load_inputs(data / "test"); o = model.predict(q, h); lap("predict test: ours")
a = v7c.predict_ids(data, "test", None); b = v7c.predict_ids(data, "test", None, flip=False); lap("predict test: v7c x2")
combine(o, a, cfg["win_weight"], cfg["extra_peer"], q, h, cfg["center"], cfg["signal_coef"], b, cfg["flip_transfer"], cfg["subcell"]); lap("combine")
