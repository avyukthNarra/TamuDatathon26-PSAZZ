"""B0 (pooled) and B3 (pair model prototype) wrapped for the harness."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "mainFiles"))


def pooled(split, method="pooled"):
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels
    from .data import DATA
    tq, th = load_inputs(DATA / "train")
    m = FrequencyForecaster(method).fit(tq, th, read_labels(DATA / "train/labels.csv", tq))
    vq, vh = load_inputs(DATA / split.name)
    pr = m.predict(vq, vh)
    return np.array([pr[q] for q in split.qid])


def b3_prototype(split):
    spec = importlib.util.spec_from_file_location("pair_proto", HERE.parent / "scripts" / "pair_model_prototype.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q = split.frame()
    seqs = {m: split.seq(i) for i, m in enumerate(split.machine)}
    tshare = mod.type_shares(q, seqs)
    out = [mod.predict_one(r, seqs, tshare)[0] for r in q.itertuples()]
    P = np.array(out)
    return P / P.sum(1, keepdims=True)
