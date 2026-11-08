"""Own W4 feature cache rebased on the lead B4 implementation.

Reuses the already-tested historical feature definitions; no B3 predictions
or coefficients enter this cache. Dense grids and train lag screening are
read from the lead cache. Only own gitignored feature artifacts are written.
"""
import hashlib
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SAI = HERE.parent
sys.path.insert(0, str(SAI))
from harness import load_split, read_pred_csv
import harness.data as hd
import harness.results as hr

SHARED = Path(os.environ.get("HE_SHARED_CACHE", SAI.parents[1] / "TamuDatathon26-PSAZZ/docs/sai/cache"))
# The default is the sibling lead checkout; explicit override supports new kits.
if not SHARED.is_dir():
    SHARED = SAI.parents[2] / "TamuDatathon26-PSAZZ/docs/sai/cache"
CACHE = SAI / "cache"
CACHE.mkdir(exist_ok=True)
hr.CACHE = CACHE
hd.CACHE = SHARED
sys.path.insert(0, str(HERE))
import _core

CONFIGS = [dict(zip(("signal", "pairs", "prior", "density"), x))
           for x in itertools.product((0, 1), repeat=4)]
_DATA = None

def _priors(t,types,signals):
    tab=np.zeros((len(types),len(signals),5)); eps={}
    for i,g in enumerate(t.grid):
        c=np.array([(g==p).sum() for p in hd.PAY],float)
        a,b=types.index(t.mtype[i]),signals.index(int(t.signal[i]))
        tab[a,b]+=c
        eps.setdefault(str(t.episode[i]),np.zeros_like(tab))[a,b]+=c
    return tab,eps

def _historical_features(g,cut,ty,sg,prior,lags,pairs):
    observed=np.flatnonzero(np.isfinite(g[:cut])); vals=g[observed]
    assert np.all(observed<cut)
    win=np.where(np.isfinite(g),2*(g>0)-1,0).astype(float)
    old=observed<cut-40; oldfrac=old.sum()/max(cut-40,1); recentfrac=(~old).sum()/min(cut,40)
    density=np.where(old,np.clip(recentfrac/max(oldfrac,1e-3),1,20),1.)
    freq=np.empty((2,2,2,5)); ess=[]
    for pi in (0,1):
        for de in (0,1):
            for hi,hl in enumerate((16.,128.)):
                w=2.**(-(cut-1-observed)/hl)*(density if de else 1)
                c=np.array([w[vals==p].sum() for p in hd.PAY])
                freq[pi,de,hi]=(c+8*prior[pi])/(c.sum()+8)
                if pi==de==0:ess.append(w.sum()**2/max(w@w,1e-9))
    X=[];names=[]
    def add(n,v):names.append(n);X.append(float(v))
    for k in range(1,17):
        p=g[cut-k] if cut>=k else np.nan
        add('lag_win_'+str(k),0 if not np.isfinite(p) else 2*(p>0)-1)
        add('lag_missing_'+str(k),not np.isfinite(p))
        add('lag_payout_'+str(k),0 if not np.isfinite(p) else p)
        add('lag_log_'+str(k),0 if not np.isfinite(p) else np.log1p(p))
    for a,b in pairs:add(f'pair_{a}_{b}',win[cut-a]*win[cut-b] if cut>=max(a,b) else 0)
    for hi,h in enumerate((16,128)):
        for k in range(5):add(f'freq_{h}_{k}',freq[0,0,hi,k])
        add(f'ess_{h}',ess[hi])
    add('obs_recent',recentfrac);add('obs_old',oldfrac);add('obs_count',len(observed));add('obs_cutoff',cut)
    for h in (4,8,16,40):
        z=vals[observed>=cut-h]
        for label,v in [('mean',z.mean() if len(z) else 0),('logmean',np.log1p(z).mean() if len(z) else 0),('var',z.var() if len(z) else 0),('missing',1-len(z)/h)]:add(f'moment_{h}_{label}',v)
    for thr in (1,5,20):
        tt=observed[vals>=thr];add('since_'+str(thr),cut-int(tt[-1]) if len(tt) else cut+1)
    a,b=freq[0,0];m=(a+b)/2
    add('drift_tv',np.abs(a-b).sum()/2);add('drift_l2',((a-b)**2).sum());add('drift_jsd',.5*((a*np.log(a/m)).sum()+(b*np.log(b/m)).sum()))
    w=(vals>0).astype(float);cs=np.cumsum(w-w.mean());add('cusum_range',np.ptp(cs)/np.sqrt(max(len(w),1)))
    return np.array(X),names,freq


def _input_hash(split):
    h = hashlib.sha256()
    for name in ("queries.csv", "history.csv"):
        h.update((hd.DATA / split / name).read_bytes())
    return h.hexdigest()


def _block(split, train, priors, types, signals, pairs):
    tab, by_episode = priors
    xs, frequencies, names = [], [], None
    for i, cutoff in enumerate(split.cutoff):
        ty, signal = types.index(split.mtype[i]), signals.index(int(split.signal[i]))
        counts = tab.copy()
        if split.name == "train":
            counts -= by_episode[str(split.episode[i])]
        prior = [(counts[ty].sum(0) + .5) / (counts[ty].sum() + 2.5),
                 (counts[ty, signal] + .5) / (counts[ty, signal].sum() + 2.5)]
        x, names, f = _historical_features(split.grid[i], int(cutoff), ty, signal, prior, [7], pairs)
        cat = [float(split.mtype[i] == value) for value in types]
        cat += [float(int(split.signal[i]) == value) for value in signals]
        cat += [float(split.mtype[i] == t and int(split.signal[i]) == z) for t in types for z in signals]
        xs.append(np.r_[x, cat]); frequencies.append(f)
    names += [f"type_{t}" for t in types] + [f"signal_{z}" for z in signals]
    names += [f"interaction_{t}_{z}" for t in types for z in signals]
    freq = np.asarray(frequencies)
    peer = np.zeros((split.n, 7))
    groups = {}
    for i, key in enumerate(zip(split.episode, split.cutoff)):
        groups.setdefault(key, []).append(i)
    for ids in groups.values():
        for i in ids:
            peers = [j for j in ids if j != i]
            if peers:
                delta = freq[peers, 0, 0, 0] - freq[peers, 0, 0, 1]
                peer[i] = np.r_[delta.mean(0), np.abs(delta).sum(1).mean()/2, len(peers)]
    names += [f"peer_residual_{k}" for k in range(5)] + ["peer_mean_tv", "peer_count"]
    pw, fits = _core.predict_pw(split, _B4_CFG, return_extra=True)
    base = _core.finish(pw, split)
    return dict(X=np.column_stack([xs, peer]).astype("float32"), names=np.array(names),
                freq=freq.astype("float32"), base=base,
                hl=np.log2(np.minimum([fit[2] for fit in fits], 1024.)),
                episode=split.episode, mtype=split.mtype, signal=split.signal,
                cutoff=split.cutoff, outcome=split.outcome)


def cell_X(block, cfg, ablation=None):
    names = list(block["names"])
    x = block["X"].copy()
    for j, name in enumerate(names):
        if ((not cfg["signal"] and name.startswith(("signal_", "interaction_")))
                or (not cfg["pairs"] and name.startswith("pair_"))):
            x[:, j] = 0
        if name.startswith("freq_"):
            _, h, k = name.split("_")
            x[:, j] = block["freq"][:, cfg["prior"], cfg["density"], 0 if h == "16" else 1, int(k)]
    if ablation:
        patterns = {"lags": ("lag_", "pair_"), "frequency": ("freq_", "ess_", "obs_"),
                    "drift": ("drift_", "cusum_"), "context": ("peer_", "moment_", "since_")}
        if ablation == "compact":
            keep = [n.startswith(("freq_", "ess_", "obs_", "drift_", "type_", "signal_", "interaction_")) for n in names]
        else:
            keep = [not n.startswith(patterns[ablation]) for n in names]
        x = x[:, keep]
    return np.column_stack([x, block["base"], block["hl"]])


def load_data():
    global _DATA, _B4_CFG
    if _DATA is not None:
        return _DATA
    train, val = load_split("train"), load_split("validation")
    doc = json.loads((SAI / "results/B4.json").read_text())
    _B4_CFG = _core.C(**doc["params"])
    # Use the existing train screen, never build a worker-specific replacement.
    screen_name = f"w1_screen_{train.n}_{int(np.nansum(train.grid))}.pkl"
    if not (CACHE / screen_name).is_file():
        if not (SHARED / screen_name).is_file():
            raise RuntimeError("Lead train lag-screen cache missing")
        shutil.copyfile(SHARED / screen_name, CACHE / screen_name)
    for p in SHARED.glob("w1_sizes_*.pkl"):
        if not (CACHE / p.name).is_file():
            shutil.copyfile(p, CACHE / p.name)
    structure = _core.learned(train)
    metadata = dict(base="B4", base_params=doc["params"], train_hash=_input_hash("train"),
                    validation_hash=_input_hash("validation"),
                    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    core_hash=hashlib.sha256((HERE / "_core.py").read_bytes()).hexdigest(),
                    screen_sha256=hashlib.sha256((CACHE / screen_name).read_bytes()).hexdigest(),
                    feature_pairs=structure["pair_top3"],
                    signal_toggle="explicit signal columns only; A2 prior remains signal-conditioned",
                    hl_grid="B4 memory fixed; historical summary half-lives16/128 unchanged",
                    train_features="priors leave out the whole train episode; no validation target use")
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"w4_b4_features_{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            blocks = {tag: {k[len(tag)+1:]: z[k] for k in z.files if k.startswith(tag+"_")} for tag in ("tr", "va")}
    else:
        types, signals = sorted(set(train.mtype)), sorted(set(int(s) for s in train.signal))
        prior = _priors(train, types, signals)
        blocks = {tag: _block(split, train, prior, types, signals, structure["pair_top3"])
                  for tag, split in (("tr", train), ("va", val))}
        np.savez_compressed(path, **{f"{tag}_{k}": x for tag, block in blocks.items() for k, x in block.items()})
    canonical = read_pred_csv(SAI / "preds/B4_validation.csv", val)
    difference = float(np.max(np.abs(blocks["va"]["base"] - canonical)))
    assert difference < 1e-12, ("B4 mismatch", difference)
    metadata["validation_base_max_diff"] = difference
    metadata["cache"] = path.name
    metadata["feature_count"] = len(blocks["va"]["names"]) + 6
    metadata["rows"] = {tag: len(block["base"]) for tag, block in blocks.items()}
    (CACHE / f"w4_b4_manifest_{key}.json").write_text(json.dumps(metadata, indent=1))
    _DATA = dict(blocks, v=val, baseline=canonical, metadata=metadata)
    return _DATA


def validate_causality():
    d = load_data()
    train = load_split("train")
    g = train.grid[0].copy(); cutoff = int(train.cutoff[0]) - 40
    pairs = d["metadata"]["feature_pairs"]
    prior = [np.ones(5)/5, np.ones(5)/5]
    a = _historical_features(g, cutoff, 0, 0, prior, [7], pairs)
    altered = g.copy(); altered[cutoff:] = 20
    b = _historical_features(altered, cutoff, 0, 0, prior, [7], pairs)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    p1 = _core.fit_machine(g, cutoff, _B4_CFG)
    p2 = _core.fit_machine(altered, cutoff, _B4_CFG)
    assert p1[0] == p2[0] and np.array_equal(p1[1], p2[1]) and p1[2] == p2[2]
    missing = g.copy(); missing[cutoff-1] = np.nan
    loss = g.copy(); loss[cutoff-1] = 0
    m = _historical_features(missing, cutoff, 0, 0, prior, [7], pairs)
    z = _historical_features(loss, cutoff, 0, 0, prior, [7], pairs)
    assert m[0][m[1].index("lag_missing_1")] == 1 and m[0][m[1].index("lag_win_1")] == 0
    assert z[0][z[1].index("lag_missing_1")] == 0 and z[0][z[1].index("lag_win_1")] == -1
    return dict(strict_cutoff_invariance=True, missing_ticks_unknown=True, base_matches_B4=d["metadata"]["validation_base_max_diff"])


if __name__ == "__main__":
    print(json.dumps(dict(metadata=load_data()["metadata"], leakage_checks=validate_causality())))
