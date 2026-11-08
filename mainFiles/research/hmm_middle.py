"""HMM with an extra live level: each rule state comes in a 'live at the live level' and a
'live at the middle level' version (levels from the type|signal win prior). Scored inside the
v10 recipe (log-odds pool 0.7/0.3 + peers 1.0) on both checks (run from mainFiles/)."""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
sys.argv = sys.argv[:1] + ["--methods", "none"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import peer_correction as P
stack = P.stack
from slot_forecasting.io import load_inputs, read_targets
from slot_forecasting.pair_mixer import choose_pairs, win_prior_by_type_from_targets
from slot_forecasting.rule_hmm import HiddenRuleHMM

class MiddleHMM(HiddenRuleHMM):
    def __init__(self, pairs, levels3, middle_weight=0.3, **kw):
        super().__init__(pairs, {}, **kw)
        self.levels3 = levels3
        base = self.states[:-1]
        self.states = [(s, "live") for s in base] + [(s, "mid") for s in base] + [None]
        share = (1.0 - self.none_weight) / len(base)
        self.prior = [share * (1 - middle_weight)] * len(base) + [share * middle_weight] * len(base) + [self.none_weight]
        self._lv = None
    def _emission(self, seq, t, state, rate, live, dead):
        if state is None: return rate
        rule, kind = state
        on = self._live(seq, t, rule)
        level = self._lv[1] if kind == "mid" else self._lv[0]
        return rate if on is None else (level if on else self._lv[2])
    def win_chances(self, queries, history):
        from slot_forecasting.schema import index_history, unique_queries
        indexed = index_history(queries, history); out = {}
        for q in unique_queries(queries).values():
            self._lv = self.levels3.get(f"{q.machine_type}|{q.signal}") or self.levels3[q.machine_type]
            seq = {p.tick: p.payout for p in indexed.get((q.episode_id, q.machine_id), [])}
            out[q.query_id] = self.win_probability(seq, q.cutoff, 0, 0)
        return out

def levels3(prior):
    return {k: (c[1][1], c[2][1], c[0][1]) for k, c in prior.items()}

tq, th = load_inputs("public/train"); vq, vh = load_inputs("public/validation"); vt = read_targets("public/validation/targets.csv")
pairs = choose_pairs(tq, th)
from slot_forecasting.schema import Query
def hmm_for(rows, prior, split, mw):
    m = MiddleHMM(pairs, levels3(prior), middle_weight=mw)
    out = {}
    for r in rows:
        q = Query(r["qid"], r["episode"], r["machine"], r["type"], r["signal"], r["cutoff"])
        seq = {t: v for t, v in P.R.HIST[split][(r["episode"], r["machine"])].items() if t < r["cutoff"]}
        m._lv = m.levels3.get(f"{r['type']}|{r['signal']}") or m.levels3[r["type"]]
        out[id(r)] = m.win_probability(seq, r["cutoff"], 0, 0)
    return out
def lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
def win(r, h):
    z = 0.7 * lg(1 - r["mixer"][0]) + 0.3 * lg(h)
    sh = r["x"]["peers 1-5"]
    return 1 / (1 + math.exp(-(z + (0 if sh is None else (sh - 0.4187)))))
from slot_forecasting.experiments import episode_folds
for mw in (0.2, 0.4):
    hv = {}
    for scored, other in zip(stack.HALVES, reversed(stack.HALVES)):
        pr = win_prior_by_type_from_targets([t for t in vt if t.episode_id in other], vq)
        hv.update(hmm_for([r for r in P.VAL if r["episode"] in scored], pr, "validation", mw))
    ht = hmm_for(P.TRAIN, win_prior_by_type_from_targets(vt, vq), "train", mw)
    preds = {r["qid"]: stack.as_probs(r, win(r, hv[id(r)])) for r in P.VAL}
    s = [stack.subset_score(preds, stack.vt, stack.vcond, stack.ref, e, 60.0) for e in (stack.ALL_EP, *stack.HALVES)]
    res = {}
    for which in ("cutoff", "window"):
        per_q = defaultdict(list)
        for r in P.TRAIN:
            if which == "cutoff" and r["back"]: continue
            per_q[(r["condition"], r["episode"], r["qid"])].append(stack.brier(stack.as_probs(r, win(r, ht[id(r)])), r["outcome"]) - stack.brier(stack.as_probs(r, win(r, r["hmm"])), r["outcome"]))
        dd = defaultdict(lambda: defaultdict(list))
        for (c, e, _), v in per_q.items(): dd[c][e].append(sum(v) / len(v))
        res[which] = sum(sum(sum(x) / len(x) for x in dd[c].values()) / len(dd[c]) / stack.BASE_ERR[c] for c in dd) / len(dd)
    bc = s[0]["by_condition"]
    print(f"middle-state weight {mw}: validation {s[0]['technical_points']:.2f} ({s[1]['technical_points']:.2f} / {s[2]['technical_points']:.2f}) " + " ".join(f"{c[:5]} {bc[c]['loss_ratio']:.3f}" for c in sorted(bc)) + f" | train vs v10 cutoff {res['cutoff']:+.4f} window {res['window']:+.4f}", flush=True)
