"""v13, the strongest build: our v12 win chance blended with Avyukth's v7c, plus v7c's payout sizes.

  win chance  = sigmoid(0.5 logit(v12 win) + 0.5 logit(v7c win))
  size | win  = v7c's (its train-learned size rule: two atoms per type, a per-machine lag and
                polarity bit, static fallback)
v12 = v11 (log-odds pool of pair mixer + hidden-rule HMM, peer corrections, plateau average) with
the size-rule HMM from the zhill branch; v7c = vendored unchanged in v7c_forecasting/.
Our win chance is better in stable / short_history / drift, v7c's (episode flip swap) in
shared_change; v7c's sizes are better in every condition. Cross-fitted validation: v12 57.33
(57.47 / 57.18), v7c 57.34 (57.31 / 57.39), this blend 57.50 (57.55 / 57.44) - above both parents
on both halves (research/combine_v12_v7c.py). Equal win weights on purpose (no tuning).
"""
import math

from .schema import probabilities

MIN_PROBABILITY = 1e-4


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class V7C:
    """Avyukth's v7c pipeline (as in its predict.py), fitted once, predicting like a model."""

    def __init__(self, data_dir, targets_available, saved_priors=None, prior_episodes=None, log=print):
        from v7c_forecasting.entry_config import USE_FLIP, USE_ROUND2, USE_SIZE, USE_SIZE_GATE
        from v7c_forecasting.forecast import fit_flip, fit_size, predict_split
        from v7c_forecasting.io import load_inputs, read_labels, read_targets
        from v7c_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_by_type_from_targets
        self._load, self._predict_split = load_inputs, predict_split
        tq, th = load_inputs(data_dir / "train")
        labels_path = data_dir / "train" / "labels.csv"
        labels = read_labels(labels_path, tq) if labels_path.exists() else None
        if targets_available:
            targets = read_targets(data_dir / "validation" / "targets.csv")
            if prior_episodes is not None:
                targets = [t for t in targets if t.episode_id in prior_episodes]
            vq = load_inputs(data_dir / "validation")[0]
            self.size_modes = size_modes_from_targets(targets, vq)
            self.win_prior = win_prior_by_type_from_targets(targets, vq, two_means=USE_ROUND2)
        else:
            self.size_modes, self.win_prior = saved_priors["size_modes"], saved_priors["win_prior"]
        quiet = (lambda *_: None)
        self.model = adapt(tq, th, labels, log=quiet, size_modes=self.size_modes, win_prior=self.win_prior, round2=USE_ROUND2)
        self.size_rule = fit_size(tq, th, USE_SIZE, labels, USE_SIZE_GATE)
        self.gate = fit_flip(self.model, tq, th, USE_FLIP)
        log(f"v7c member: size rule {'on' if self.size_rule else 'off'}, flip gate {'on' if self.gate else 'off'}")

    def predict_ids(self, data_dir, split, keep, flip=True):
        """Predict a split with v7c's own loaders (restricted to query ids in keep, if given)."""
        queries, history = self._load(data_dir / split)
        if keep is not None:
            queries = tuple(q for q in queries if q.query_id in keep)
            keys = {(q.episode_id, q.machine_id) for q in queries}
            history = tuple(p for p in history if (p.episode_id, p.machine_id) in keys)
        return self._predict_split(self.model, queries, history, self.size_rule, self.gate if flip else None)


WIN_WEIGHT = 0.4    # v14: share of v7c's win chance in the log-odds blend (v13 used 0.5)
EXTRA_PEER = 0.5    # v14: extra peer term (v7c ignores other machines, so the blend halved it)


def combine(ours, theirs, win_weight=0.5, extra_peer=0.0, queries=None, history=None, center=0.5, signal_coef=0.0,
            theirs_noflip=None, flip_transfer=0.0, subcell=False):
    """Blend ours and v7c. With extra_peer > 0 (and queries / history), add
    extra_peer x (peers' lag 1-5 win share - center) to the blended log-odds (peers.py).
    v14 research (research/v13_variants.py, v13_eval.py): win 0.4 + extra peer 0.5 vs v13 -
    validation 57.50 (57.56 / 57.44) vs 57.50 (57.55 / 57.44); held-out train window dR
    -0.0019 [-0.0031, -0.0007], cutoff -0.0020 [-0.0049, +0.0007] (v7c refit per fold).
    signal_coef (v15: 0.05): the residual of v14's win chance rises with the signal column on both
    validation (z +3.4) and train (z +3.1); 0.05 (half the validation optimum) lifts both
    validation halves (57.56 / 57.44 -> 57.58 / 57.47), train dR -0.0004 window, -0.0008 cutoff.
    flip_transfer (v16: 1.0, with win_weight 0.3): our win log-odds += v7c's own flip-swap effect
    (logit v7c with swap - without); vs v15 validation 57.52 (57.58 / 57.47) -> 57.57 (57.62 /
    57.52), held-out train window -0.0015 [-0.0029, -0.0002] (research/flip_transfer.py)."""
    signals = {q.query_id: q.signal for q in queries} if queries is not None else {}
    shifts = {}
    if subcell and queries is not None:  # v17: jackpot sub-cell correction on our win model
        from .subcell import subcell_shifts
        shifts = subcell_shifts(queries, history)
    shares = {}
    if extra_peer and queries is not None:
        from .peers import _by_machine, peer_share
        by_machine = _by_machine(queries, history)
        shares = {q.query_id: peer_share((q.episode_id, q.machine_id), q.cutoff, by_machine) for q in queries}
    out = {}
    for qid, a in ours.items():
        b = theirs[qid]
        wa, wb = 1.0 - a[0], 1.0 - b[0]
        share = shares.get(qid)
        bump = 0.0 if share is None else extra_peer * (share - center)
        bump += signal_coef * signals.get(qid, 0)
        if theirs_noflip is not None and flip_transfer:  # v16: give our win model v7c's flip swap
            wa_logit = _logit(wa) + flip_transfer * (_logit(wb) - _logit(1.0 - theirs_noflip[qid][0]))
        else:
            wa_logit = _logit(wa)
        wa_logit += shifts.get(qid, 0.0)
        win = 1.0 / (1.0 + math.exp(-((1 - win_weight) * wa_logit + win_weight * _logit(wb) + bump)))
        sizes = [x / wb for x in b[1:]]
        raw = [max(p, MIN_PROBABILITY) for p in [1.0 - win] + [win * s for s in sizes]]
        total = math.fsum(raw)
        out[qid] = probabilities([p / total for p in raw], qid)
    return out
