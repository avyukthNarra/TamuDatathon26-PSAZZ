"""Causal mixer6 features and separately fitted, opposite-prior-fold stackers."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_submission
from slot_forecasting.schema import CONDITIONS, PAYOUTS
from .mixer6_snapshot import adapt, size_modes_from_targets, win_prior_by_type_from_targets
from .run import load_dataset, calibrate, apply_calibration, head_predictions


def folds(queries, conditions, hashed=False):
    result = {}
    for condition in CONDITIONS:
        episodes = sorted({q.episode_id for q in queries if conditions[q.query_id] == condition},
                          key=(lambda e: hashlib.sha256(e.encode()).hexdigest()) if hashed else None)
        for i, episode in enumerate(episodes):
            result[episode] = 'A' if i % 2 == 0 else 'B'
    return result


def read_conditions(data):
    with (data / 'validation/conditions.csv').open() as stream:
        return {r['query_id']: r['condition'] for r in csv.DictReader(stream)}


def augment_features(x, p, used):
    return np.column_stack([x[:, used], p, logit(np.clip(1 - p[:, 0], 1e-6, 1 - 1e-6))]).astype(np.float32)


def normalized_curve(model, own, machine_type, stop):
    """Matches PairMixer.predict's floor at every strictly pre-absorption tick."""
    curve = np.empty((stop + 1, 5))
    model.walk(own, machine_type, stop, lambda t, p: curve.__setitem__(t, p))
    curve = np.maximum(curve, 1e-4)
    return curve / curve.sum(axis=1, keepdims=True)


def opposite_fold_targets(targets, assignment, scored_fold):
    return tuple(t for t in targets if assignment[t.episode_id] != scored_fold)


def mixer_features(data, output, dataset, prior_targets, name):
    path = output / (name + '_features.npz')
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    start = time.perf_counter()
    tq, th = load_inputs(data / 'train')
    labels = read_labels(data / 'train/labels.csv', tq)
    vq, _ = load_inputs(data / 'validation')
    model = adapt(tq, th, labels, size_modes=size_modes_from_targets(prior_targets, vq),
                  win_prior=win_prior_by_type_from_targets(prior_targets, vq), log=lambda *_: None)
    qindices = {str(qid): i for i, qid in enumerate(dataset['ids'])}
    row_indices = defaultdict(list)
    for i, qi in enumerate(dataset['row_query_indices']):
        row_indices[int(qi)].append(i)
    row_probs = np.zeros((len(dataset['y']), 5))
    query_probs = np.zeros((len(qindices), 5))
    for split in ('train', 'validation', 'test'):
        qs, history = load_inputs(data / split)
        sequences = {(q.episode_id, q.machine_id): [None] * (q.cutoff + 1) for q in qs}
        for p in history:
            sequences[p.episode_id, p.machine_id][p.tick] = PAYOUTS.index(p.payout)
        for q in qs:
            qi = qindices[q.query_id]
            curve = normalized_curve(model, sequences[q.episode_id, q.machine_id], q.machine_type, q.cutoff)
            indices = np.array(row_indices[qi], dtype=int)
            row_probs[indices] = curve[dataset['row_ticks'][indices]]
            query_probs[qi] = curve[q.cutoff]
        print('mixer features', name, split, round(time.perf_counter() - start, 2), flush=True)
    assert (row_probs > 0).all() and (query_probs > 0).all()
    result = {'p': row_probs, 'qp': query_probs}
    np.savez_compressed(path, **result)
    with (output / (name + '_mixer.pkl')).open('wb') as stream:
        pickle.dump(model, stream)
    (output / (name + '_priors.json')).write_text(json.dumps({
        'episodes': sorted({t.episode_id for t in prior_targets}),
        'win_prior': win_prior_by_type_from_targets(prior_targets, vq),
        'size_modes': size_modes_from_targets(prior_targets, vq),
        'dataset_signature': str(dataset['signature']), 'seconds': time.perf_counter() - start}, indent=2) + '\n')
    return result


def residual_predictions(base, delta):
    win = np.clip(1 - base[:, 0] + delta[:, 0], 1e-4, 1 - 1e-4)
    sizes = base[:, 1:] / base[:, 1:].sum(axis=1, keepdims=True)
    sizes = np.maximum(sizes + delta[:, 1:], 1e-5)
    sizes /= sizes.sum(axis=1, keepdims=True)
    return np.column_stack([1 - win, win[:, None] * sizes])


def train_stack(dataset, mix, output, name, iterations=160):
    path = output / (name + '_predictions.npz')
    training_path = output / (name + '_training.json')
    if path.exists() and training_path.exists():
        return dict(np.load(path)), json.loads(training_path.read_text())
    start = time.perf_counter()
    train, cal, selection = (dataset['role'] == k for k in range(3))
    used = dataset['names'] != 'signal'
    x, qx = augment_features(dataset['x'], mix['p'], used), augment_features(dataset['qx'], mix['qp'], used)
    y = dataset['y']
    candidates, stats = {'mixer': mix['qp']}, {}
    stats['mixer'] = {'selection_logloss': log_loss(y[selection], mix['p'][selection], labels=np.arange(5))}
    kwargs = dict(max_iter=iterations, max_leaf_nodes=15, learning_rate=.06,
                  min_samples_leaf=120, l2_regularization=50, early_stopping=False, random_state=914)
    print('stack classifiers', name, x[train].shape, flush=True)
    win = HistGradientBoostingClassifier(**kwargs).fit(x[train], y[train] > 0)
    size = HistGradientBoostingClassifier(**kwargs).fit(x[train & (y > 0)], y[train & (y > 0)] - 1)
    cp, sp, qp = (head_predictions(win, size, a) for a in (x[cal], x[selection], qx))
    parameters = calibrate(cp, y[cal])
    for suffix, pp, ss in (('raw', qp, sp), ('cal', apply_calibration(qp, parameters), apply_calibration(sp, parameters))):
        key = 'classifier_' + suffix
        candidates[key] = pp
        stats[key] = {'selection_logloss': log_loss(y[selection], ss, labels=np.arange(5)), 'calibration': parameters}
        print(name, key, stats[key], flush=True)
    with (output / (name + '_classifier.pkl')).open('wb') as stream:
        pickle.dump((win, size, parameters, used), stream)
    print('stack residual regressors', name, flush=True)
    regressors = []
    delta = np.zeros((len(y), 5))
    qdelta = np.zeros((len(qx), 5))
    for k in range(5):
        mask = train if k == 0 else train & (y > 0)
        targets = ((y > 0).astype(float) - (1 - mix['p'][:, 0]) if k == 0 else
                   (y == k).astype(float) - mix['p'][:, k] / mix['p'][:, 1:].sum(axis=1))
        reg = HistGradientBoostingRegressor(**(kwargs | {'max_iter': 120, 'learning_rate': .04,
                                                        'min_samples_leaf': 200, 'l2_regularization': 100})).fit(x[mask], targets[mask])
        delta[:, k], qdelta[:, k] = reg.predict(x), reg.predict(qx)
        regressors.append(reg)
    for strength in (.25, .5, 1.):
        key = f'residual_{strength}'
        candidates[key] = residual_predictions(mix['qp'], strength * qdelta)
        sp = residual_predictions(mix['p'][selection], strength * delta[selection])
        stats[key] = {'selection_logloss': log_loss(y[selection], sp, labels=np.arange(5))}
        print(name, key, stats[key], flush=True)
    with (output / (name + '_residual.pkl')).open('wb') as stream:
        pickle.dump((regressors, used), stream)
    stats['_seconds'] = time.perf_counter() - start
    np.savez_compressed(path, **candidates)
    training_path.write_text(json.dumps(stats, indent=2) + '\n')
    return candidates, stats


class Evaluator:
    def __init__(self, data, ids, episode_folds):
        ts = {t.query_id: t for t in read_targets(data / 'validation/targets.csv')}
        ref = read_submission(data / 'validation/reference.csv')
        groups = read_conditions(data)
        self.truth = np.array([ts[k].probabilities for k in ids])
        self.ref = np.array([ref[k] for k in ids])
        self.episodes = np.array([ts[k].episode_id for k in ids])
        self.conditions = np.array([groups[k] for k in ids])
        self.fold = np.array([episode_folds[e] for e in self.episodes])
        self.weights = {}
        for name, mask in [('all', np.ones(len(ids), dtype=bool)), ('A', self.fold == 'A'), ('B', self.fold == 'B')]:
            w = np.zeros(len(ids))
            for c in CONDITIONS:
                selected_episodes = sorted(set(self.episodes[mask & (self.conditions == c)]))
                referr = np.mean([np.square(self.ref[self.episodes == e] - self.truth[self.episodes == e]).sum(axis=1).mean()
                                  for e in selected_episodes])
                for e in selected_episodes:
                    idx = self.episodes == e
                    w[idx] = 1 / (4 * len(selected_episodes) * idx.sum() * max(1e-9, referr))
            self.weights[name] = w

    def ratio(self, p, fold='all'):
        return float(np.square(p - self.truth).sum(axis=1) @ self.weights[fold])

    def report(self, p):
        per_condition = {}
        rng = np.random.default_rng(1914)
        boot_ratios = []
        for c in CONDITIONS:
            errors, refs = [], []
            for e in sorted(set(self.episodes[self.conditions == c])):
                ix = self.episodes == e
                errors.append(np.square(p[ix] - self.truth[ix]).sum(axis=1).mean())
                refs.append(np.square(self.ref[ix] - self.truth[ix]).sum(axis=1).mean())
            per_condition[c] = float(np.mean(errors) / np.mean(refs))
            sample = rng.integers(0, len(errors), (4000, len(errors)))
            boot_ratios.append(np.asarray(errors)[sample].mean(axis=1) / np.asarray(refs)[sample].mean(axis=1))
        return {'points': 60 / (1 + self.ratio(p)), 'ratios': per_condition,
                'ci95': np.quantile(60 / (1 + np.mean(boot_ratios, axis=0)), [.025, .975]).tolist(),
                'A': 60 / (1 + self.ratio(p, 'A')), 'B': 60 / (1 + self.ratio(p, 'B'))}

    def paired_gain_ci(self, p, baseline):
        rng = np.random.default_rng(1914)
        model_ratios, base_ratios = [], []
        for c in CONDITIONS:
            model, base, refs = [], [], []
            for e in sorted(set(self.episodes[self.conditions == c])):
                ix = self.episodes == e
                model.append(np.square(p[ix] - self.truth[ix]).sum(axis=1).mean())
                base.append(np.square(baseline[ix] - self.truth[ix]).sum(axis=1).mean())
                refs.append(np.square(self.ref[ix] - self.truth[ix]).sum(axis=1).mean())
            sample = rng.integers(0, len(model), (4000, len(model)))
            denominator = np.asarray(refs)[sample].mean(axis=1)
            model_ratios.append(np.asarray(model)[sample].mean(axis=1) / denominator)
            base_ratios.append(np.asarray(base)[sample].mean(axis=1) / denominator)
        gain = 60 / (1 + np.mean(model_ratios, axis=0)) - 60 / (1 + np.mean(base_ratios, axis=0))
        return np.quantile(gain, [.025, .975]).tolist()


def blend(base, stack, win_weight, size_weight):
    bw, sw = 1 - base[:, 0], 1 - stack[:, 0]
    win = (1 - win_weight) * bw + win_weight * sw
    sizes = ((1 - size_weight) * base[:, 1:] / bw[:, None]
             + size_weight * stack[:, 1:] / sw[:, None])
    return np.column_stack([1 - win, win[:, None] * sizes])


def evaluate(data, output, dataset, plans, scheme, full=None):
    vq, _ = load_inputs(data / 'validation')
    conditions = read_conditions(data)
    prior_folds = folds(vq, conditions, hashed=scheme == 'hashed')
    vm = dataset['splits'] == 'validation'
    ids = dataset['ids'][vm]
    query_episode = {q.query_id: q.episode_id for q in vq}
    prior_a = np.array([prior_folds[query_episode[k]] == 'A' for k in ids])
    plan_a = {key: p[vm] for key, p in plans[scheme + '_A'][0].items()}
    plan_b = {key: p[vm] for key, p in plans[scheme + '_B'][0].items()}
    candidates = {key: np.where(prior_a[:, None], plan_a[key], plan_b[key])
                  for key in plans[scheme + '_A'][0]}
    # Sorted is the brief's primary protocol. Hash folds are a separately
    # labelled historical-protocol companion, never mixed into the headline.
    ev = Evaluator(data, ids, prior_folds)
    keys = [k for k in candidates if k != 'mixer']
    # Plan B uses priors from A for ALL rows. Tune on A with that same
    # plan, then score B. Choosing on already-cross-fitted A forecasts
    # would use B-derived priors, indirectly leaking B into model choice.
    chosen_a = min(keys, key=lambda k: ev.ratio(plan_b[k], 'A'))
    chosen_b = min(keys, key=lambda k: ev.ratio(plan_a[k], 'B'))
    stack = np.where((ev.fold == 'A')[:, None], candidates[chosen_b], candidates[chosen_a])
    combinations = [(key, ww, ws) for key in keys for ww in (0., .1, .25, .5, .75, 1.)
                    for ws in (0., .1, .25, .5, .75, 1.)]
    blend_a = min(combinations, key=lambda v: ev.ratio(blend(plan_b['mixer'], plan_b[v[0]], v[1], v[2]), 'A'))
    blend_b = min(combinations, key=lambda v: ev.ratio(blend(plan_a['mixer'], plan_a[v[0]], v[1], v[2]), 'B'))
    bp_a = blend(plan_b['mixer'], plan_b[blend_a[0]], blend_a[1], blend_a[2])
    bp_b = blend(plan_a['mixer'], plan_a[blend_b[0]], blend_b[1], blend_b[2])
    blended = np.where((ev.fold == 'A')[:, None], bp_b, bp_a)
    report = {'scheme': scheme, 'base': ev.report(candidates['mixer']), 'stacked_crossfit': ev.report(stack),
              'blend_crossfit': ev.report(blended), 'stack_selected_on_A_for_B': chosen_a,
              'stack_selected_on_B_for_A': chosen_b, 'blend_selected_on_A_for_B': blend_a,
              'blend_selected_on_B_for_A': blend_b, 'candidates': {k: ev.report(p) for k, p in candidates.items()}}
    report['selection_protocol'] = 'For each outer scored fold, fitting, calibration and selection use only opposite-fold target-derived priors.'
    report['blend_gain_vs_mixer_ci95'] = ev.paired_gain_ci(blended, candidates['mixer'])
    conditional_train_a = min(plan_a, key=lambda k: plans[scheme + '_A'][1][k]['selection_logloss'])
    conditional_train_b = min(plan_b, key=lambda k: plans[scheme + '_B'][1][k]['selection_logloss'])
    report['train_label_choice_conditional_on_validation_priors'] = ev.report(np.where(prior_a[:, None], plan_a[conditional_train_a], plan_b[conditional_train_b]))
    report['conditional_train_choices'] = [conditional_train_a, conditional_train_b]
    previous = Path('runs/fresh-ml-repro/validation_crossfit.csv')
    if previous.exists():
        old = read_submission(previous)
        previous_blend = blend(candidates['mixer'], np.array([old[k] for k in ids]), .15, .15)
        report['fixed_previous_ml1_blend_015'] = ev.report(previous_blend)
        report['blend_gain_vs_previous_ci95'] = ev.paired_gain_ci(blended, previous_blend)
    if scheme == 'hashed':
        snapshot = Path('/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/_tm/mixer6/mainFiles/runs/m6/crossfit_validation_predictions.csv')
        original = read_submission(snapshot)
        report['snapshot_reference_max_abs_diff'] = float(np.abs(candidates['mixer'] - np.array([original[k] for k in ids])).max())
        assert report['snapshot_reference_max_abs_diff'] < 1e-8, report['snapshot_reference_max_abs_diff']
    for name, p in [('mixer', candidates['mixer']), ('stack_crossfit', stack), ('blend_crossfit', blended)]:
        write_submission(output / f'{scheme}_{name}_validation.csv', dict(zip(ids, p)))
    np.savez_compressed(output / f'{scheme}_validation_candidates.npz', **candidates)
    if full is not None:
        # Deployment choices use all released validation observations; the
        # reported score above remains the opposite-fold score.
        choice = min(combinations, key=lambda v: ev.ratio(blend(full[0]['mixer'][vm], full[0][v[0]][vm], v[1], v[2])))
        tm = dataset['splits'] == 'test'
        test = blend(full[0]['mixer'][tm], full[0][choice[0]][tm], choice[1], choice[2])
        write_submission(output / f'{scheme}_submission.csv', dict(zip(dataset['ids'][tm], test)))
        report['deployment_choice_all_validation'] = choice
    (output / f'{scheme}_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('STACK SUMMARY', scheme, json.dumps({k: report[k] for k in ('base', 'stacked_crossfit', 'blend_crossfit')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('public'))
    parser.add_argument('--out', type=Path, default=Path('runs/fresh-ml2'))
    parser.add_argument('--schemes', nargs='+', choices=['sorted', 'hashed'], default=['sorted', 'hashed'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        dataset = load_dataset(args.data, args.out, 180000)
        vq, _ = load_inputs(args.data / 'validation')
        targets = read_targets(args.data / 'validation/targets.csv')
        conditions = read_conditions(args.data)
        plans = {}
        if args.evaluate_only:
            for scheme in args.schemes:
                for fold in ('A', 'B'):
                    name = scheme + '_' + fold
                    plans[name] = (dict(np.load(args.out / (name + '_predictions.npz'))),
                                   json.loads((args.out / (name + '_training.json')).read_text()))
            full_path = args.out / 'full_predictions.npz'
            full = (dict(np.load(full_path)), json.loads((args.out / 'full_training.json').read_text())) if full_path.exists() else None
            for scheme in args.schemes:
                evaluate(args.data, args.out, dataset, plans, scheme, full)
            return
        for scheme in args.schemes:
            assignment = folds(vq, conditions, hashed=scheme == 'hashed')
            for scored_fold in ('A', 'B'):
                name = scheme + '_' + scored_fold
                priors = opposite_fold_targets(targets, assignment, scored_fold)
                mix = mixer_features(args.data, args.out, dataset, priors, name)
                if not args.prepare_only:
                    plans[name] = train_stack(dataset, mix, args.out, name)
            if not args.prepare_only:
                evaluate(args.data, args.out, dataset, plans, scheme)
        if not args.prepare_only:
            mix = mixer_features(args.data, args.out, dataset, targets, 'full')
            full = train_stack(dataset, mix, args.out, 'full')
            for scheme in args.schemes:
                evaluate(args.data, args.out, dataset, plans, scheme, full)
    runtime = {'seconds': time.perf_counter() - started,
               'peak_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024),
               'settings': vars(args) | {'data': str(args.data), 'out': str(args.out)}}
    runtime['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (args.out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    print('STACK RUNTIME', runtime, flush=True)


if __name__ == '__main__':
    main()
