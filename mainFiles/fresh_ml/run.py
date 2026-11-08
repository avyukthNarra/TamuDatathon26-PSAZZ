"""Reproduce independent self-supervised global-learning experiments.

Run from mainFiles: OMP_NUM_THREADS=2 python -m fresh_ml.run --out runs/fresh-ml
"""
import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import pickle
import resource
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_submission, read_targets, write_submission
from slot_forecasting.schema import CONDITIONS, PAYOUTS

from .features import machine_features


def load_dataset(data, output, max_examples):
    cache = output / 'features.npz'
    input_files = [data / split / name for split in ('train', 'validation', 'test')
                   for name in ('queries.csv', 'history.csv')] + [data / 'train/labels.csv']
    signature = json.dumps({'max_examples': max_examples, 'sampler_version': 2,
                            'feature_source': hashlib.sha256(Path(__file__).with_name('features.py').read_bytes()).hexdigest(),
                            'inputs': {str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}}, sort_keys=True)
    if cache.exists():
        cached = dict(np.load(cache, allow_pickle=False))
        if str(cached.get('signature', '')) == signature:
            return cached
    rng = np.random.default_rng(914)
    all_x, all_y, roles, query_x, query_ids, query_splits = [], [], [], [], [], []
    row_query_indices, row_ticks = [], []
    metadata = {}
    names = groups = None
    # 24 train episodes (stratified by their existing sorted order) excluded
    # from global fitting; half for calibration and half for train-only choice.
    train_q, _ = load_inputs(data / 'train')
    train_episodes = sorted({q.episode_id for q in train_q})
    heldout = train_episodes[::4]
    calibration, selection = set(heldout[::2]), set(heldout[1::2])
    for split in ('train', 'validation', 'test'):
        queries, history = load_inputs(data / split)
        labels = read_labels(data / split / 'labels.csv', queries) if split == 'train' else {}
        episodes = defaultdict(list)
        for q in queries:
            episodes[q.episode_id].append(q)
        lookup = {(q.episode_id, q.machine_id): q for q in queries}
        arrays = {key: np.full(q.cutoff + 1, -1, dtype=np.int8) for key, q in lookup.items()}
        for p in history:
            arrays[p.episode_id, p.machine_id][p.tick] = PAYOUTS.index(p.payout)
        split_rows = 0
        for episode, qs in sorted(episodes.items()):
            assert len({q.cutoff for q in qs}) == 1
            matrix = np.stack([arrays[q.episode_id, q.machine_id] for q in qs])
            for k, q in enumerate(qs):
                sequence = matrix[k]
                x, names, groups = machine_features(sequence, np.delete(matrix, k, axis=0), q.machine_type, q.signal)
                query_x.append(x[q.cutoff])
                query_ids.append(q.query_id)
                query_splits.append(split)
                y = sequence.copy()
                if q.query_id in labels:
                    y[q.cutoff] = PAYOUTS.index(labels[q.query_id])
                eligible = (y >= 0) & (np.arange(len(y)) >= 32)
                role = 1 if episode in calibration else 2 if episode in selection else 0
                if role:
                    eligible &= np.arange(len(y)) >= 128
                else:
                    # Deterministic uniform sample. No outcomes affect sampling.
                    eligible &= rng.random(len(y)) < 0.5
                all_x.append(x[eligible])
                all_y.append(y[eligible])
                roles.append(np.full(eligible.sum(), role, dtype=np.int8))
                row_query_indices.append(np.full(eligible.sum(), len(query_ids) - 1, dtype=np.int32))
                row_ticks.append(np.flatnonzero(eligible).astype(np.int16))
                split_rows += int(eligible.sum())
        metadata[split] = {'episodes': len(episodes), 'queries': len(queries),
                           'history_rows': len(history), 'retained_examples': split_rows}
        print('features', split, metadata[split], flush=True)
    x, y, role = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(roles)
    row_query_indices, row_ticks = np.concatenate(row_query_indices), np.concatenate(row_ticks)
    if (role == 0).sum() > max_examples:
        keep = np.ones(len(y), dtype=bool)
        fit_indices = np.flatnonzero(role == 0)
        keep[fit_indices] = False
        keep[rng.choice(fit_indices, max_examples, replace=False)] = True
        x, y, role = x[keep], y[keep], role[keep]
        row_query_indices, row_ticks = row_query_indices[keep], row_ticks[keep]
    result = dict(x=x, y=y, role=role, qx=np.stack(query_x), ids=np.array(query_ids),
                  splits=np.array(query_splits), names=np.array(names), groups=np.array(groups),
                  metadata=np.array(json.dumps(metadata)),
                  signature=np.array(signature),
                  row_query_indices=row_query_indices, row_ticks=row_ticks,
                  calibration_episodes=np.array(sorted(calibration)), selection_episodes=np.array(sorted(selection)))
    np.savez_compressed(cache, **result)
    return result


def calibrate(p, y):
    """Two-parameter win calibration and positive-head temperature; train only."""
    z = logit(np.clip(1 - p[:, 0], 1e-6, 1 - 1e-6))
    target = y > 0
    def win_objective(ab):
        logits = ab[0] * z + ab[1]
        return np.mean(np.logaddexp(0, logits) - target * logits) + 1e-4 * (ab[0] - 1) ** 2
    ab = minimize(win_objective, [1., 0.], bounds=[(0.25, 2), (-2, 2)], method='L-BFGS-B').x
    sizes = p[target, 1:] / p[target, 1:].sum(axis=1, keepdims=True)
    logs = np.log(np.clip(sizes, 1e-9, 1))
    def size_objective(v):
        return log_loss(y[target] - 1, softmax(logs / v[0], axis=1), labels=np.arange(4))
    temperature = float(minimize(size_objective, [1.], bounds=[(0.5, 2)], method='L-BFGS-B').x[0])
    return [float(ab[0]), float(ab[1]), temperature]


def apply_calibration(p, parameters):
    a, b, t = parameters
    win = expit(a * logit(np.clip(1 - p[:, 0], 1e-7, 1 - 1e-7)) + b)
    sizes = softmax(np.log(np.clip(p[:, 1:], 1e-10, 1)) / t, axis=1)
    return np.column_stack([1 - win, win[:, None] * sizes])


def head_predictions(win_model, size_model, x):
    win = win_model.predict_proba(x)[:, 1]
    size = size_model.predict_proba(x)
    return np.column_stack([1 - win, win[:, None] * size])


def empirical(qx, names, window):
    indices = [list(names).index(f'self{window}_p{k}') for k in PAYOUTS]
    result = qx[:, indices].astype(float)
    return result / result.sum(axis=1, keepdims=True)


def train_models(dataset, output, iterations):
    x, y, role, qx = (dataset[k] for k in ('x', 'y', 'role', 'qx'))
    train, calibration, selection = (role == k for k in range(3))
    predictions, results = {}, {}
    for window in ('all', '128', '48', '16'):
        name = 'empirical_' + window
        p = empirical(x, dataset['names'], window)
        predictions[name] = empirical(qx, dataset['names'], window)
        results[name] = {'selection_logloss': log_loss(y[selection], p[selection], labels=np.arange(5))}
    for mode in ('base', 'context'):
        features = dataset['groups'] != 'context' if mode == 'base' else np.ones(x.shape[1], dtype=bool)
        # signal's timing is unspecified for historical pseudo-queries, so it
        # is excluded from every model rather than assumed historical input.
        features &= dataset['names'] != 'signal'
        for leaves in ((15,) if mode == 'base' else (15, 31)):
            name = f'hgb_{mode}_{leaves}'
            start = time.perf_counter()
            fitx = x[train][:, features]
            fity = y[train]
            kwargs = dict(max_iter=iterations, max_leaf_nodes=leaves, learning_rate=0.075,
                          min_samples_leaf=80, l2_regularization=30, early_stopping=False, random_state=914)
            win_model = HistGradientBoostingClassifier(**kwargs)
            size_model = HistGradientBoostingClassifier(**kwargs)
            print('fit', name, fitx.shape, flush=True)
            win_model.fit(fitx, fity > 0)
            size_model.fit(fitx[fity > 0], fity[fity > 0] - 1)
            cp = head_predictions(win_model, size_model, x[calibration][:, features])
            sp = head_predictions(win_model, size_model, x[selection][:, features])
            qp = head_predictions(win_model, size_model, qx[:, features])
            params = calibrate(cp, y[calibration])
            for suffix, pp, ss in (('raw', qp, sp), ('cal', apply_calibration(qp, params), apply_calibration(sp, params))):
                candidate = name + '_' + suffix
                predictions[candidate] = pp
                results[candidate] = {'selection_logloss': log_loss(y[selection], ss, labels=np.arange(5)),
                                      'calibration': params, 'seconds': time.perf_counter() - start}
                print(candidate, results[candidate], flush=True)
            with (output / (name + '.pkl')).open('wb') as stream:
                pickle.dump((win_model, size_model, features, params), stream)
            np.savez_compressed(output / 'candidate_predictions.npz', **predictions)
            (output / 'training.json').write_text(json.dumps(results, indent=2) + '\n')
    return predictions, results


def train_mlp(dataset, output, predictions, results):
    """A multiclass MLP tests smooth combinations of in-context evidence."""
    train, calibration, selection = (dataset['role'] == k for k in range(3))
    features = dataset['names'] != 'signal'
    x, y, qx = (dataset[k] for k in ('x', 'y', 'qx'))
    scaler = StandardScaler().fit(x[train][:, features])
    tx = scaler.transform(x[train][:, features])
    cx = scaler.transform(x[calibration][:, features])
    sx = scaler.transform(x[selection][:, features])
    qx = scaler.transform(qx[:, features])
    model = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=10,
                          batch_size=512, learning_rate_init=0.001,
                          random_state=914, solver='adam', max_iter=1)
    start = time.perf_counter()
    for epoch in range(1, 46):
        model.partial_fit(tx, y[train], classes=np.arange(5))
        if epoch not in (10, 25, 45):
            continue
        params = calibrate(model.predict_proba(cx), y[calibration])
        qp, sp = model.predict_proba(qx), model.predict_proba(sx)
        for suffix, pp, ss in (('raw', qp, sp), ('cal', apply_calibration(qp, params), apply_calibration(sp, params))):
            candidate = f'mlp_{epoch}_{suffix}'
            predictions[candidate] = pp
            results[candidate] = {'selection_logloss': log_loss(y[selection], ss, labels=np.arange(5)),
                                  'calibration': params, 'seconds': time.perf_counter() - start}
            print(candidate, results[candidate], flush=True)
        with (output / f'mlp_{epoch}.pkl').open('wb') as stream:
            pickle.dump((model, scaler, features, params), stream)
        np.savez_compressed(output / 'candidate_predictions.npz', **predictions)
        (output / 'training.json').write_text(json.dumps(results, indent=2) + '\n')
    return predictions, results


def extend_boosting(dataset, output, predictions, results):
    """Select additional training duration using only train-episode labels."""
    train, calibration, selection = (dataset['role'] == k for k in range(3))
    x, y, qx = (dataset[k] for k in ('x', 'y', 'qx'))
    with (output / 'hgb_context_31.pkl').open('rb') as stream:
        win_model, size_model, features, _ = pickle.load(stream)
    tx, ty = x[train][:, features], y[train]
    for iterations in (320, 640):
        start = time.perf_counter()
        for model in (win_model, size_model):
            model.set_params(warm_start=True, max_iter=iterations)
        win_model.fit(tx, ty > 0)
        size_model.fit(tx[ty > 0], ty[ty > 0] - 1)
        params = calibrate(head_predictions(win_model, size_model, x[calibration][:, features]), y[calibration])
        qp = head_predictions(win_model, size_model, qx[:, features])
        sp = head_predictions(win_model, size_model, x[selection][:, features])
        for suffix, pp, ss in (('raw', qp, sp), ('cal', apply_calibration(qp, params), apply_calibration(sp, params))):
            candidate = f'hgb_context_31_iter{iterations}_{suffix}'
            predictions[candidate] = pp
            results[candidate] = {'selection_logloss': log_loss(y[selection], ss, labels=np.arange(5)),
                                  'calibration': params, 'additional_seconds': time.perf_counter() - start}
            print(candidate, results[candidate], flush=True)
        with (output / f'hgb_context_31_iter{iterations}.pkl').open('wb') as stream:
            pickle.dump((win_model, size_model, features, params), stream)
        np.savez_compressed(output / 'candidate_predictions.npz', **predictions)
        (output / 'training.json').write_text(json.dumps(results, indent=2) + '\n')
    return predictions, results


def evaluation(data, output, dataset, predictions, training):
    # Only this stage can open validation targets. No features or model
    # fitting function takes this information as an argument.
    queries, history = load_inputs(data / 'validation')
    targets = read_targets(data / 'validation/targets.csv')
    groups = {r['query_id']: r['condition'] for r in csv.DictReader((data / 'validation/conditions.csv').open())}
    reference = read_submission(data / 'validation/reference.csv')
    target_map = {t.query_id: t for t in targets}
    trainq, trainh = load_inputs(data / 'train')
    trainlabels = read_labels(data / 'train/labels.csv', trainq)
    pooled = FrequencyForecaster('pooled').fit(trainq, trainh, trainlabels).predict(queries, history)
    mask = dataset['splits'] == 'validation'
    ids = dataset['ids'][mask]
    candidates = {name: dict(zip(ids, p[mask])) for name, p in predictions.items()}
    candidates['pooled'] = pooled
    candidate_names = list(predictions)  # pooled is an anchor, not selectable
    episodes_by_condition = {c: sorted({t.episode_id for t in targets if groups[t.query_id] == c}) for c in CONDITIONS}
    fold_a = {e for c in CONDITIONS for e in episodes_by_condition[c][::2]}
    fold_b = {e for c in CONDITIONS for e in episodes_by_condition[c][1::2]}

    def score(pred, episodes=None):
        selected = [t for t in targets if episodes is None or t.episode_id in episodes]
        selected_ids = {t.query_id for t in selected}
        return contest_score({k: pred[k] for k in selected_ids}, selected,
                             {k: groups[k] for k in selected_ids}, {k: reference[k] for k in selected_ids}, max_points=60)

    table = {name: {'all': score(p), 'A': score(p, fold_a), 'B': score(p, fold_b)} for name, p in candidates.items()}
    chosen_a = min(candidate_names, key=lambda n: table[n]['A']['mean_loss_ratio'])
    chosen_b = min(candidate_names, key=lambda n: table[n]['B']['mean_loss_ratio'])
    crossfit = {t.query_id: candidates[chosen_b if t.episode_id in fold_a else chosen_a][t.query_id] for t in targets}
    train_choice = min(training, key=lambda n: training[n]['selection_logloss'])

    def interval(pred):
        rng = np.random.default_rng(1914)
        ratios = []
        for c in CONDITIONS:
            error, referror = [], []
            for episode in episodes_by_condition[c]:
                ts = [t for t in targets if t.episode_id == episode]
                error.append(np.mean([np.square(np.asarray(pred[t.query_id]) - t.probabilities).sum() for t in ts]))
                referror.append(np.mean([np.square(np.asarray(reference[t.query_id]) - t.probabilities).sum() for t in ts]))
            sample = rng.integers(0, len(error), (4000, len(error)))
            ratios.append(np.asarray(error)[sample].mean(axis=1) / np.maximum(1e-9, np.asarray(referror)[sample].mean(axis=1)))
        return np.quantile(60 / (1 + np.mean(ratios, axis=0)), [.025, .975]).tolist()

    report = {'crossfit': score(crossfit), 'crossfit_ci95': interval(crossfit),
              'selected_on_A_for_B': chosen_a, 'selected_on_B_for_A': chosen_b,
              'train_only_choice': train_choice, 'train_only': table[train_choice]['all'],
              'train_only_ci95': interval(candidates[train_choice]),
              'pooled': table['pooled'], 'candidates': table,
              'fitting_examples': int((dataset['role'] == 0).sum()),
              'calibration_examples': int((dataset['role'] == 1).sum()),
              'selection_examples': int((dataset['role'] == 2).sum()),
              'features': len(dataset['names']), 'data': json.loads(str(dataset['metadata'])),
              'train_only_rule': 'Min log loss on held-out train episodes, after calibration on disjoint train episodes.',
              'bootstrap': '4000 fixed-choice stratified episode resamples, model and reference resampled together.'}
    write_submission(output / 'validation_crossfit.csv', crossfit)
    write_submission(output / 'validation_predictions.csv', candidates[train_choice])
    testmask = dataset['splits'] == 'test'
    write_submission(output / 'submission.csv', dict(zip(dataset['ids'][testmask], predictions[train_choice][testmask])))
    if chosen_a == chosen_b:
        write_submission(output / 'submission_validation_choice.csv',
                         dict(zip(dataset['ids'][testmask], predictions[chosen_a][testmask])))
        report['validation_choice_test_candidate'] = chosen_a
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SUMMARY', json.dumps({k: report[k] for k in ('crossfit_ci95', 'train_only_choice', 'selected_on_A_for_B', 'selected_on_B_for_A')}), flush=True)
    for name, value in table.items():
        print(name, round(value['all']['technical_points'], 5),
              {c: round(value['all']['by_condition'][c]['loss_ratio'], 5) for c in CONDITIONS}, flush=True)
    return report


def predict_without_targets(output, dataset, predictions, training):
    """Submission path: training labels choose the model; no scoring files."""
    chosen = min(training, key=lambda n: training[n]['selection_logloss'])
    for split, filename in (('validation', 'validation_predictions.csv'), ('test', 'submission.csv')):
        mask = dataset['splits'] == split
        write_submission(output / filename, dict(zip(dataset['ids'][mask], predictions[chosen][mask])))
    return {'train_only_choice': chosen, 'training': training}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('public'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-examples', type=int, default=180000)
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--include-mlp', action='store_true')
    parser.add_argument('--resume-models', action='store_true')
    parser.add_argument('--extend-boosting', action='store_true')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    with threadpool_limits(limits=2):
        dataset = load_dataset(args.data, args.out, args.max_examples)
        if args.evaluate_only or args.resume_models:
            predictions = dict(np.load(args.out / 'candidate_predictions.npz'))
            training = json.loads((args.out / 'training.json').read_text())
        else:
            predictions, training = train_models(dataset, args.out, args.iterations)
        if args.include_mlp and not args.evaluate_only:
            predictions, training = train_mlp(dataset, args.out, predictions, training)
        if args.extend_boosting and not args.evaluate_only:
            predictions, training = extend_boosting(dataset, args.out, predictions, training)
        if args.predict_only:
            report = predict_without_targets(args.out, dataset, predictions, training)
        else:
            report = evaluation(args.data, args.out, dataset, predictions, training)
    report['total_seconds'] = time.perf_counter() - start
    report['settings'] = vars(args) | {'data': str(args.data), 'out': str(args.out)}
    report['source_sha256'] = source_hashes
    input_paths = [p for p in args.data.glob('*/*.csv')
                   if not args.predict_only or p.name in ('queries.csv', 'history.csv')
                   or p == args.data / 'train/labels.csv']
    report['input_sha256'] = {str(p.relative_to(args.data)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    report['versions'] = {'python': platform.python_version(),
                          **{p: version(p) for p in ('numpy', 'scipy', 'scikit-learn')}}
    report['peak_process_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
    (args.out / ('prediction_run.json' if args.predict_only else 'results.json')).write_text(json.dumps(report, indent=2) + '\n')
    print('total seconds', report['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
