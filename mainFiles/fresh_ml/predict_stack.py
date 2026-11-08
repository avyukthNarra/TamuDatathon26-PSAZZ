"""Checkpoint-only test prediction: no train files or validation answers needed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS
from .features import machine_features
from .run import apply_calibration, head_predictions
from .stack import augment_features, blend, normalized_curve, residual_predictions


def predict(data, model_dir, choice, output):
    with (model_dir / 'full_mixer.pkl').open('rb') as stream:
        mixer = pickle.load(stream)
    key, win_weight, size_weight = choice
    with (model_dir / ('full_classifier.pkl' if key.startswith('classifier') else 'full_residual.pkl')).open('rb') as stream:
        saved = pickle.load(stream)
    used = saved[-1]
    queries, history = load_inputs(data / 'test')
    episodes = defaultdict(list)
    sequences = {}
    for q in queries:
        episodes[q.episode_id].append(q)
        sequences[q.episode_id, q.machine_id] = np.full(q.cutoff + 1, -1, dtype=np.int8)
    for play in history:
        sequences[play.episode_id, play.machine_id][play.tick] = PAYOUTS.index(play.payout)
    qids, features, mixer_probs = [], [], []
    for _, qs in sorted(episodes.items()):
        assert len({q.cutoff for q in qs}) == 1
        matrix = np.stack([sequences[q.episode_id, q.machine_id] for q in qs])
        for k, q in enumerate(qs):
            x, _, _ = machine_features(matrix[k], np.delete(matrix, k, axis=0), q.machine_type, q.signal)
            features.append(x[q.cutoff])
            own = [None if v < 0 else int(v) for v in matrix[k]]
            mixer_probs.append(normalized_curve(mixer, own, q.machine_type, q.cutoff)[q.cutoff])
            qids.append(q.query_id)
    base = np.stack(mixer_probs)
    x = augment_features(np.stack(features), base, used)
    if key.startswith('classifier'):
        win, size, calibration, _ = saved
        stack = head_predictions(win, size, x)
        if key.endswith('_cal'):
            stack = apply_calibration(stack, calibration)
    else:
        models, _ = saved
        delta = np.column_stack([model.predict(x) for model in models])
        stack = residual_predictions(base, float(key.rsplit('_', 1)[1]) * delta)
    final = blend(base, stack, win_weight, size_weight)
    write_submission(output, dict(zip(qids, final)))
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scheme', choices=['sorted', 'hashed'], default='sorted')
    args = parser.parse_args()
    choice = json.loads((args.model_dir / (args.scheme + '_results.json')).read_text())['deployment_choice_all_validation']
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        predict(args.data, args.model_dir, choice, args.out)
    stats = {'prediction_seconds': time.perf_counter() - start,
             'peak_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024),
             'choice': choice, 'scheme': args.scheme, 'input_files': ['test/queries.csv', 'test/history.csv']}
    args.out.with_suffix('.runtime.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
