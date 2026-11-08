"""Compare independent refits and enforce the entry's resource tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np

from slot_forecasting.io import read_submission


def verify(first, second):
    with np.load(first / 'candidate_predictions.npz') as left, np.load(second / 'candidate_predictions.npz') as right:
        assert set(left.files) == set(right.files)
        # Empirical v1 was saved at float32 precision before renormalization;
        # all learned candidates and the final CSVs must meet the 1e-8 rule.
        differences = {k: float(np.abs(left[k] - right[k]).max())
                       for k in left.files if not k.startswith('empirical_')}
    assert max(differences.values()) <= 1e-8, differences
    csv_differences = {}
    for filename in ('validation_crossfit.csv', 'validation_predictions.csv', 'submission.csv'):
        left, right = read_submission(first / filename), read_submission(second / filename)
        assert left.keys() == right.keys()
        csv_differences[filename] = max(abs(a - b) for k in left for a, b in zip(left[k], right[k]))
    assert max(csv_differences.values()) <= 1e-8, csv_differences
    runtime = json.loads((second / 'prediction_run.json').read_text())
    assert runtime['total_seconds'] <= 1800
    assert runtime['peak_process_mib'] <= 16384
    assert set(runtime['input_sha256']) == {
        'train/queries.csv', 'train/history.csv', 'train/labels.csv',
        'validation/queries.csv', 'validation/history.csv', 'test/queries.csv', 'test/history.csv'}
    report = {'learned_candidate_max_abs_diff': differences, 'csv_max_abs_diff': csv_differences,
              'full_fit_seconds': runtime['total_seconds'], 'peak_process_mib': runtime['peak_process_mib'],
              'prediction_inputs_exclude_validation_answers': True}
    (second / 'reproducibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    verify(args.first, args.second)
