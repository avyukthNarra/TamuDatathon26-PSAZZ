#!/usr/bin/env python3
"""Serial validation and determinism checks; never invokes audit scoring."""
import argparse
import json
from pathlib import Path
import sys
import time

from common import KIT, ROOT, compare, run

HERE = Path(__file__).resolve().parent


def determinism(output, data):
    started = time.monotonic()
    runtimes = []
    files = [output / 'test_first.csv', output / 'test_second.csv']
    for path in files:
        remaining = 1800 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('30 minute determinism budget exhausted')
        runtimes.append(run([sys.executable, KIT / 'combo' / 'predict.py',
                             '--data', data, '--split', 'test', '--out', path],
                            timeout=min(600, remaining)))
    result = compare(*files, tolerance=1e-8)
    result.update(prediction_runtime_seconds=runtimes,
                  total_runtime_seconds=time.monotonic() - started)
    (output / 'determinism.json').write_text(json.dumps(result, indent=2) + '\n')
    return files[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('all', 'determinism'))
    parser.add_argument('--out', type=Path, default=ROOT / '.pane' / 'combo-verification')
    args = parser.parse_args()
    output = args.out.resolve()
    # Refuse to overwrite earlier evidence. Use a new directory for an intentional rerun.
    output.mkdir(parents=True, exist_ok=False)
    data = KIT / 'public'
    if args.mode == 'all':
        validation = output / 'validation.csv'
        run([sys.executable, KIT / 'combo' / 'predict.py', '--data', data,
             '--split', 'validation', '--out', validation])
        run([sys.executable, HERE / 'compare_to_tm.py', validation])
        run([sys.executable, HERE / 'score_dev.py', validation])
        run([sys.executable, KIT / 'check_submission.py', validation,
             '--data', data, '--split', 'validation'])
    test = determinism(output, data)
    if args.mode == 'all':
        run([sys.executable, KIT / 'check_submission.py', test,
             '--data', data, '--split', 'test'])


if __name__ == '__main__':
    main()
