"""Strict CSV comparison and bounded, serial verification commands."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / 'mainFiles'
SAI = ROOT / 'docs' / 'sai'
REFERENCE = SAI / 'preds' / 'TM_COMBINED3_validation.csv'
COLUMNS = ('p_0', 'p_1', 'p_2', 'p_5', 'p_20')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS')
for name in THREAD_VARS:
    os.environ[name] = '1'


def read_csv(path):
    rows = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['query_id', *COLUMNS]:
            raise ValueError(f'{path}: expected query_id followed by {COLUMNS}')
        for row in reader:
            qid = row['query_id']
            if not qid or qid in rows or None in row:
                raise ValueError(f'{path}: empty/duplicate query_id or extra fields: {qid}')
            values = tuple(float(row[c]) for c in COLUMNS)
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
                raise ValueError(f'{path}: invalid probability for {qid}')
            if abs(sum(values) - 1) > 1e-6:
                raise ValueError(f'{path}: probabilities do not sum to 1 for {qid}')
            rows[qid] = values
    if not rows:
        raise ValueError(f'{path}: no predictions')
    return rows


def compare(actual, expected, tolerance):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    a, b = read_csv(actual), read_csv(expected)
    if a.keys() != b.keys():
        raise ValueError(f'query IDs differ: missing={len(b.keys() - a.keys())}, '
                         f'extra={len(a.keys() - b.keys())}')
    maxima = {c: max(abs(a[q][j] - b[q][j]) for q in a)
              for j, c in enumerate(COLUMNS)}
    result = {'rows': len(a), 'max_abs_difference': maxima,
              'maximum': max(maxima.values()), 'tolerance': tolerance}
    print(json.dumps(result, sort_keys=True), flush=True)
    if result['maximum'] >= tolerance:
        raise ValueError('prediction difference must be strictly below tolerance')
    return result


def reserve_audit(marker, log, prediction):
    """Consume permission before scoring; exclusive create also prevents concurrent access."""
    if log.exists() and any(line.split()[-1:] == ['combo-sai']
                            for line in log.read_text().splitlines()):
        raise RuntimeError('combo-sai already appears in audit_access.log; refusing repeat')
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'who': 'combo-sai', 'prediction': str(prediction),
                   'sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
                   'started_unix': time.time()}, stream, indent=2)
        stream.write('\n')


def score(path, scope):
    path = Path(path).resolve()
    read_csv(path)
    if scope == 'audit':
        reserve_audit(ROOT / '.pane' / 'combo-audit.once.json',
                      SAI / 'results' / 'audit_access.log', path)
    elif scope != 'dev':
        raise ValueError('only dev and audit scoring are supported')
    os.chdir(SAI)
    sys.path.insert(0, str(SAI))
    from harness import evaluate, load_split, read_pred_csv
    validation = load_split('validation')
    predictions = read_pred_csv(path, validation)
    result = evaluate(predictions, validation, scope,
                      who='combo-sai' if scope == 'audit' else None)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if scope == 'audit':
        output = ROOT / '.pane' / 'combo-audit.result.json'
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def run(command, timeout=600, memory_bytes=16 * 1024**3):
    # macOS rejects RLIMIT_AS here. Check resident memory every 250 ms instead.
    # Each command is serial; predict.py is a single process, with thread envs = 1.
    print('+ ' + ' '.join(map(str, command)), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(['nice', '-n', '10', *map(str, command)],
                               cwd=ROOT, start_new_session=True)
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                code = process.wait(timeout=min(.25, remaining))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break
            except subprocess.TimeoutExpired:
                rss = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                                     capture_output=True, text=True, check=False)
                if rss.returncode == 0 and rss.stdout.strip():
                    if int(rss.stdout.strip()) * 1024 > memory_bytes:
                        raise MemoryError(f'command exceeded {memory_bytes} bytes RSS')
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    elapsed = time.monotonic() - started
    print(f'runtime_seconds={elapsed:.3f}', flush=True)
    return elapsed
