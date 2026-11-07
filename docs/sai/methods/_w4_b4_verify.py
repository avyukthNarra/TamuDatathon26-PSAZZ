"""Reproduce each B4 learner artifact with the shared harness on DEV only."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SAI = HERE.parent
sys.path.insert(0, str(SAI))
from harness import evaluate, diff_ci, read_pred_csv
from _w4_b4_features import load_data, validate_causality


def verify(mid):
    data = load_data()
    v, base = data['v'], data['baseline']
    path = SAI / 'results' / f'{mid}.json'
    doc = json.loads(path.read_text())
    prediction = SAI / 'preds' / f'{mid}_validation.csv'
    with prediction.open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20']
        rows = list(reader)
    assert [r['query_id'] for r in rows] == list(v.qid)
    assert len(set(r['query_id'] for r in rows)) == v.n
    p = read_pred_csv(prediction, v)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert np.max(np.abs(p.sum(1)-1)) < 1e-6
    ev = evaluate(p, v, 'dev')
    for key in ('points', 'R', 'ci95'):
        np.testing.assert_allclose(ev[key], doc[key], rtol=0, atol=1e-10)
    for key in ('condition_ratios', 'class_error_shares'):
        for label, value in ev[key].items():
            assert abs(value-doc[key][label]) < 1e-10
    paired = diff_ci(p, base, v, 'dev')
    np.testing.assert_allclose(paired['ci95'], doc['extra']['standalone_vs_B4']['ci95'], rtol=0, atol=1e-10)
    assert abs(paired['diff']-doc['extra']['standalone_vs_B4']['diff']) < 1e-10
    assert doc['determinism_max_diff'] < 1e-8
    assert doc['n_episodes'] == 72 and doc['scope'] == 'dev'
    grid = json.loads((SAI/'grids'/f'{mid}.json').read_text())
    completed = [r for r in grid if 'points' in r]
    assert len(completed) == doc['extra']['completed_cells'] == doc['extra']['declared_cells']
    assert len({tuple(r['config'][k] for k in ('signal','pairs','prior','density')) for r in completed}) == 16
    if mid == 'S2_B4':
        assert {r['model']['family'] for r in completed} == {'cat', 'light'}
    if mid == 'S5_B4':
        from _w4_b4_learners import nested_blend
        nested = read_pred_csv(SAI/'preds'/f'{mid}_NESTED_validation.csv', v)
        expected = doc['extra']['nested_fixed_selected_hyperparameters']
        ne = evaluate(nested, v, 'dev')
        assert abs(ne['points']-expected['evaluation']['points']) < 1e-10
        np.testing.assert_allclose(ne['ci95'], expected['evaluation']['ci95'], rtol=0, atol=1e-10)
        actual, weights = nested_blend(data, doc['params']['config'], doc['params']['model'], doc['params']['strength'])
        det = float(np.abs(nested-actual).max())
        assert det < 1e-8
        expected['determinism_max_diff'] = det
    assert len((SAI/'reports'/f'{mid}.md').read_text().splitlines()) <= 15
    for pth in (HERE/f'{mid}.py', HERE/'_w4_b4_features.py', HERE/'_w4_b4_learners.py'):
        compile(pth.read_text(), str(pth), 'exec')
    leakage = validate_causality()
    doc['extra']['verification'] = dict(prediction_rows=v.n, unique_ids=True,
        probability_bounds_and_sum=True, saved_scores_and_intervals_reproduced=True,
        completed_grid_cells=len(completed), unique_feature_configs=16,
        leakage_checks=leakage, audit_scored=False)
    path.write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(id=mid, points=ev['points'], paired_vs_B4=paired,
        grid_cells=len(completed), predictions=v.n, validation='passed')))


if __name__ == '__main__':
    verify(sys.argv[1])
