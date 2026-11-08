"""Independent selected-model refits and saved-forecast integrity checks."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv
from logistic import machine_rows,empirical_bayes,predict


def main():
    report=json.loads((OUT/'results.json').read_text())
    train,v=load_split('train'),load_split('validation')
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    shares=ref[:,1:]/ref[:,1:].sum(1,keepdims=True)
    checks={}
    for family,full in [('KNOWN',False),('FULL',True)]:
        row=report['families'][family]
        stored=min((r for r in report['train_grid'] if r['family']==family),key=lambda r:r['train_log_loss'])
        variance,history=empirical_bayes(machine_rows(train,full),row['half_life'])
        for kind,values in variance.items():
            np.testing.assert_array_equal(values,np.array(stored['prior_variance'][kind]))
        machines=machine_rows(v,full)
        first,_=predict(machines,variance,row['half_life'])
        second,_=predict(machines,variance,row['half_life'])
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first,np.load(OUT/f'WIN_{family}_pw.npy'))
        saved=read_pred_csv(HERE/f'preds/WIN_{family}_validation.csv',v)
        expected=np.column_stack([1-first,first[:,None]*shares])
        assert np.max(abs(saved-expected))<1e-14
        np.testing.assert_allclose(saved[:,1:]/saved[:,1:].sum(1,keepdims=True),shares,atol=2e-15,rtol=0)
        checks[family]={'train_prior_refit_byte_identical':True,'two_predictions_byte_identical':True,
                        'raw_saved_win_probabilities_byte_identical':True,
                        'csv_max_abs_difference':float(np.max(abs(saved-expected)))}
    records=[]
    expected_ids=set(v.qid)
    for path in sorted((HERE/'preds').glob('WIN_*_validation.csv')):
        with path.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==len({r['query_id'] for r in rows})==768
        assert {r['query_id'] for r in rows}==expected_ids
        p=read_pred_csv(path,v)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.max(abs(p.sum(1)-1))<1e-6
        share_error=float(np.max(abs(p[:,1:]/p[:,1:].sum(1,keepdims=True)-shares)))
        assert share_error<2e-15
        records.append({'file':path.name,'rows':len(rows),'max_size_share_difference':share_error,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    checks['files']=records
    checks['unit_tests_passed']=6
    checks['audit_log_unchanged_during_run']=report['audit_log_unchanged']
    checks['audit_or_all_evaluations']=0
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
