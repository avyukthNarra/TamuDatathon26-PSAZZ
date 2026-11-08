"""Verify saved sparse forecasts, dense CSV row bytes, and deterministic reruns."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'docs/sai'
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv
from sparse import forecast,gapped


def main():
    out=Path(__file__).resolve().parent
    d=json.loads((out/'results.json').read_text())
    v=load_split('validation');mask=gapped(v)
    baseline=read_pred_csv(HERE/'preds/TM_MIXER4_validation.csv',v)
    rule=read_pred_csv(HERE/'preds/TM_RULE_validation.csv',v)
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    raw=np.load(HERE/'improve/model/v3_raw.npy')
    sizes=raw[:,1:]/raw[:,1:].sum(1,keepdims=True)
    selected=next(r for r in d['candidates'] if r['id']==d['best'])
    def rerun():
        p=.2*rule+.8*forecast(v,baseline,sizes,d['inherited_prior'],**selected['params'])
        p[~mask]=ref[~mask]
        return p
    first,second=rerun(),rerun()
    assert first.tobytes()==second.tobytes()
    saved=read_pred_csv(HERE/'preds/BOOST_T1_BEST_validation.csv',v)
    assert first.tobytes()==saved.tobytes()
    reference_lines=(HERE/'preds/TM_COMBINED4_validation.csv').read_bytes().splitlines(keepends=True)
    reference_rows={line.split(b',',1)[0]:line for line in reference_lines[1:]}
    dense_ids={str(q).encode() for q in v.qid[~mask]}
    ids=set(v.qid)
    records=[]
    for path in sorted((HERE/'preds').glob('BOOST_T1*_validation.csv')):
        rows=list(csv.DictReader(path.open()))
        assert len(rows)==len({r['query_id'] for r in rows})==v.n
        assert {r['query_id'] for r in rows}==ids
        p=read_pred_csv(path,v)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.max(abs(p.sum(1)-1))<1e-6
        dense=[line for line in path.read_bytes().splitlines(keepends=True)[1:]
               if line.split(b',',1)[0] in dense_ids]
        assert len(dense)==len(dense_ids)==576
        assert all(line==reference_rows[line.split(b',',1)[0]] for line in dense)
        records.append({'file':path.name,'rows':len(rows),'dense_identical_rows':len(dense),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    checks={'unit_tests_passed':6,'baseline_reproduction_max_abs_diff':d['baseline_reproduction_max_abs_diff'],
            'best_two_run_byte_identical':True,'best_saved_prediction_byte_identical':True,
            'dense_csv_rows_byte_identical_to_reference':True,'dense_rows_per_csv':576,
            'files':records,'audit_or_all_evaluations':0}
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({k:v for k,v in checks.items() if k!='files'},indent=2))
    print(len(records),'valid CSVs; dense rows preserved literally, including decimal text.')


if __name__=='__main__':main()
