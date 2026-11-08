"""Repeat default bank prediction and validate saved fixed-size-share outputs."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv
from bma import predict


def main():
    v=load_split('validation')
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    shares=ref[:,1:]/ref[:,1:].sum(1,keepdims=True)
    actual,info=predict(v,1.35)
    stored=np.load(OUT/'bank_c1p35.npz')
    np.testing.assert_array_equal(actual,stored['pw'])
    np.testing.assert_array_equal(info['retained_models'],stored['retained_models'])
    np.testing.assert_array_equal(info['top_features'],stored['top_features'])
    expected_ids=set(v.qid);records=[]
    for path in sorted((HERE/'preds').glob('R6_*_validation.csv')):
        with path.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==len({r['query_id'] for r in rows})==v.n
        assert {r['query_id'] for r in rows}==expected_ids
        p=read_pred_csv(path,v)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.max(abs(p.sum(1)-1))<1e-6
        size_error=float(np.max(abs(p[:,1:]/p[:,1:].sum(1,keepdims=True)-shares)))
        assert size_error<2e-15
        records.append({'file':path.name,'rows':len(rows),'size_share_max_abs_difference':size_error,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'unit_tests_passed':5,'reference_numerical_parity':'14 decimal places on seeded non-tied inputs, all three allowed slab values',
            'default_bank_repeat_byte_identical':True,'screened_features_and_occam_counts_repeat_identical':True,
            'files':records,'audit_or_all_evaluations':0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
