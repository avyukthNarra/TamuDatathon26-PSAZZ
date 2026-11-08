"""Validate exact-size determinism, all CSVs, and the unchanged win head."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'docs/sai'),str(ROOT/'mainFiles')]
from harness import load_split
from slot_forecasting.io import read_submission
from exact_size import ExactSizeSwitch,train_big_win_prior


def main():
    train,v=load_split('train'),load_split('validation')
    meta=json.loads((OUT/'baseline.json').read_text());cache=np.load(OUT/'baseline_arrays.npz')
    model=ExactSizeSwitch(meta['size_modes'],train_big_win_prior(train))
    a=model.predict(v);b=model.predict(v)
    difference=float(abs(a-b).max());assert difference==0.
    saved=np.load(OUT/'exact_sizes.npz')
    assert np.array_equal(saved['qid'],v.qid)
    reproduction=float(abs(a-saved['sizes']).max());assert reproduction==0.
    base=read_submission(ROOT/'docs/sai/preds/R5_BASE_validation.csv',v.qid)
    files={};win_error=0.
    for path in sorted((ROOT/'docs/sai/preds').glob('R5_*_validation.csv')):
        pp=read_submission(path,v.qid)
        assert len(pp)==v.n
        error=max(abs(base[q][0]-pp[q][0]) for q in pp)
        win_error=max(win_error,error);assert error<1e-14,error
        files[path.name]={'rows':len(pp),'p0_max_diff':error,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report={'two_run_exact_sizes_max_diff':difference,'saved_sizes_max_diff':reproduction,
            'all_csv_p0_max_diff':win_error,'csv_count':len(files),'files':files}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))

if __name__=='__main__':main()
