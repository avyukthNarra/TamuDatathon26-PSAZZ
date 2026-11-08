"""Determinism, baseline reproduction and all saved probability contracts."""
import hashlib
import json
import numpy as np
from models import *
from harness import read_pred_csv
sys.path.insert(0,str(ROOT/'mainFiles'))
from slot_forecasting.io import read_submission


def main():
    v=load_split('validation')
    a=win_walk(v);b=win_walk(v)
    difference=max(float(abs(x-y).max()) for x,y in zip(a,b))
    size=size_walk(v);size2=size_walk(v)
    difference=max(difference,float(abs(size-size2).max()))
    assert difference==0.,difference
    p=np.maximum(np.column_stack((1-a[0],a[0][:,None]*size)),1e-4);p/=p.sum(1,keepdims=True)
    ref=read_pred_csv(HERE/'preds/TM_MIXER4_validation.csv',v)
    error=float(abs(p-ref).max());assert error<1e-10,error
    files={}
    for path in sorted((HERE/'preds').glob('BOOST_T[23]*_validation.csv')):
        pp=read_submission(path,v.qid);assert len(pp)==v.n
        files[path.name]={'rows':len(pp),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    result={'two_run_max_abs_diff':difference,'baseline_max_abs_diff':error,
            'n_csv_files':len(files),'files':files}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:x for k,x in result.items() if k!='files'}))

if __name__=='__main__':main()
