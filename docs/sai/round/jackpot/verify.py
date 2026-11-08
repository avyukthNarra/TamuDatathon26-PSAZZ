"""Reproduce the train-prior change; validate forecasts and untouched other types."""
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'docs/sai';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'worktrees/_tm/mixer6/mainFiles'))
sys.path.insert(1,str(HERE))
from slot_forecasting.io import load_inputs,read_targets
from harness import load_split,read_pred_csv
from harness.scoring import scope_episodes
from mixer6_copy import PairMixer,size_modes_from_targets
from diagnose import metrics


def main():
    report=json.loads((OUT/'fix_results.json').read_text())
    v=load_split('validation');refpath=HERE/'round/mixer6/MIXER6_rule00_insample_validation.csv'
    ref=read_pred_csv(refpath,v);mask=v.mtype=='jackpot'
    q,h=load_inputs(ROOT/'mainFiles/public/train')
    vq,vh=load_inputs(ROOT/'mainFiles/public/validation')
    jq=tuple(x for x in vq if x.machine_type=='jackpot')
    keys={(x.episode_id,x.machine_id) for x in jq}
    jh=tuple(x for x in vh if (x.episode_id,x.machine_id) in keys)
    modes=size_modes_from_targets(read_targets(ROOT/'mainFiles/public/validation/targets.csv'),vq)
    priors=copy.deepcopy(report['baseline_priors'])
    weight=priors['jackpot'][1][0];prior=report['train_active_prior']
    priors['jackpot'][1]=(weight,prior['mean'],prior['strength'])
    def rerun():
        model=PairMixer(pairs=report['pairs'],size_keys=report['size_keys'],size_modes=modes,
                        win_prior=priors,eta=6.,cell_mode='parity').fit(q,h)
        prediction=model.predict(jq,jh);out=ref.copy()
        for i in np.flatnonzero(mask):out[i]=prediction[str(v.qid[i])]
        return out
    first,second=rerun(),rerun()
    np.testing.assert_array_equal(first,second)
    saved=read_pred_csv(HERE/'preds/J1_TRAIN_validation.csv',v)
    np.testing.assert_array_equal(first,saved)
    reference_rows={line.split(b',',1)[0]:line for line in refpath.read_bytes().splitlines(keepends=True)[1:]}
    other_ids={str(qid).encode() for qid in v.qid[~mask]}
    checks=[]
    for path in sorted((HERE/'preds').glob('J1_*_validation.csv')):
        with path.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==len({r['query_id'] for r in rows})==v.n
        assert {r['query_id'] for r in rows}==set(v.qid)
        pp=read_pred_csv(path,v)
        assert np.isfinite(pp).all() and (pp>=0).all() and (pp<=1).all()
        assert np.max(abs(pp.sum(1)-1))<1e-6
        np.testing.assert_array_equal(pp[~mask],ref[~mask])
        for line in path.read_bytes().splitlines(keepends=True)[1:]:
            key=line.split(b',',1)[0]
            if key in other_ids:assert line==reference_rows[key]
        checks.append({'file':path.name,'rows':len(rows),'other_type_rows_byte_identical':len(other_ids),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    dev=np.concatenate([v.episode_rows(e) for e in scope_episodes(v,'dev')])
    ids=dev[mask[dev]];qwin=1-v.truth[ids,0]
    errors={}
    for name,p in [('baseline',ref),('train_prior',saved)]:
        errors[name]={}
        for lo,hi in [(0,.1),(.2,.5),(.5,.8),(.8,1.0001)]:
            sel=ids[(qwin>=lo)&(qwin<hi)]
            errors[name][f'{lo}-{hi}']=metrics(1-v.truth[sel,0],1-p[sel,0])
    result={'tests_passed':5,'two_run_train_prior_prediction_byte_identical':True,
            'saved_train_prior_forecast_byte_identical':True,'files':checks,
            'dev_level_error_comparison':errors,'audit_or_all_evaluations':0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='dev_level_error_comparison'},indent=2))


if __name__=='__main__':main()
