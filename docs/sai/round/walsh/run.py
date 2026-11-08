"""R6 kill-test. Fixed uniform horizon pool; no horizon or expert retuning."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv,evaluate,diff_ci,to_csv
from harness.scoring import scope_episodes
from harness.stack import blend_gain
from bma import predict,HALF_LIVES


def win_mse(pw,ref,v):
    rows=np.concatenate([v.episode_rows(e) for e in scope_episodes(v,'dev')])
    truth=1-v.truth[rows,0]
    error=(pw[rows]-truth)**2
    base=(1-ref[rows,0]-truth)**2
    return {'mse':float(error.mean()),'baseline_mse':float(base.mean()),
            'ratio':float(error.mean()/base.mean()),
            'condition_mse':{c:float(error[v.condition[rows]==c].mean()) for c in sorted(set(v.condition[rows]))},
            'baseline_condition_mse':{c:float(base[v.condition[rows]==c].mean()) for c in sorted(set(v.condition[rows]))}}


def main(c_slab):
    start=time.perf_counter()
    v=load_split('validation')
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    shares=ref[:,1:]/ref[:,1:].sum(1,keepdims=True)
    auditlog=HERE/'results/audit_access.log'
    before=hashlib.sha256(auditlog.read_bytes()).hexdigest()
    bank,diagnostic=predict(v,c_slab)
    pw=bank.mean(1)
    tag=str(c_slab).replace('.','p')
    np.savez(OUT/f'bank_c{tag}.npz',pw=bank,**diagnostic)
    results={'scope':'dev','c_slab':c_slab,'c0':2.,'k':6,'subsets':42,'max_subset_size':3,
             'pg_passes':2,'occam_nats':3.,'half_lives':HALF_LIVES,
             'horizon_pool':'fixed equal weights; not selected on dev',
             'centering':'strictly pre-tick EW signed mean, same horizon clock, clipped [-.85,.85]',
             'reference_code_sha256':hashlib.sha256((HERE.parents[1]/'gemini-code-1794099642432.md').read_bytes()).hexdigest(),
             'inherited_caveat':'Reference win prior and size shares derive from validation targets; BMA reads only pre-cutoff histories.',
             'reference':evaluate(ref,v,'dev'),'heads':{}}
    def assess(name,probability,save_oof=True):
        p=np.column_stack([1-probability,probability[:,None]*shares])
        to_csv(HERE/f'preds/R6_C{tag}_{name}_validation.csv',v,p)
        blend=blend_gain(p,ref,v,'dev',folds=6,grid=21)
        oof=blend.pop('oof')
        if save_oof:to_csv(HERE/f'preds/R6_C{tag}_{name}_OOF_validation.csv',v,oof)
        return {'win':win_mse(probability,ref,v),'plug_in':evaluate(p,v,'dev'),
                'plug_in_difference':diff_ci(p,ref,v,'dev'),
                'oof_blend':blend,'oof_win':win_mse(1-oof[:,0],ref,v)}
    results['heads']['UNIFORM']=assess('UNIFORM',pw)
    # Diagnostics are not a search over memory settings. They prevent a poor
    # fixed horizon pool from masking a good individual horizon in the verdict.
    for j,hl in enumerate(HALF_LIVES):
        name='INF' if hl is None else f'H{int(hl)}'
        results['heads'][name]=assess(name,bank[:,j])
    main=results['heads']['UNIFORM']
    results['killed']=main['win']['ratio']>2 or main['oof_blend']['w_mean']<=.01
    results['reason']={'mse_exceeds_twice_reference':main['win']['ratio']>2,
                       'oof_weight_near_zero':main['oof_blend']['w_mean']<=.01}
    results['further_c_tuning_skipped']=results['killed']
    results['audit_log_unchanged']=before==hashlib.sha256(auditlog.read_bytes()).hexdigest()
    # Other worker may independently log; our own code only calls dev. Do not
    # overwrite its access log if its activity changes the hash.
    results['seconds']=time.perf_counter()-start
    (OUT/f'results_c{tag}.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError(f'Set {key}=1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slab',type=float,choices=(1.,1.35,2.),default=1.35)
    args=parser.parse_args()
    main(args.slab)
