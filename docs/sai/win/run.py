"""Train-selected Bayesian logistic win heads; dev-only held-out blend tests."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv,evaluate,diff_ci,to_csv
from harness.scoring import scope_episodes,_resamples
from harness.stack import blend_gain
from logistic import terms,machine_rows,empirical_bayes,predict

HALF_LIVES=(None,120.,60.,30.)


def win_diagnostic(pw,reference,v):
    episodes=scope_episodes(v,'dev')
    cond=[];errors=[];base=[]
    for e in episodes:
        r=v.episode_rows(e);truth=1-v.truth[r,0]
        cond.append(list(CONDITIONS).index(v.condition[r[0]]))
        errors.append(float(np.mean((pw[r]-truth)**2)))
        base.append(float(np.mean((1-reference[r,0]-truth)**2)))
    cond=np.array(cond);errors=np.array(errors);base=np.array(base)
    delta=errors-base
    boots=[delta[i].mean() for i in _resamples(cond,1000)]
    return {'mse':float(errors.mean()),'baseline_mse':float(base.mean()),
            'difference':float(delta.mean()),'difference_ci95':np.percentile(boots,[2.5,97.5]).tolist(),
            'condition_mse':{c:float(errors[cond==i].mean()) for i,c in enumerate(CONDITIONS)},
            'baseline_condition_mse':{c:float(base[cond==i].mean()) for i,c in enumerate(CONDITIONS)}}


CONDITIONS=('stable','short_history','drift','shared_change')


def main():
    start=time.perf_counter()
    train,v=load_split('train'),load_split('validation')
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    shares=ref[:,1:]/ref[:,1:].sum(1,keepdims=True)
    dev=scope_episodes(v,'dev');rows=np.concatenate([v.episode_rows(e) for e in dev])
    auditlog=HERE/'results/audit_access.log'
    audit_sha=hashlib.sha256(auditlog.read_bytes()).hexdigest()
    report={'reference':evaluate(ref,v,'dev'),'selection_source':'train next-outcome log loss only; EB uses train pre-cutoff histories only',
            'reference_caveat':'TM_COMBINED4 inherited win prior and size modes use all validation targets. Logistic models never fit validation targets.',
            'train_grid':[],'families':{},'versions':{'numpy':np.__version__},'half_life_grid':HALF_LIVES}
    plugs=[]
    for family,full in [('KNOWN',False),('FULL',True)]:
        machines=machine_rows(train,full)
        train_candidates=[]
        for hl in HALF_LIVES:
            t=time.perf_counter()
            variance,history=empirical_bayes(machines,hl)
            pw,diagnostic=predict(machines,variance,hl)
            y=(train.outcome>0).astype(float)
            loss=-(y*np.log(np.clip(pw,1e-12,1))+(1-y)*np.log(np.clip(1-pw,1e-12,1)))
            row={'family':family,'feature_count':len(terms(full)),'half_life':hl,
                 'train_log_loss':float(loss.mean()),'train_brier':float(((pw-y)**2).mean()),
                 'train_condition_log_loss':{c:float(loss[train.condition==c].mean()) for c in CONDITIONS},
                 'prior_variance':{k:a.tolist() for k,a in variance.items()},'eb_history':history,
                 'fit_diagnostics':diagnostic,'seconds':time.perf_counter()-t}
            report['train_grid'].append(row);train_candidates.append((row,variance))
            print('TRAIN',family,'hl',hl,'logloss',row['train_log_loss'],'seconds',row['seconds'],flush=True)
        selected,variance=min(train_candidates,key=lambda x:x[0]['train_log_loss'])
        hl=selected['half_life'];vm=machine_rows(v,full)
        pw,diagnostic=predict(vm,variance,hl)
        p=np.column_stack([1-pw,pw[:,None]*shares])
        mid=f'WIN_{family}'
        to_csv(HERE/f'preds/{mid}_validation.csv',v,p)
        np.save(OUT/f'{mid}_pw.npy',pw)
        blend=blend_gain(p,ref,v,'dev',folds=6,grid=21)
        oof=blend.pop('oof')
        to_csv(HERE/f'preds/{mid}_BLEND_OOF_validation.csv',v,oof)
        family_report={'id':mid,'features':terms(full),'half_life':hl,
                       'selection_train_log_loss':selected['train_log_loss'],'fit_diagnostics':diagnostic,
                       'win':win_diagnostic(pw,ref,v),'plug_in':evaluate(p,v,'dev'),
                       'plug_in_difference':diff_ci(p,ref,v,'dev'),
                       'oof_blend':blend,'oof_blend_win':win_diagnostic(1-oof[:,0],ref,v)}
        report['families'][family]=family_report
        plugs.append((mid,p))
        (OUT/'results.partial.json').write_text(json.dumps(report,indent=2)+'\n')
        print('DEV',family,json.dumps(family_report),flush=True)
    # Opposite-fold selection checks feature-family and blend-weight selection.
    # Memory settings and priors above are train-selected and never retuned.
    fold={}
    for condition in CONDITIONS:
        eps=[e for e in dev if v.condition[v.episode_rows(e)[0]]==condition]
        eps.sort(key=lambda e:hashlib.sha256(f'win-half-v1|{e}'.encode()).hexdigest())
        fold.update({e:i%2 for i,e in enumerate(eps)})
    oof=ref.copy();choices=[]
    for f in (0,1):
        tr=np.concatenate([v.episode_rows(e) for e in dev if fold[e]!=f])
        te=np.concatenate([v.episode_rows(e) for e in dev if fold[e]==f])
        possibilities=[]
        for name,p in plugs:
            for weight in np.linspace(0,1,21):
                pp=weight*p+(1-weight)*ref
                loss=float(((pp[tr]-v.truth[tr])**2).sum(1).mean())
                possibilities.append((loss,name,float(weight),pp))
        _,name,weight,pp=min(possibilities,key=lambda row:row[0])
        oof[te]=pp[te]
        choices.append({'test_fold':f,'family':name,'weight':weight})
    report['opposite_fold']={'evaluation':evaluate(oof,v,'dev'),'difference':diff_ci(oof,ref,v,'dev'),
                             'win':win_diagnostic(1-oof[:,0],ref,v),'choices':choices}
    to_csv(HERE/'preds/WIN_SELECTED_OPPOSITE_OOF_validation.csv',v,oof)
    # The user asked to stop if full EB logistic is clearly worse on win MSE.
    full=report['families']['FULL']['win']
    report['stopped_early']=full['difference_ci95'][0]>0
    report['optional_fast_slow_skipped']='Full 37-term EB win MSE is clearly worse than v4; stop per task.' if report['stopped_early'] else 'Optional extension not needed for this bounded test.'
    report['audit_log_unchanged']=hashlib.sha256(auditlog.read_bytes()).hexdigest()==audit_sha
    assert report['audit_log_unchanged']
    report['seconds']=time.perf_counter()-start
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'results.partial.json').unlink()
    print('DONE',report['stopped_early'],report['opposite_fold'],flush=True)


if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError(f'Set {key}=1')
    main()
