"""Jackpot-only active-level prior changes in a pinned copy of mixer6.

All baseline target-derived parameters are reproduced unchanged first. New
prior means/variances use TRAIN only; dev selects a tiny fixed strength grid.
No new pair family is added because stationary train evidence favors known pairs.
"""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'docs/sai';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'worktrees/_tm/mixer6/mainFiles'))
sys.path.insert(1,str(HERE))
from slot_forecasting.io import load_inputs,read_targets
from harness import load_split,read_pred_csv,evaluate,diff_ci
from harness.scoring import scope_episodes
from mixer6_copy import PairMixer,choose_pairs,discover_size_keys,size_modes_from_targets,win_prior_by_type_from_targets


def train_active_prior():
    data=json.loads((OUT/'structure.json').read_text())['machines']
    values=[];sizes=[];ids=[]
    for row in data:
        if row['type']!='jackpot' or row['condition']!='stable':continue
        t=row['families']['known_win_pair']['table']
        if len(t['rates'])!=2 or min(t['n'])<10:continue
        high=int(np.argmax(t['rates']))
        values.append(t['rates'][high]);sizes.append(t['n'][high]);ids.append(row['query_id'])
    values=np.array(values);sizes=np.array(sizes)
    mean=float(values.mean())
    noise=float(np.mean(values*(1-values)/np.maximum(sizes-1,1)))
    variance=max(float(np.var(values,ddof=1))-noise,1e-4)
    strength=float(np.clip(mean*(1-mean)/variance-1,2,100))
    return {'mean':mean,'strength':strength,'sample_machines':len(values),'machines':ids,
            'observed_variance':float(np.var(values,ddof=1)),'estimated_binomial_variance':noise,
            'method':'Known-pair high cells, stable jackpot train machines only, both cells n>=10; method-of-moments Beta with binomial noise subtracted. Selection bias remains.'}


def main():
    started=time.perf_counter()
    v=load_split('validation');reference_path=HERE/'round/mixer6/MIXER6_rule00_insample_validation.csv'
    ref=read_pred_csv(reference_path,v);jackpot=v.mtype=='jackpot'
    q,h=load_inputs(ROOT/'mainFiles/public/train')
    vq,vh=load_inputs(ROOT/'mainFiles/public/validation')
    # These are INHERITED baseline fits from all 96 targets, identical to the
    # existing source scorer. They are not a new audit evaluation or new fit.
    targets=read_targets(ROOT/'mainFiles/public/validation/targets.csv')
    modes=size_modes_from_targets(targets,vq);priors=win_prior_by_type_from_targets(targets,vq)
    pairs=choose_pairs(q,h);size_keys=discover_size_keys(q,h)
    jq=tuple(x for x in vq if x.machine_type=='jackpot')
    keys={(x.episode_id,x.machine_id) for x in jq}
    jh=tuple(x for x in vh if (x.episode_id,x.machine_id) in keys)
    kwargs={'pairs':pairs,'size_keys':size_keys,'size_modes':modes,'win_prior':priors,
            'eta':6.,'cell_mode':'parity'}
    auditlog=HERE/'results/audit_access.log';before=hashlib.sha256(auditlog.read_bytes()).hexdigest()
    def predict(priors):
        model=PairMixer(**{**kwargs,'win_prior':priors}).fit(q,h)
        pp=model.predict(jq,jh)
        out=ref.copy()
        for i in np.flatnonzero(jackpot):out[i]=pp[str(v.qid[i])]
        return out
    baseline=predict(priors)
    mismatch=float(np.max(abs(baseline-ref)))
    assert mismatch<1e-12,f'Copied baseline failed reproduction: {mismatch}'
    prior=train_active_prior()
    configurations=[('J1_BASE',None,None),('J1_LIVE2',None,2.),('J1_LIVE6',None,6.),
                    ('J1_TRAIN',prior['mean'],prior['strength'])]
    dev=scope_episodes(v,'dev');folds=[[],[]]
    for condition in sorted(set(v.condition)):
        eps=[e for e in dev if v.condition[v.episode_rows(e)[0]]==condition]
        eps.sort(key=lambda e:hashlib.sha256(f'j1-fold|{e}'.encode()).hexdigest())
        for f in (0,1):folds[f].extend(i for e in eps[f::2] for i in v.episode_rows(e))
    folds=[np.array(sorted(f)) for f in folds]
    devrows=np.concatenate([v.episode_rows(e) for e in dev]);jdev=devrows[jackpot[devrows]]
    original_lines=reference_path.read_text().splitlines(keepends=True)
    original_rows={line.split(',',1)[0]:line for line in original_lines[1:]}
    def save(mid,p):
        lines=[original_lines[0]]
        for i,qid in enumerate(v.qid):
            lines.append(original_rows[qid] if not jackpot[i] or np.array_equal(p[i],ref[i])
                         else str(qid)+','+','.join(f'{x:.17g}' for x in p[i])+'\n')
        (HERE/f'preds/{mid}_validation.csv').write_text(''.join(lines))
    def mse(p):
        return {'overall_jackpot_win_mse':float(np.mean((p[jdev,0]-v.truth[jdev,0])**2)),
            'condition_mse':{c:float(np.mean((p[jdev[v.condition[jdev]==c],0]-v.truth[jdev[v.condition[jdev]==c],0])**2)) for c in sorted(set(v.condition))}}
    candidates=[]
    for mid,mean,strength in configurations:
        adapted=copy.deepcopy(priors)
        if strength is not None:
            weight,level,_=adapted['jackpot'][1]
            adapted['jackpot'][1]=(weight,level if mean is None else mean,strength)
        p=ref.copy() if mid=='J1_BASE' else predict(adapted)
        assert p[~jackpot].tobytes()==ref[~jackpot].tobytes()
        save(mid,p)
        row={'id':mid,'live_prior':adapted['jackpot'][1],'dev':evaluate(p,v,'dev'),
             'difference':diff_ci(p,ref,v,'dev'),'jackpot_win':mse(p),
             'fold_points':[evaluate(p[ids],v.subset(ids),'dev',bootstrap=0)['points'] for ids in folds]}
        candidates.append((row,p));print('RESULT',mid,row,flush=True)
    oof=ref.copy();choices=[]
    for f in (0,1):
        row,p=max(candidates,key=lambda item:item[0]['fold_points'][1-f])
        oof[folds[f]]=p[folds[f]];choices.append({'test_fold':f,'selected':row['id']})
    save('J1_OOF',oof)
    result={'reference':'mixer6 alone, 814cb60','baseline_max_abs_reproduction_error':mismatch,
        'baseline_priors':priors,'pairs':pairs,'size_keys':size_keys,'train_active_prior':prior,
        'inherited_dependency':'Baseline win/size priors fit all 96 validation targets; preserved. New fit uses train only. Scoring/selection dev only.',
        'variants':[r for r,p in candidates],
        'opposite_half':{'dev':evaluate(oof,v,'dev'),'difference':diff_ci(oof,ref,v,'dev'),
                         'jackpot_win':mse(oof),'choices':choices},
        'audit_log_unchanged':before==hashlib.sha256(auditlog.read_bytes()).hexdigest(),
        'seconds':time.perf_counter()-started}
    (OUT/'fix_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',result['opposite_half'],flush=True)


if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError(f'Set {key}=1')
    main()
