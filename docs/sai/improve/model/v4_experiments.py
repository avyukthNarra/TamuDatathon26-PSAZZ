"""M1-M3 on pinned mixer v4. Supersedes v3 research, which is archived.

The baseline's validation-derived Beta prior and size modes are inherited
unchanged. All ADDITIONAL population structure is fitted on TRAIN histories.
"""
import json
import time
from pathlib import Path
import numpy as np
from experiments import HERE,OUT,ROOT,PAIRS,train_structure,win_head
from harness import load_split,read_pred_csv,evaluate,diff_ci,to_csv
from harness.scoring import scope_episodes


def main():
    started=time.time()
    train,v=load_split('train'),load_split('validation')
    priors,cells,beta,peers=train_structure(train)
    metadata=json.loads((HERE/'combos/TM_MIXER4.json').read_text())
    pop=metadata['win_prior']
    mixer=read_pred_csv(HERE/'preds/TM_MIXER4_validation.csv',v)
    rule=read_pred_csv(HERE/'preds/TM_RULE_validation.csv',v)
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    # Independent size head is IDENTICAL between v3 and v4. Use exact raw
    # v3 forecasts already generated with original baseline size modes.
    raw=np.load(OUT/'v3_raw.npy')
    sizes=raw[:,1:]/raw[:,1:].sum(1,keepdims=True)
    def predict(**params):
        params={'eta_short':8.,'eta_long':8.,**params}
        pw=win_head(v,priors,cells,beta,population_prior=pop,**params)
        pp=np.maximum(np.column_stack((1-pw,pw[:,None]*sizes)),1e-4)
        return pp/pp.sum(1,keepdims=True)
    baseline=predict()
    error=float(abs(baseline-mixer).max())
    print('V4_BASELINE_REPRODUCTION',error,flush=True)
    assert error<1e-10,error
    assert abs(.2*rule+.8*baseline-ref).max()<1e-10
    dev_eps=scope_episodes(v,'dev')
    folds=[[],[]]
    for c in sorted(set(v.condition)):
        eps=[e for e in dev_eps if v.condition[v.episode_rows(e)[0]]==c]
        for f in (0,1):
            folds[f]+=[i for e in eps[f::2] for i in v.episode_rows(e)]
    folds=[np.array(sorted(f)) for f in folds]
    def score(p):return evaluate(p,v,'dev',bootstrap=0)['points']
    def fold_score(p,f):
        ids=folds[f]
        return evaluate(p[ids],v.subset(ids),'dev',bootstrap=0)['points']
    results={};grid=[];weights=(.1,.2,.3)
    def experiment(mid,settings):
        candidates=[]
        for params in settings:
            pp=predict(**params)
            for rw in weights:
                blend=rw*rule+(1-rw)*pp
                row={'id':mid,'params':params,'rule_weight':rw,'dev':score(blend),
                     'fold0':fold_score(blend,0),'fold1':fold_score(blend,1)}
                candidates.append((row,blend))
            print('GRID',mid,params,candidates[-2][0]['dev'],flush=True)
        grid.extend(r for r,_ in candidates)
        best,pp=max(candidates,key=lambda x:x[0]['dev'])
        to_csv(HERE/f'preds/{mid}_validation.csv',v,pp)
        fixed20=max(
            [(r,p) for r,p in candidates if r['rule_weight']==.2],key=lambda x:x[0]['dev'])
        results[mid]={**best,'evaluation':evaluate(pp,v,'dev'),'difference':diff_ci(pp,ref,v,'dev'),
                      'fixed20':{'settings':fixed20[0],'evaluation':evaluate(fixed20[1],v,'dev'),
                                 'difference':diff_ci(fixed20[1],ref,v,'dev')}}
        oof=ref.copy();choices=[]
        for f in (0,1):
            selected,pred=max(candidates,key=lambda x:x[0][f'fold{1-f}'])
            oof[folds[f]]=pred[folds[f]];choices.append(selected)
        results[mid]['crossfit']={'evaluation':evaluate(oof,v,'dev'),
            'difference':diff_ci(oof,ref,v,'dev'),'choices':choices}
        print('RESULT',mid,json.dumps(results[mid]),flush=True)
    experiment('IMP_M0V4',[{}])
    experiment('IMP_M1V4',[dict(prior_power=p,eta_short=e) for p in (0.,1.,4.) for e in (4.,8.,16.)])
    experiment('IMP_M2V4',[dict(k_short=k,k_long=l) for k in (0.,.0625,.25,1.,4.) for l in (0.,.0625,.25)])
    experiment('IMP_M3V4',[dict(peer_strength=k) for k in (0.,.0625,.25,1.,4.)])
    # Individual M1/M2 components are chosen on full dev for this exploratory
    # final fit. Exclude M3 unless it beats a matched no-peer pseudo-count control.
    peer_best=results['IMP_M3V4'];pk=peer_best['params']['peer_strength'];rw=peer_best['rule_weight']
    control=rw*rule+(1-rw)*predict(k_short=pk)
    peer=read_pred_csv(HERE/'preds/IMP_M3V4_validation.csv',v)
    peer_increment=diff_ci(peer,control,v,'dev')
    peer_best['peer_increment_vs_matched_type_shrink']=peer_increment
    members=['IMP_M1V4','IMP_M2V4']
    if peer_increment['diff']>0 and peer_increment['ci95'][0]>0:
        members+=['IMP_M3V4']
    params={}
    for mid in members:params.update(results[mid]['params'])
    experiment('IMP_M5V4',[params])
    # Reselect each component's parameters on opposite dev folds.
    oof=ref.copy();choices=[]
    for f in (0,1):
        params={}
        for mid in members:
            row=max((r for r in grid if r['id']==mid),key=lambda r:r[f'fold{1-f}'])
            params.update(row['params'])
        pp=predict(**params)
        rw=max(weights,key=lambda rw:fold_score(rw*rule+(1-rw)*pp,1-f))
        blend=rw*rule+(1-rw)*pp;oof[folds[f]]=blend[folds[f]]
        choices.append({'test_fold':f,'params':params,'rule_weight':rw})
    results['IMP_M5V4']['crossfit']={'evaluation':evaluate(oof,v,'dev'),
        'difference':diff_ci(oof,ref,v,'dev'),'choices':choices}
    baseline_best=read_pred_csv(HERE/'preds/IMP_M0V4_validation.csv',v)
    for mid in results:
        p=read_pred_csv(HERE/f'preds/{mid}_validation.csv',v)
        results[mid]['difference_vs_weight_only']=diff_ci(p,baseline_best,v,'dev')
    output={'reference_name':'TM_COMBINED4','source':'39e5757','reference':evaluate(ref,v,'dev'),
            'baseline_reproduction_max_abs_diff':error,'inherited_population_prior':pop,
            'inherited_validation_target_dependency':'Both Beta mixture means and size modes fit all 96 validation targets, including reserved audit episodes. No new audit scoring.',
            'new_structure_source':'train histories only','train_peer_checks':peers,
            'experiments':results,'grid':grid,'seconds':time.time()-started}
    (OUT/'v4_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('FINISHED',time.time()-started,flush=True)

if __name__=='__main__':main()
