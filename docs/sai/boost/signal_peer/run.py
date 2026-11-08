"""Train-only signal evidence/detectors; small dev-only T2/T3 grids."""
import json
import time
import numpy as np
from models import *
from harness import read_pred_csv,to_csv,evaluate,diff_ci
from harness.scoring import scope_episodes


def main():
    started=time.time();train=load_split('train');v=load_split('validation')
    reference=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    mixer=read_pred_csv(HERE/'preds/TM_MIXER4_validation.csv',v)
    rule=read_pred_csv(HERE/'preds/TM_RULE_validation.csv',v)
    wp,win_evidence=signal_evidence(train,*beta_cell_evidence(train))
    sp,size_evidence_report=signal_evidence(train,*size_evidence(train))
    tw,traintrace,_,_,train_heads=win_walk(train,trace_heads=True)
    train_score=contrast_detector_score(train,train_heads)
    thresholds={}
    for kind in TYPES:
        rows=(train.mtype==kind)&(train.condition=='stable')
        sample=train_score[rows];sample=sample[np.isfinite(sample)]
        thresholds[kind]=float(np.quantile(sample,.99))
    pw,trace,pred,logw,heads=win_walk(v,trace_heads=True);sizes=size_walk(v)
    detector=contrast_detector_score(v,heads)
    def combined(win,size=sizes):
        p=np.maximum(np.column_stack((1-win,win[:,None]*size)),1e-4)
        p/=p.sum(1,keepdims=True)
        return .2*rule+.8*p
    baseline=combined(pw);error=float(abs(baseline-reference).max())
    assert error<1e-10,error
    train_detection={}
    for condition in sorted(set(train.condition)):
        flag=trigger(train,train_score,thresholds,32,2)
        train_detection[condition]=float(flag[train.condition==condition].mean())
    evidence={'win_signal':win_evidence,'size_signal':size_evidence_report,
              'size_action':'Skipped size-mode intervention: TRAIN crossfit signal loglik CI includes zero and mean is negative.',
              'detector':'TRAIN-selected EWMA(.8) slow-minus-fast observed binary Brier, threshold=TRAIN stable per-type 99th percentile after tick40. Event stamped t+1, excludes future observations. TRAIN k2W32 trigger: shared/drift1.0 vs stable.344, beating excess-Brier detector.',
              'thresholds':thresholds,'train_peer_trigger_rate_k2_W32':train_detection}
    (OUT/'train_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('TRAIN_EVIDENCE',json.dumps(evidence),flush=True)
    dev_eps=scope_episodes(v,'dev');folds=[[],[]]
    for condition in sorted(set(v.condition)):
        eps=[e for e in dev_eps if v.condition[v.episode_rows(e)[0]]==condition]
        for f in (0,1):folds[f]+=[i for e in eps[f::2] for i in v.episode_rows(e)]
    folds=[np.array(sorted(x)) for x in folds]
    results={};forecasts={};groups={}
    def score(p,fold):
        rows=folds[fold]
        return evaluate(p[rows],v.subset(rows),'dev',bootstrap=0)['points']
    def record(mid,p,group,params=None,extra=None):
        to_csv(HERE/f'preds/{mid}_validation.csv',v,p)
        forecasts[mid]=p;groups.setdefault(group,[]).append(mid)
        results[mid]={'params':params or {},'group':group,'evaluation':evaluate(p,v,'dev'),
                      'difference':diff_ci(p,reference,v,'dev'),
                      'fold0':score(p,0),'fold1':score(p,1),**(extra or {})}
        print('RESULT',mid,results[mid]['evaluation']['points'],results[mid]['difference'],flush=True)
    record('BOOST_T2BASE',baseline,'T2')
    for strength in (.5,1.,2.):
        # Signal-only tilt: hold the baseline type marginal prior fixed, use
        # group/type likelihood ratios to isolate the signal contribution.
        base=np.array([wp[(k,None)] for k in v.mtype])
        group=np.array([wp[(k,int(g))] for k,g in zip(v.mtype,v.signal)])
        logs=np.log(DEFAULT)+strength*(np.log(group)-np.log(base))
        weights=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
        win,_,_,_=win_walk(v,weights)
        record(f'BOOST_T2S{int(strength*10)}',combined(win),'T2',{'kind':'signal-only log weight tilt','strength':strength})
    for strength in (.25,1.):
        for signal,label in ((False,'TYPE'),(True,'GROUP')):
            weights=weights_for(v,wp,strength,signal)
            win,_,_,_=win_walk(v,weights)
            record(f'BOOST_T2{label}{int(strength*100)}',combined(win),'T2',
                   {'kind':'type+signal' if signal else 'type-only control','strength':strength})
    record('BOOST_T3BASE',baseline,'T3peer')
    groups['T3own']=['BOOST_T3BASE']
    for window in (12,32):
        for multiplier in (4.,16.):
            own=trigger(v,detector,thresholds,window,0)
            win=fast_boost(pred,logw,own,multiplier)
            record(f'BOOST_T3CO{window}B{int(multiplier)}',combined(win),'T3own',
                   {'window':window,'fast_multiplier':multiplier,'k':0},
                   {'dev_trigger_rate':float(own[np.isin(v.episode,dev_eps)].mean())})
            for k in (2,3):
                flags=trigger(v,detector,thresholds,window,k)
                win=fast_boost(pred,logw,flags,multiplier)
                record(f'BOOST_T3CP{k}W{window}B{int(multiplier)}',combined(win),'T3peer',
                       {'window':window,'fast_multiplier':multiplier,'k':k},
                       {'dev_trigger_rate':float(flags[np.isin(v.episode,dev_eps)].mean())})
    best={};crossfit={}
    for group,names in groups.items():
        best[group]=max(names,key=lambda n:results[n]['evaluation']['points'])
        oof=reference.copy();choices=[]
        for f in (0,1):
            name=max(names,key=lambda n:results[n][f'fold{1-f}'])
            oof[folds[f]]=forecasts[name][folds[f]];choices.append(name)
        crossfit[group]={'choices':choices,'evaluation':evaluate(oof,v,'dev'),
                         'difference':diff_ci(oof,reference,v,'dev')}
    type_names=['BOOST_T2BASE','BOOST_T2TYPE25','BOOST_T2TYPE100']
    typed=max(type_names,key=lambda n:results[n]['evaluation']['points'])
    peer=best['T3peer'];own=best['T3own'];signal=best['T2']
    comparisons={'best_T2_vs_best_type_control':diff_ci(forecasts[signal],forecasts[typed],v,'dev'),
                 'best_peer_vs_best_own':diff_ci(forecasts[peer],forecasts[own],v,'dev')}
    output={'reference':evaluate(reference,v,'dev'),'baseline_reproduction_max_abs_diff':error,
            'train_evidence':evidence,'variants':results,'best':best,'crossfit':crossfit,
            'comparisons':comparisons,'seconds':time.time()-started,
            'validation_target_dependency':'Unchanged v4 levels (.0435/.898), Beta strengths, and size modes are inherited from all validation targets. All NEW EM weights and detector thresholds use TRAIN histories; no validation-target fitting or audit scoring.'}
    (OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(OUT/'baseline_trace.npz',train_trace=traintrace,validation_trace=trace,validation_expert_pred=pred,validation_logw=logw)
    print('BEST',best,'CROSSFIT',crossfit,'SECONDS',time.time()-started,flush=True)

if __name__=='__main__':main()
