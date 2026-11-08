"""Dev-only evaluation of exact size hypotheses with a frozen mixer5 win head."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'docs/sai'))
from harness import load_split,evaluate,diff_ci,to_csv
from harness.scoring import scope_episodes
from exact_size import ExactSizeSwitch,train_big_win_prior,HYPOTHESIS_NAMES


def main():
    start=time.time();train=load_split('train');v=load_split('validation')
    arrays=np.load(OUT/'baseline_arrays.npz');meta=json.loads((OUT/'baseline.json').read_text())
    assert np.array_equal(arrays['qid'],v.qid)
    base,mix,rule=arrays['base'],arrays['mixer'],arrays['rule']
    win=mix[:,1:].sum(1);base_size=mix[:,1:]/win[:,None]
    prior=train_big_win_prior(train)
    model=ExactSizeSwitch(meta['size_modes'],prior)
    sizes,details=model.predict(v,return_details=True)
    configurations={'H12':sizes[0],'H40':sizes[1],'HINF':sizes[2],
                    'M12_40':sizes[:2].mean(0),'M40_INF':sizes[1:].mean(0),
                    'MALL':sizes.mean(0)}
    eps=scope_episodes(v,'dev');dev_rows=np.concatenate([v.episode_rows(e) for e in eps])
    short=dev_rows[v.condition[dev_rows]=='short_history']
    folds=[[],[]]
    for condition in sorted(set(v.condition)):
        ce=[e for e in eps if v.condition[v.episode_rows(e)[0]]==condition]
        for f in (0,1):folds[f]+=[i for e in ce[f::2] for i in v.episode_rows(e)]
    folds=[np.array(sorted(x)) for x in folds]
    def fold_score(p,f):
        rows=folds[f]
        return evaluate(p[rows],v.subset(rows),'dev',bootstrap=0)['points']
    def size_error(p):
        true_win=1-v.truth[short,0]
        truth_size=v.truth[short,1:]/true_win[:,None]
        actual_size=p[short,1:]/p[short,1:].sum(1,keepdims=True)
        difference=(actual_size-truth_size)**2
        return {'short_history_conditional_size_mse':float(difference.sum(1).mean()),
                'short_history_size_only_sse_true_win_fixed':float((difference.sum(1)*true_win**2).mean())}
    results={};predictions={}
    def record(mid,p,settings):
        if mid!='R5_BASE':to_csv(ROOT/f'docs/sai/preds/{mid}_validation.csv',v,p)
        forecasts=p[:,0]
        assert abs(forecasts-base[:,0]).max()<1e-14,'Win head must remain frozen'
        results[mid]={'settings':settings,'evaluation':evaluate(p,v,'dev'),
                      'difference_vs_R5_BASE':diff_ci(p,base,v,'dev'),**size_error(p),
                      'fold0':fold_score(p,0),'fold1':fold_score(p,1)}
        predictions[mid]=p
        print('RESULT',mid,results[mid]['evaluation']['points'],results[mid]['difference_vs_R5_BASE'],size_error(p),flush=True)
    record('R5_BASE',base,{'baseline':'20% rule + 80% mixer5'})
    for name,size in configurations.items():
        for amount in (1.,.5):
            adapted=amount*size+(1-amount)*base_size
            # Preserve the SAVED mixer win probability exactly, including its
            # existing floor normalization. Only conditional win sizes change.
            new_mix=np.column_stack((mix[:,0],win[:,None]*adapted))
            p=.2*rule+.8*new_mix
            name_id=f'R5_{name}'+('_HALF' if amount==.5 else '')
            record(name_id,p,{'memories':name,'exact_size_weight':amount})
    best=max(results,key=lambda k:results[k]['evaluation']['points'])
    oof=base.copy();choices=[]
    for f in (0,1):
        chosen=max(results,key=lambda k:results[k][f'fold{1-f}'])
        oof[folds[f]]=predictions[chosen][folds[f]]
        choices.append({'test_fold':f,'chosen':chosen})
    to_csv(ROOT/'docs/sai/preds/R5_OOF_validation.csv',v,oof)
    output={'reference':'R5_BASE','variants':results,'best':best,'train_big_win_prior':prior,
            'opposite_half':{'choices':choices,'evaluation':evaluate(oof,v,'dev'),
                             'difference_vs_R5_BASE':diff_ci(oof,base,v,'dev'),**size_error(oof)},
            'diagnostics':{'short_history_mean_posterior_entropy_by_memory':[
                float((-(details['posterior'][i,short]*np.log(np.maximum(details['posterior'][i,short],1e-15))).sum(1)).mean()) for i in range(3)],
                'short_history_mean_missing_query_triggers':float(details['final_missing_triggers'][short].mean()),
                'hypotheses':HYPOTHESIS_NAMES},'seconds':time.time()-start,
            'target_dependency':meta['target_dependency'],
            'size_error_definition':'Mean five-class SSE with true win rate held fixed and forecast conditional size shares; reported only on DEV short_history rows. Conditional-size MSE also recorded.'}
    (OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(OUT/'exact_sizes.npz',sizes=sizes,posterior=details['posterior'],qid=v.qid)
    print('BEST',best,'OPPOSITE_HALF',output['opposite_half'],'SECONDS',time.time()-start,flush=True)

if __name__=='__main__':main()
