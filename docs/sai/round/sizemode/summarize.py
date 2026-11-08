"""Summarize train evidence, apply the requested train gate, score baseline only."""
import json
import sys
from pathlib import Path
import numpy as np
from discover import OUT,ROOT,FAMILIES,bootstrap_episode_delta
from harness import load_split,read_pred_csv,evaluate,diff_ci


def cell_ci(train,values,mask):
    eps=sorted(set(train.episode[mask]));per=[]
    for ep in eps:
        x=values[mask&(train.episode==ep)]
        if np.isfinite(x).any():per.append(np.nanmean(x))
    per=np.asarray(per);rng=np.random.RandomState(719)
    bootstrap=np.array([rng.choice(per,len(per)).mean() for _ in range(1000)])
    return {'mean':float(per.mean()),'ci95':np.percentile(bootstrap,[2.5,97.5]).tolist(),'episodes':len(per)}


def main():
    train=load_split('train');v=load_split('validation')
    r=json.loads((OUT/'train_discovery.json').read_text());cache=np.load(OUT/'train_scores.npz')
    tail=cache['tail_n'];deltas=(cache['family_tail_ll']-cache['baseline_tail_ll'][:,None])/np.maximum(tail[:,None],1)
    deltas[tail==0]=np.nan
    focus={}
    for kind in ('balanced','steady'):
        mask=(train.condition=='short_history')&(train.mtype==kind)
        focus[kind]={family:cell_ci(train,deltas[:,j],mask) for j,family in enumerate(FAMILIES)}
    baseline_path=ROOT/'docs/sai/round/mixer6/MIXER6_rule00_insample_validation.csv'
    p=read_pred_csv(baseline_path,v)
    # The requested gate is a family CLEARLY beating mixer6's size experts on
    # held-out TRAIN. None passes globally or on the target short-history types.
    passed=[family for family,stats in r['tail_holdout'].items() if stats['ci95'][0]>0]
    target_passed={kind:[family for family,stats in stats_by_family.items() if stats['ci95'][0]>0] for kind,stats_by_family in focus.items()}
    output={'baseline_file':str(baseline_path.relative_to(ROOT)),'baseline':evaluate(p,v,'dev'),
            'baseline_diff_ci':diff_ci(p,p,v,'dev'),'focused_train_holdout':focus,
            'global_passed_train_gate':passed,'target_passed_train_gate':target_passed,
            'model_action':'No model change or dev tuning: no family passed the held-out-train gate. Opposite-half DEV tuning check is not applicable.',
            'prediction_files_written':[],'target_dependency':'Known size modes inherited from validation targets, as authorized. All rule/BIC/peer selection uses TRAIN histories only. Baseline alone scored on DEV; no audit/all calls.'}
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Train gates',passed,target_passed)
    print('Short steady HMM',focus['steady']['hmm'])
    print('Short balanced HMM',focus['balanced']['hmm'])
    print('Baseline',output['baseline'])

if __name__=='__main__':main()
