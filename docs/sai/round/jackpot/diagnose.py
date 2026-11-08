"""J1 target/error description. Validation diagnostics are DEV only."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split,read_pred_csv
from harness.scoring import scope_episodes


def metrics(q,p):
    wrong=(q>=.5)!=(p>=.5)
    error=(p-q)**2
    return {'n':len(q),'true_mean':float(q.mean()),'predicted_mean':float(p.mean()),
            'bias':float((p-q).mean()),'mse':float(error.mean()),
            'wrong_half_fraction':float(wrong.mean()),
            'wrong_half_error_share':float(error[wrong].sum()/max(error.sum(),1e-20)),
            'wrong_half_mse':float(error[wrong].mean()) if wrong.any() else None,
            'same_half_mse':float(error[~wrong].mean()) if (~wrong).any() else None}


def main():
    train,v=load_split('train'),load_split('validation')
    ref=read_pred_csv(HERE/'round/mixer6/MIXER6_rule00_insample_validation.csv',v)
    rows=np.concatenate([v.episode_rows(e) for e in scope_episodes(v,'dev')])
    pred=1-ref[:,0];truth=1-v.truth[:,0]
    result={'scope':'dev','types':{},'train_win_counts':{}}
    for kind in sorted(set(v.mtype)):
        ids=rows[v.mtype[rows]==kind];q,p=truth[ids],pred[ids]
        ordered=np.sort(q);clusters={}
        for width in (.01,.025,.05):
            groups=np.split(ordered,np.flatnonzero(np.diff(ordered)>width)+1)
            clusters[str(width)]=[{'n':len(g),'min':float(g[0]),'max':float(g[-1])} for g in groups]
        record={'summary':metrics(q,p),'deciles':np.quantile(q,np.arange(0,1.01,.1)).tolist(),
                'unique_round12':int(len(np.unique(np.round(q,12)))),
                'gap_clusters':clusters,'true_level':{},'conditions':{},'signals':{},'history':{}}
        for lo,hi in [(0,.05),(.05,.1),(.1,.2),(.2,.5),(.5,.8),(.8,1.0001)]:
            keep=(q>=lo)&(q<hi)
            if keep.any():record['true_level'][f'{lo}-{hi}']=metrics(q[keep],p[keep])
        for condition in sorted(set(v.condition[ids])):
            use=ids[v.condition[ids]==condition]
            record['conditions'][condition]=metrics(truth[use],pred[use])
        for signal in sorted(set(v.signal[ids])):
            use=ids[v.signal[ids]==signal]
            record['signals'][str(int(signal))]=metrics(truth[use],pred[use])
        nobs=np.isfinite(v.grid[ids]).sum(1)
        wins=np.nansum(v.grid[ids]>0,axis=1)
        for name,keep in [('64_observed',nobs==64),('dense',nobs>64),
                          ('cutoff_lt224',v.cutoff[ids]<224),('cutoff_ge224',v.cutoff[ids]>=224),
                          ('at_most20_wins',wins<=20),('21_to40_wins',(wins>20)&(wins<=40)),('over40_wins',wins>40)]:
            if keep.any():record['history'][name]=metrics(q[keep],p[keep])
        if kind=='jackpot':
            threshold=.15
            wrong=(q>=threshold)!=(p>=threshold)
            error=(p-q)**2
            record['dead_active_regime']={'threshold_in_observed_target_gap':threshold,
                'wrong_count':int(wrong.sum()),'wrong_fraction':float(wrong.mean()),
                'wrong_error_share':float(error[wrong].sum()/max(error.sum(),1e-20)),
                'predicted_dead_range':[float(p[q<threshold].min()),float(p[q<threshold].max())],
                'predicted_active_range':[float(p[q>=threshold].min()),float(p[q>=threshold].max())]}
        result['types'][kind]=record
        tr=np.flatnonzero(train.mtype==kind)
        w=np.sum(train.grid[tr]>0,axis=1);n=np.isfinite(train.grid[tr]).sum(1)
        result['train_win_counts'][kind]={'machines':len(tr),'pooled_win_rate':float(w.sum()/n.sum()),
            'win_count_quantiles':np.quantile(w,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'machines_at_most10_wins':int((w<=10).sum()),'machines_at_most20_wins':int((w<=20).sum()),
            'conditions':{}}
        for condition in sorted(set(train.condition[tr])):
            c=train.condition[tr]==condition
            result['train_win_counts'][kind]['conditions'][condition]={'machines':int(c.sum()),
                'pooled_win_rate':float(w[c].sum()/n[c].sum()),
                'win_count_quantiles':np.quantile(w[c],[0,.25,.5,.75,1]).tolist()}
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    for kind,r in result['types'].items():
        print(kind,r['summary'],'deciles',r['deciles'],'unique',r['unique_round12'],flush=True)
    print('JACKPOT TRAIN',result['train_win_counts']['jackpot'],flush=True)
    print('JACKPOT LEVELS',result['types']['jackpot']['true_level'],flush=True)


if __name__=='__main__':main()
