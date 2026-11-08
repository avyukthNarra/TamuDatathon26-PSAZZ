"""Train-only rule discovery with common rows, BIC, and temporal holdout checks."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import xlogy

HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from harness import load_split

PAIRS=list(itertools.combinations(range(1,9),2))
TRIPLES=list(itertools.combinations(range(1,9),3))
KNOWN=((1,3),(2,5),(4,7))
PAY={0:0,1:1,2:2,5:3,20:4}


def table_score(keys,y):
    _,key=np.unique(keys,return_inverse=True)
    n=np.bincount(key);w=np.bincount(key,weights=y)
    p=w/n
    ll=float(np.sum(xlogy(w,p)+xlogy(n-w,1-p)))
    rate=y.mean();null=float(xlogy(y.sum(),rate)+xlogy(len(y)-y.sum(),1-rate))
    gain=2*(ll-null)-(len(n)-1)*np.log(len(y))
    return float(gain),{'n':n.astype(int).tolist(),'wins':w.astype(int).tolist(),'rates':p.tolist(),'cells':len(n)}


def designs(lags):
    win=lags>0;big=lags>=5
    candidates=[]
    for coding,values in [('win',win),('big',big)]:
        for a,b in PAIRS:
            key=(values[:,a-1]==values[:,b-1]).astype(int)
            candidates.append((f'{coding}_pair',{ 'lags':[a,b],'coding':coding,'kind':'parity'},key))
        for lags3 in TRIPLES:
            key=np.sum(values[:,np.array(lags3)-1],axis=1)%2
            candidates.append((f'{coding}_triple',{'lags':list(lags3),'coding':coding,'kind':'parity'},key))
        for a,b in PAIRS:
            key=values[:,a-1].astype(int)*2+values[:,b-1]
            candidates.append((f'{coding}_full_pair',{'lags':[a,b],'coding':coding,'kind':'full'},key))
    classes=np.vectorize(PAY.get)(lags)
    for a,b in PAIRS:
        candidates.append(('payout_pair',{'lags':[a,b],'coding':'payout','kind':'full'},classes[:,a-1]*5+classes[:,b-1]))
    bits=np.column_stack([win[:,a-1]==win[:,b-1] for a,b in KNOWN]).astype(int)
    for flip in itertools.product((0,1),repeat=3):
        key=(bits^np.array(flip)).sum(1)
        candidates.append(('active_pair_count',{'known_pairs':KNOWN,'flips':flip,'kind':'count'},key))
    candidates.append(('joint_three_pairs',{'known_pairs':KNOWN,'kind':'joint'},bits@np.array([1,2,4])))
    return candidates


def temporal_score(key,y,fit_rows,test_rows):
    allkeys,encoded=np.unique(key,return_inverse=True)
    n=np.bincount(encoded[fit_rows],minlength=len(allkeys))
    w=np.bincount(encoded[fit_rows],weights=y[fit_rows],minlength=len(allkeys))
    prior=(y[fit_rows].sum()+.5)/(len(fit_rows)+1.)
    p=(w+prior)/(n+1.)
    pred=p[encoded[test_rows]]
    yy=y[test_rows]
    return {'logloss_sum':float(-np.sum(yy*np.log(pred)+(1-yy)*np.log(1-pred))),
            'brier_sum':float(np.sum((pred-yy)**2)),'rows':len(test_rows)}


def main():
    s=load_split('train');records=[]
    for i in range(s.n):
        seq=s.grid[i,:s.cutoff[i]]
        ticks=np.arange(8,len(seq));lagmat=np.column_stack([seq[ticks-lag] for lag in range(1,9)])
        good=np.isfinite(seq[ticks])&np.isfinite(lagmat).all(1)
        ticks,lagmat=ticks[good],lagmat[good]
        if len(ticks)<16:continue
        y=(seq[ticks]>0).astype(float)
        candidates=designs(lagmat)
        rr={'query_id':str(s.qid[i]),'type':str(s.mtype[i]),'condition':str(s.condition[i]),
            'n_observed':int(np.isfinite(seq).sum()),'total_wins':int(np.nansum(seq>0)),
            'matched_rows':len(y),'matched_wins':int(y.sum()),'families':{}}
        split=max(8,int(.7*len(y)));fit=np.arange(split);test=np.arange(split,len(y))
        byfamily={}
        for family,description,key in candidates:
            gain,table=table_score(key,y)
            fitgain,_=table_score(key[fit],y[fit])
            entry={'gain_bic':gain,'model':description,'table':table}
            if family in ('win_pair','big_pair'):
                entry['polarity']='match_high' if len(table['rates'])==2 and table['rates'][1]>table['rates'][0] else 'differ_high_or_one_cell'
            byfamily.setdefault(family,[]).append((entry,fitgain,key))
        byfamily['known_win_pair']=[row for row in byfamily['win_pair'] if tuple(row[0]['model']['lags']) in KNOWN]
        for family,rows in byfamily.items():
            best=max(rows,key=lambda r:r[0]['gain_bic'])[0]
            chosen=max(rows,key=lambda r:r[1])
            rr['families'][family]={**best,'holdout_model':chosen[0]['model'],
                                    'holdout':temporal_score(chosen[2],y,fit,test)}
        records.append(rr)
    summary={}
    for kind in sorted(set(r['type'] for r in records)):
        rows=[r for r in records if r['type']==kind]
        summary[kind]={'machines':len(rows),'families':{},'win_pair_winners':{},'polarity':{},'known_pair_fraction':0}
        known=0
        for row in rows:
            win=row['families']['win_pair'];pair=str(win['model']['lags']);pol=win['polarity']
            summary[kind]['win_pair_winners'][pair]=summary[kind]['win_pair_winners'].get(pair,0)+1
            summary[kind]['polarity'][pol]=summary[kind]['polarity'].get(pol,0)+1
            known+=tuple(win['model']['lags']) in KNOWN
        summary[kind]['known_pair_fraction']=known/len(rows)
        for family in rows[0]['families']:
            gains=np.array([r['families'][family]['gain_bic'] for r in rows])
            held=[r['families'][family]['holdout'] for r in rows]
            n=sum(r['rows'] for r in held)
            delta=np.array([r['families'][family]['gain_bic']-r['families']['known_win_pair']['gain_bic'] for r in rows])
            summary[kind]['families'][family]={'median_gain_bic':float(np.median(gains)),
                'strong_gain_over_null_count':int((gains>10).sum()),
                'median_bic_advantage_over_known_pair':float(np.median(delta)),
                'beats_known_by10_count':int((delta>10).sum()),
                'temporal_logloss':sum(r['logloss_sum'] for r in held)/n,
                'temporal_brier':sum(r['brier_sum'] for r in held)/n}
        summary[kind]['conditions']={}
        for condition in sorted(set(r['condition'] for r in rows)):
            cr=[r for r in rows if r['condition']==condition]
            strong=[r for r in cr if r['families']['win_pair']['gain_bic']>10]
            counts={}
            for r in strong:
                label=str(r['families']['win_pair']['model']['lags'])
                counts[label]=counts.get(label,0)+1
            summary[kind]['conditions'][condition]={'machines':len(cr),
                'strong_win_pair_count':len(strong),
                'strong_known_pair_count':sum(tuple(r['families']['win_pair']['model']['lags']) in KNOWN for r in strong),
                'strong_winner_counts':counts,
                'strong_polarity_counts':{pol:sum(r['families']['win_pair']['polarity']==pol for r in strong)
                    for pol in ('match_high','differ_high_or_one_cell')},
                'temporal_logloss':{fam:sum(r['families'][fam]['holdout']['logloss_sum'] for r in cr)/sum(r['families'][fam]['holdout']['rows'] for r in cr)
                    for fam in cr[0]['families']}}
        j=[r for r in rows if r['families']['known_win_pair']['gain_bic']>10]
        high=[];low=[]
        for r in j:
            tab=r['families']['known_win_pair']['table']
            if len(tab['rates'])==2 and min(tab['n'])>=10:
                high.append(max(tab['rates']));low.append(min(tab['rates']))
        summary[kind]['well_supported_known_cell_rates']={'machines':len(high),
            'high_deciles':np.quantile(high,np.arange(0,1.01,.1)).tolist() if high else [],
            'low_deciles':np.quantile(low,np.arange(0,1.01,.1)).tolist() if low else []}
    (OUT/'structure.json').write_text(json.dumps({'scope':'train histories only','matched_rows':'current outcome and all lags 1..8 observed, t>=8',
        'bic':'2*(cell Bernoulli MLE loglik-null)-extra occupied cells*log(n); model search uncorrected',
        'temporal':'choose each family by BIC on first 70% matched ticks; smoothed tables scored on final 30%',
        'summary':summary,'machines':records},indent=2)+'\n')
    print(json.dumps(summary['jackpot'],indent=2),flush=True)


if __name__=='__main__':main()
