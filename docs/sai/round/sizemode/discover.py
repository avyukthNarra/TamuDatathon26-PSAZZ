"""TRAIN-only size-rule screening, BIC, held-out history tail and peer checks."""
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'docs/sai'),str(ROOT/'mainFiles')]
from harness import load_split
from slot_forecasting.io import load_inputs,read_queries,read_targets
import mixer6_source as pm

RULES=[]
for feature in ('big','win'):
    for lag in range(1,9):
        for polarity in (1,-1):RULES.append((f'single_{feature}',feature,(lag,),'single',polarity))
    for pair in itertools.combinations(range(1,9),2):
        for op in ('xor','and','or'):
            for polarity in (1,-1):RULES.append((f'{op}_{feature}',feature,pair,op,polarity))
FAMILIES=tuple(dict.fromkeys(rule[0] for rule in RULES))+('static','hmm')
HMM_SWITCH=(.005,.02,.08,.25,.5)


def modes_array(s,modes):
    output=[]
    for kind in s.mtype:
        a=np.asarray(modes[kind]['modes'],float);a/=a.sum(1,keepdims=True)
        if a.shape!=(2,4):raise ValueError('Screen assumes two modes per type')
        output.append(a[np.argsort(a@np.array([1,2,5,20]))])
    return np.asarray(output)


def rule_probability(own,rates):
    """Predict P(big mode) for all hypotheses BEFORE seeing each current spin."""
    T=len(own);state={}
    observed=np.isfinite(own);count=np.r_[0,np.cumsum(observed)[:-1]]
    for feature in ('big','win'):
        value=own>=5 if feature=='big' else own>0
        successes=np.r_[0,np.cumsum(observed&value)[:-1]]
        running=(successes+2*rates[feature])/(count+2)
        lagged=np.empty((T,8))
        for lag in range(1,9):
            previous=np.r_[np.full(lag,np.nan),own][:T]
            known=np.isfinite(previous)
            event=previous>=5 if feature=='big' else previous>0
            lagged[:,lag-1]=np.where(known,event,running)
        state[feature]=lagged
    result=np.empty((T,len(RULES)))
    for j,(_,feature,lags,op,polarity) in enumerate(RULES):
        a=state[feature][:,lags[0]-1]
        if op=='single':p=a
        else:
            b=state[feature][:,lags[1]-1]
            p=a+b-2*a*b if op=='xor' else a*b if op=='and' else a+b-a*b
        result[:,j]=p if polarity==1 else 1-p
    return result


def static_fit(emissions):
    p=.5
    for _ in range(100):
        den=(1-p)*emissions[:,0]+p*emissions[:,1]
        nxt=float((p*emissions[:,1]/den).mean()) if len(emissions) else .5
        if abs(nxt-p)<1e-9:break
        p=np.clip(nxt,1e-6,1-1e-6)
    return float(np.clip(p,1e-6,1-1e-6))


def baseline_size_trace(s,modes,keys,eta=4.):
    """Exact size half of mixer6; no win-head changes or target fitting."""
    n=s.n;keys=tuple(key[0] if key else 0 for key in keys);nk=len(keys)
    cls=np.full(s.grid.shape,-1,int)
    for k,pay in enumerate((0,1,2,5,20)):cls[s.grid==pay]=k
    ma=modes_array(s,modes);llr=np.log(np.maximum(ma[:,1],1e-3)/np.maximum(ma[:,0],1e-3))
    counts=np.zeros((n,nk,5,4));machine=np.zeros((n,4));logw=np.zeros((n,nk))
    trace=np.zeros((n,int(s.cutoff.max())+1,4));ii=np.arange(n)[:,None];jj=np.arange(nk)[None,:]
    for t in range(trace.shape[1]):
        cell=np.array([np.zeros(n,int) if lag==0 else cls[:,t-lag] if t>=lag else np.full(n,-1) for lag in keys]).T
        cc=counts[ii,jj,np.maximum(cell,0)].copy()
        empty=(cc.sum(2)==0)|(cell<0);cc=np.where(empty[...,None],machine[:,None,:],cc)
        p=expit(np.clip((cc*llr[:,None,:]).sum(2),-50,50))
        pred=ma[:,0,None,:]+p[...,None]*(ma[:,1,None,:]-ma[:,0,None,:])
        weight=np.exp(logw-logw.max(1,keepdims=True))
        trace[:,t]=(weight[...,None]*pred).sum(1)/weight.sum(1)[:,None]
        if t==cls.shape[1]:break
        value=cls[:,t];won=value>0;target=np.eye(4)[np.maximum(value-1,0)]
        logw-=eta*((pred-target[:,None,:])**2).sum(2)*won[:,None]
        counts*=2**(-1/80);machine*=2**(-1/80)
        for k in range(4):
            use=won&(value==k+1);machine[use,k]+=1
            counts[ii,jj,np.maximum(cell,0),k]+=use[:,None]*(cell>=0)
    return trace


def hmm_likelihood(s,ma,split_tick):
    """Five symmetric transition rates; transitions occur even on missing/loss ticks."""
    n=s.n;ps=np.asarray(HMM_SWITCH)[None,:]
    state=np.full((n,len(ps[0])),.5);full=np.zeros_like(state);prefix=np.zeros_like(state)
    mode_trace=np.full((n,s.grid.shape[1]+1,len(HMM_SWITCH)),np.nan)
    for t in range(s.grid.shape[1]):
        mode_trace[:,t]=state
        value=s.grid[:,t];won=np.isfinite(value)&(value>0)
        cls=np.zeros(n,int)
        for k,pay in enumerate((1,2,5,20)):cls[value==pay]=k
        small=ma[np.arange(n),0,cls][:,None];big=ma[np.arange(n),1,cls][:,None]
        evidence=(1-state)*small+state*big
        ll=np.log(np.maximum(evidence,1e-12))*won[:,None]
        full+=ll;prefix+=ll*(t<split_tick)[:,None]
        updated=state*big/np.maximum(evidence,1e-12)
        state=np.where(won[:,None],updated,state)
        state=state*(1-ps)+(1-state)*ps
    return full,prefix,mode_trace


def bootstrap_episode_delta(s,values,seed=293):
    per=np.array([np.nanmean(values[s.episode==e]) for e in s.episodes])
    rng=np.random.RandomState(seed)
    boots=np.array([rng.choice(per,len(per)).mean() for _ in range(1000)])
    return {'mean':float(per.mean()),'ci95':np.percentile(boots,[2.5,97.5]).tolist()}


def main():
    started=time.time();train=load_split('train');data=ROOT/'mainFiles/public'
    tq,th=load_inputs(data/'train');vq=read_queries(data/'validation/queries.csv')
    modes=pm.size_modes_from_targets(read_targets(data/'validation/targets.csv'),vq)
    keys=pm.discover_size_keys(tq,th);ma=modes_array(train,modes)
    rates={}
    for kind in sorted(set(train.mtype)):
        x=train.grid[train.mtype==kind];obs=np.isfinite(x)
        rates[kind]={'big':float((x>=5).sum()+.5)/float(obs.sum()+1),'win':float((x>0).sum()+.5)/float(obs.sum()+1)}
    cut=(train.cutoff*.7).astype(int);nr=len(RULES);nf=len(FAMILIES)
    ll=np.zeros((train.n,nr));prefix=np.zeros_like(ll);nw=np.zeros(train.n);tail_n=np.zeros(train.n)
    baseline=baseline_size_trace(train,modes,keys)
    baseline_tail=np.zeros(train.n);family_tail=np.zeros((train.n,nf))
    family_full=np.zeros_like(family_tail);family_prefix=np.zeros_like(family_tail)
    family_bic=np.zeros_like(family_tail);full_winner=np.zeros((train.n,nf),int)
    chosen_rule=np.zeros(train.n,int);emission_signal=np.full(train.grid.shape,np.nan)
    for i in range(train.n):
        own=train.grid[i,:train.cutoff[i]];prob=rule_probability(own,rates[train.mtype[i]])
        ticks=np.flatnonzero(np.isfinite(own)&(own>0));nw[i]=len(ticks)
        tail=ticks>=cut[i];tail_n[i]=tail.sum()
        cls=np.array([[1,2,5,20].index(own[t]) for t in ticks],dtype=int)
        small=ma[i,0,cls];big=ma[i,1,cls]
        observations=np.log(np.maximum(small[:,None]+prob[ticks]*(big-small)[:,None],1e-12))
        ll[i]=observations.sum(0);prefix[i]=observations[~tail].sum(0)
        baseline_tail[i]=np.log(np.maximum(baseline[i,ticks[tail],cls[tail]],1e-12)).sum()
        for j,family in enumerate(FAMILIES[:-2]):
            candidates=np.flatnonzero(np.array([x[0] for x in RULES])==family)
            selected=candidates[np.argmax(prefix[i,candidates])]
            chosen=candidates[np.argmax(ll[i,candidates])]
            family_full[i,j]=ll[i,chosen];family_prefix[i,j]=prefix[i,selected]
            family_tail[i,j]=ll[i,selected]-prefix[i,selected]
            family_bic[i,j]=-2*ll[i,chosen] # zero fitted continuous parameters
            full_winner[i,j]=chosen
        emissions=np.column_stack((small,big));p=static_fit(emissions)
        pp=static_fit(emissions[~tail])
        full_static=np.log((1-p)*small+p*big).sum()
        prefix_static=np.log((1-pp)*small[~tail]+pp*big[~tail]).sum()
        tail_static=np.log((1-pp)*small[tail]+pp*big[tail]).sum()
        # Include fixed50/50 as a zero-parameter static alternative.
        half_ll=np.log((small+big)*.5);fixed_full=half_ll.sum();fixed_prefix=half_ll[~tail].sum()
        js=nf-2;family_full[i,js]=max(full_static,fixed_full)
        fitted_bic=-2*full_static+np.log(max(len(ticks),1));fixed_bic=-2*fixed_full
        family_bic[i,js]=min(fitted_bic,fixed_bic)
        # Select static form on PREFIX BIC only, then evaluate tail.
        fit_pre_bic=-2*prefix_static+np.log(max((~tail).sum(),1))
        if fit_pre_bic < -2*fixed_prefix:
            family_prefix[i,js]=prefix_static;family_tail[i,js]=tail_static
        else:
            family_prefix[i,js]=fixed_prefix;family_tail[i,js]=half_ll[tail].sum()
        emission_signal[i,ticks]=np.log(np.maximum(big,1e-12)/np.maximum(small,1e-12))
        if i%128==0:print('SCREENED',i,flush=True)
    hfull,hpre,htrace=hmm_likelihood(train,ma,cut);j=nf-1
    hi=hfull.argmax(1);hp=hpre.argmax(1)
    family_full[:,j]=hfull[np.arange(train.n),hi];family_prefix[:,j]=hpre[np.arange(train.n),hp]
    family_tail[:,j]=(hfull-hpre)[np.arange(train.n),hp]
    family_bic[:,j]=-2*family_full[:,j]+np.log(np.maximum(nw,1))
    winner=family_bic.argmin(1)
    eligible=nw>=12
    # Discrete family search penalty supplements raw BIC; it is not called BIC.
    family_size=np.array([sum(x[0]==f for x in RULES) if f not in ('static','hmm') else 1 for f in FAMILIES])
    penalized=family_bic+2*np.log(family_size)[None,:]
    tally={};heldout={};lag_tally={}
    for kind in sorted(set(train.mtype)):
        rows=(train.mtype==kind)&eligible
        tally[kind]={'bic':{f:int((winner[rows]==j).sum()) for j,f in enumerate(FAMILIES)},
                     'search_penalized':{f:int((penalized.argmin(1)[rows]==j).sum()) for j,f in enumerate(FAMILIES)}}
        lag_tally[kind]={}
        for i in np.flatnonzero(rows):
            j=winner[i]
            if j<nf-2:
                rule=RULES[full_winner[i,j]];key=str(rule)
                lag_tally[kind][key]=lag_tally[kind].get(key,0)+1
    for j,f in enumerate(FAMILIES):
        # Per-machine tail LL/win: positive difference means better than baseline.
        delta=np.divide(family_tail[:,j]-baseline_tail,tail_n,out=np.full(train.n,np.nan),where=tail_n>0)
        heldout[f]=bootstrap_episode_delta(train,delta)
        heldout[f]['mean_tail_loglik_per_win']=float(family_tail[:,j].sum()/tail_n.sum())
    # Exact rule identity and emission co-movement across episode peers.
    filtered=htrace[np.arange(train.n)[:,None],np.arange(htrace.shape[1])[None,:],hi[:,None]]
    filtered[np.arange(filtered.shape[1])[None,:]>=train.cutoff[:,None]]=np.nan
    for kind in sorted(set(train.mtype)):
        ids=train.mtype==kind;filtered[ids]-=np.nanmean(filtered[ids])
    identity=np.empty(train.n,object)
    for i,j in enumerate(winner):
        identity[i]=str(RULES[full_winner[i,j]]) if j<nf-2 else f'{FAMILIES[j]}:{HMM_SWITCH[hi[i]] if j==nf-1 else "static"}'
    # Type-center the noisy one-win mode evidence, using TRAIN only.
    for kind in sorted(set(train.mtype)):
        ids=train.mtype==kind;emission_signal[ids]-=np.nanmean(emission_signal[ids])
    pairs=np.array([(a,b) for ep in train.episodes for a in np.flatnonzero(train.episode==ep) for b in np.flatnonzero(train.episode==ep) if a<b and eligible[a] and eligible[b]]);aa,bb=pairs.T
    def peer_metrics(order):
        a,b=order[aa],order[bb]
        product=emission_signal[a]*emission_signal[b]
        return np.array([(identity[a]==identity[b]).mean(),(winner[a]==winner[b]).mean(),np.nanmean(product),np.nanmean(filtered[a]*filtered[b])])
    actual=peer_metrics(np.arange(train.n));rng=np.random.RandomState(993);null=[]
    strata=[np.flatnonzero((train.mtype==kind)&(train.condition==condition)&eligible) for kind in sorted(set(train.mtype)) for condition in sorted(set(train.condition))]
    for _ in range(300):
        order=np.arange(train.n)
        for rows in strata:order[rows]=rng.permutation(rows)
        null.append(peer_metrics(order))
    null=np.array(null);peer={key:{'actual':float(actual[k]),'null_mean':float(null[:,k].mean()),'null_ci95':np.percentile(null[:,k],[2.5,97.5]).tolist(),'upper_tail_p':float((1+(null[:,k]>=actual[k]).sum())/301)} for k,key in enumerate(('exact_rule_agreement','family_agreement','simultaneous_win_mode_evidence_covariance','hmm_filtered_mode_covariance'))}
    lag_participation={}
    for kind in sorted(set(train.mtype)):
        lag_participation[kind]={str(k):0 for k in range(1,9)}
        for i in np.flatnonzero((train.mtype==kind)&eligible):
            j=winner[i]
            if j<nf-2:
                for lag in RULES[full_winner[i,j]][2]:lag_participation[kind][str(lag)]+=1
    focused={}
    for condition in sorted(set(train.condition)):
        for kind in sorted(set(train.mtype)):
            use=(train.mtype==kind)&(train.condition==condition)&(tail_n>0)
            focused[f'{condition}:{kind}']={family:float(((family_tail[:,j]-baseline_tail)/np.maximum(tail_n,1))[use].mean()) for j,family in enumerate(FAMILIES)}
    output={'winning_lag_participation':lag_participation,'train_tail_delta_by_condition_type':focused,'n_train_machines':train.n,'n_eligible_BIC_tally_machines':int(eligible.sum()),'BIC_tally_min_wins':12,'n_deterministic_rules':nr,'families':FAMILIES,'size_keys':keys,
        'size_modes':modes,'fallback_rates':rates,'tally':tally,'winning_lags':{k:dict(sorted(x.items(),key=lambda z:-z[1])) for k,x in lag_tally.items()},
        'tail_holdout':heldout,'baseline_tail_loglik_per_win':float(baseline_tail.sum()/tail_n.sum()),
        'peer_checks':peer,'seconds':time.time()-started,
        'method':'BIC=-2logL+dlog(n_wins). Deterministic rules/fixed static have d0; fitted static and fitted symmetric HMM switch probability have d1. Extra discrete search penalty 2log(number rules) reported separately. Fit first70% calendar ticks; score last30% with no parameter selection on tail. Modes inherited from validation; all rule selection and peer evidence TRAIN only.'}
    (OUT/'train_discovery.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(OUT/'train_scores.npz',rule_ll=ll,rule_prefix_ll=prefix,family_bic=family_bic,family_tail_ll=family_tail,baseline_tail_ll=baseline_tail,tail_n=tail_n,winner=winner)
    print('RESULT',json.dumps(output),flush=True)

if __name__=='__main__':main()
