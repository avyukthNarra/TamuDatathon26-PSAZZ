"""Team ensemble experiment: frozen prediction bank, six episode outer folds.

Predeclared jobs: direct convex probability blend; separate win/size simplex
blend; volatility/disagreement softmax gate (lambda .0001,.001,.01 chosen by
three inner episode folds). Reference is frozen TM_COMBINED. No audit scoring.
S5 DEV-trained predictions are excluded from the frozen bank to avoid indirect
label leakage. Upstream teammate tuning on validation is an explicit caveat.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
SAI=HERE.parent
sys.path.insert(0,str(SAI))
from harness import evaluate,diff_ci,blend_gain,check_determinism,to_csv,write_result,read_pred_csv,smoke_split
from harness.scoring import scope_episodes
from harness.stack_w5 import row_weights,fit as factor_fit,combine as factor_combine,parts
from _w4_b4_features import load_data

BANK=['TM_COMBINED','TM_MIXER','TM_RULE','TM_E6','TM_E7','C1','C2','A9','A1_B4','M9b_B4','M2_B4','M3_B4','L5','L7','B4','S1_B4','S2_B4']
WIN=['TM_COMBINED','TM_MIXER','TM_RULE','M9b_B4','M2_B4','L5','L7','TM_E7','S2_B4']
SIZE=['TM_COMBINED','TM_MIXER','TM_RULE','C1','A9','A1_B4','L5','S2_B4','TM_E7']
GATE=['TM_COMBINED','TM_MIXER','TM_RULE','C2','TM_E7']
REG=[.0001,.001,.01]


def folds(v,rows,n,prefix):
    out=np.full(v.n,-1,int)
    for condition in sorted(set(v.condition[rows])):
        eps=sorted(set(v.episode[rows][v.condition[rows]==condition]),key=lambda e:hashlib.sha256((prefix+'|'+str(e)).encode()).hexdigest())
        for j,e in enumerate(eps):out[np.flatnonzero(v.episode==e)]=j%n
    return out


def weights(v,rows):
    w=row_weights(v,set(rows.tolist()))[rows]
    return w/w.sum()


def simplex_fit(A,q,w,anchor=0):
    k=A.shape[1];start=np.zeros(k);start[anchor]=1
    def objective(a):
        err=np.einsum('nkj,k->nj',A,a)-q
        loss=np.sum(w[:,None]*err**2)
        jac=2*np.einsum('n,nj,nkj->k',w,err,A)
        return loss,jac
    r=minimize(objective,start,jac=True,method='SLSQP',bounds=[(0,1)]*k,
        constraints=[{'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(k)}],options={'maxiter':300,'ftol':1e-12})
    if not r.success:raise RuntimeError('simplex optimizer: '+r.message)
    a=np.maximum(r.x,0);return a/a.sum()


def gate_features(D,bank):
    b=D['va'];names=list(b['names'])
    use=[j for j,n in enumerate(names) if n.startswith(('obs_','ess_','drift_','cusum_','type_')) or n in ('moment_4_var','moment_16_var','moment_40_var','moment_4_logmean','moment_40_logmean')]
    x=b['X'][:,use].astype(float);labels=[names[j] for j in use]
    x[:,[j for j,n in enumerate(labels) if n in ('obs_count','obs_cutoff','ess_16','ess_128')]]=np.log1p(x[:,[j for j,n in enumerate(labels) if n in ('obs_count','obs_cutoff','ess_16','ess_128')]])
    pw=np.column_stack([1-bank[n][:,0] for n in GATE])
    extra=np.column_stack([pw.std(1),pw.max(1)-pw.min(1),pw[:,1]-pw[:,2],np.mean((pw-pw[:,:1])**2,axis=1)])
    return np.column_stack([x,extra]),labels+['win_disagreement_sd','win_disagreement_range','mixer_minus_rule','disagreement_to_base']


def gate_fit(X,A,q,w,reg):
    mu=X.mean(0);sd=np.maximum(X.std(0),1e-3);Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    prior=simplex_fit(A,q,w);prior=(prior+.01)/(1+.01*len(prior));logprior=np.log(prior)
    k=A.shape[1];d=Z.shape[1]
    def objective(raw):
        theta=raw.reshape(k,d);s=Z@theta.T+logprior;s-=s.max(1,keepdims=True);g=np.exp(s);g/=g.sum(1,keepdims=True)
        pred=np.einsum('nk,nkj->nj',g,A);err=pred-q
        loss=np.sum(w[:,None]*err**2)+reg*np.sum(theta**2)
        dz=2*w[:,None]*g*np.einsum('nj,nkj->nk',err,A-pred[:,None,:])
        grad=dz.T@Z+2*reg*theta
        return loss,grad.ravel()
    r=minimize(objective,np.zeros(k*d),jac=True,method='L-BFGS-B',bounds=[(-4,4)]*(k*d),options={'maxiter':200,'ftol':1e-12,'gtol':1e-7})
    if not r.success and r.nit<200:raise RuntimeError('gate optimizer: '+r.message)
    return mu,sd,logprior,r.x.reshape(k,d)


def gate_apply(model,X,A):
    mu,sd,prior,theta=model;Z=np.column_stack([np.ones(len(X)),(X-mu)/sd]);s=Z@theta.T+prior;s-=s.max(1,keepdims=True);g=np.exp(s);g/=g.sum(1,keepdims=True)
    return np.einsum('nk,nkj->nj',g,A),g


def choose_reg(v,rows,X,A):
    inner=folds(v,rows,3,'team-inner-v1');out={r:np.zeros((len(rows),5)) for r in REG};positions={int(i):j for j,i in enumerate(rows)}
    for f in range(3):
        tr=rows[inner[rows]!=f];te=rows[inner[rows]==f]
        for reg in REG:
            m=gate_fit(X[tr],A[tr],v.truth[tr],weights(v,tr),reg)
            p,_=gate_apply(m,X[te],A[te]);out[reg][[positions[int(i)] for i in te]]=p
    subset=v.subset(rows);scores={r:evaluate(p,subset,'dev',bootstrap=0)['points'] for r,p in out.items()}
    return max(REG,key=lambda r:scores[r]),scores


def predict_job(kind,D,bank):
    v=D['v'];dev=np.concatenate([v.episode_rows(e) for e in scope_episodes(v,'dev')]);outer=folds(v,dev,6,'team-outer-v1')
    out=bank['TM_COMBINED'].copy();log=[];pw=[parts(bank[n])[0] for n in WIN];sh=[parts(bank[n])[1] for n in SIZE]
    A=np.stack([bank[n] for n in BANK],axis=1);AG=np.stack([bank[n] for n in GATE],axis=1);X,xnames=gate_features(D,bank)
    for f in list(range(6))+[None]:
        tr=dev if f is None else dev[outer[dev]!=f]
        te=np.flatnonzero(outer<0) if f is None else dev[outer[dev]==f]
        assert not set(v.episode[tr])&set(v.episode[te])
        if kind=='direct':
            a=simplex_fit(A[tr],v.truth[tr],weights(v,tr));out[te]=np.einsum('nkj,k->nj',A[te],a);entry={'weights':dict(zip(BANK,a.tolist()))}
        elif kind=='factor':
            a,b=factor_fit(pw,sh,v,tr,row_weights(v,set(tr.tolist())));out[te]=factor_combine([p[te] for p in pw],[s[te] for s in sh],a,b)
            entry={'win_weights':dict(zip(WIN,a.tolist())),'size_weights':dict(zip(SIZE,b.tolist()))}
        else:
            reg,scores=choose_reg(v,tr,X,AG);m=gate_fit(X[tr],AG[tr],v.truth[tr],weights(v,tr),reg);out[te],g=gate_apply(m,X[te],AG[te])
            entry={'reg':reg,'inner_dev_points':scores,'mean_gate':(dict(zip(GATE,g.mean(0).tolist())) if len(g) else {}),'feature_names':xnames}
        entry.update(fold=f,train_episodes=len(set(v.episode[tr])),predicted_episodes=len(set(v.episode[te])));log.append(entry)
    out=np.maximum(out,1e-12);out/=out.sum(1,keepdims=True)
    return out,log


def subset_data(D,sm):
    pos={q:i for i,q in enumerate(D['v'].qid)};ix=np.array([pos[q] for q in sm.qid]);vblock={k:(a[ix] if isinstance(a,np.ndarray) and a.ndim and len(a)==D['v'].n else a) for k,a in D['va'].items()}
    return dict(D,v=sm,va=vblock),ix


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['direct','factor','gate']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):assert os.environ.get(k)=='1',k
    t=time.perf_counter();D=load_data();bank={n:read_pred_csv(SAI/'preds'/f'{n}_validation.csv',D['v']) for n in BANK}
    if a.smoke:
        D,ix=subset_data(D,smoke_split(D['v']));bank={n:P[ix] for n,P in bank.items()}
    P,log=predict_job(a.kind,D,bank)
    if a.smoke:
        print(json.dumps({'kind':a.kind,'smoke_seconds':time.perf_counter()-t,'points':evaluate(P,D['v'],'dev',bootstrap=0)['points']}));return
    Q,_=predict_job(a.kind,D,bank);det=float(np.abs(P-Q).max());assert det<1e-8
    v=D['v'];base=bank['TM_COMBINED'];delta=diff_ci(P,base,v,'dev');bg=blend_gain(P,base,v);bg.pop('oof')
    mid='TEAM_'+a.kind.upper();extra={'vs_TM_COMBINED':delta,'blend_gain':bg,'outer_folds':log,'upstream_selection_caveat':'Frozen teammates already tuned on some/all validation; their old audit is not a blind holdout. Current fitting/scoring uses DEV only.','selection':'Gate lambda selected inside each outer fold by inner episode-OOF scoring; S5 q-trained frozen predictions excluded.','base':'TM_COMBINED','new_audit_scoring':False}
    doc=write_result(mid,'codex-team','Cross-branch '+a.kind+' ensemble',{'kind':a.kind,'bank':BANK},v,P,time.perf_counter()-t,seed=0,determinism=det,extra=extra)
    to_csv(SAI/'preds'/f'{mid}_validation.csv',v,P);(SAI/'grids'/f'{mid}.json').write_text(json.dumps(log,indent=1))
    lines=[mid,f"DEV points {doc['points']}; CI95 {doc['ci95']}",f"Vs TM_COMBINED {delta}",f"Condition ratios {doc['condition_ratios']}",f"Class error shares {doc['class_error_shares']}",f"Blend {bg}",f"Runtime {doc['runtime_seconds']:.3f}s; repeated-fit max difference {det}",'Six outer whole-episode folds; gate uses three inner whole-episode folds.',extra['selection'],extra['upstream_selection_caveat'],'Audit predictions inferred using full DEV fits; audit truth never trained or scored.','Sources and outer-fold weights saved; no verdict assigned.']
    (SAI/'reports'/f'{mid}.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'id':mid,'points':doc['points'],'vs_base':delta,'seconds':doc['runtime_seconds']}))

if __name__=='__main__':main()
