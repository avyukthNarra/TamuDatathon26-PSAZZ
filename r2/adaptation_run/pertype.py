import csv, itertools, numpy as np
K='/Users/avynarra/Documents/Datathon/R2_kit40/public/validation/'; M='/Users/avynarra/Documents/Datathon/R2_members/'
def load(p): return {r['query_id']:np.array([float(r[c]) for c in ('p_0','p_1','p_2','p_5','p_20')]) for r in csv.DictReader(open(p))}
T,R=load(K+'targets.csv'),load(K+'reference.csv')
cond={r['query_id']:r['condition'] for r in csv.DictReader(open(K+'conditions.csv'))}
Q={r['query_id']:r for r in csv.DictReader(open(K+'queries.csv'))}
ids=sorted(T); eps=sorted({Q[q]['episode_id'] for q in ids}); half={e:i%2 for i,e in enumerate(eps)}
H=np.array([half[Q[q]['episode_id']] for q in ids]); C=np.array([cond[q] for q in ids]); E=np.array([Q[q]['episode_id'] for q in ids]); TY=np.array([Q[q]['machine_type'] for q in ids])
TT=np.array([T[q] for q in ids]); RR=np.array([R[q] for q in ids])
names=['v13_short','exact','sysid']; mem={n:load(M+n+'.csv') for n in names}
def L(p): p=np.clip(p,1e-5,1-1e-5); return np.log(p/(1-p))
Z=np.array([L(1-np.array([mem[n][q][0] for q in ids])) for n in names])
S=np.array([mem['v13_short'][q][1:] for q in ids]); S=S/S.sum(1,keepdims=True)
def pts(P,mask):
  r=[]
  for c in sorted(set(C)):
    m=mask&(C==c); f=lambda X: np.mean([((X[m&(E==e)]-TT[m&(E==e)])**2).sum(1).mean() for e in set(E[m])]); r.append(f(P)/f(RR))
  return 40/(1+np.mean(r))
def build(Wt, temp=None):
  z=np.zeros(len(ids))
  for t,w in Wt.items(): mk=TY==t; z[mk]=(np.array(w)[:,None]*Z[:,mk]).sum(0)
  pw=1/(1+np.exp(-z)); P=np.column_stack([1-pw,pw[:,None]*S]); P=np.maximum(P,1e-4); return P/P.sum(1,keepdims=True)
types=sorted(set(TY)); eq=(1/3,1/3,1/3)
grid=[tuple(np.array(w)/10) for w in itertools.product(range(11),repeat=3) if sum(w)==10]
base=build({t:eq for t in types}); print('default A %.4f B %.4f'%(pts(base,H==0),pts(base,H==1)))
for h in (0,1):
  W={t:eq for t in types}
  for t in types:  # coordinate-wise per type on fitting half
    W[t]=max(grid,key=lambda w: pts(build({**W,t:w}),H==h))
  o=1-h; print('fit',"AB"[h],{t:tuple(round(x,1) for x in W[t]) for t in types},'-> held-out',"AB"[o],'%.4f vs default %.4f'%(pts(build(W),H==o),pts(base,H==o)))
# simple shrunk rule: sysid heavier on jackpot/balanced, lighter on steady
for lab,W in [('sysid .5 on jackpot+balanced, .2 steady',{'jackpot':(.25,.25,.5),'balanced':(.25,.25,.5),'steady':(.4,.4,.2)}),('steady drop sysid',{'jackpot':eq,'balanced':eq,'steady':(.5,.5,0)})]:
  P=build(W); print(lab,'A %.4f B %.4f'%(pts(P,H==0),pts(P,H==1)))
