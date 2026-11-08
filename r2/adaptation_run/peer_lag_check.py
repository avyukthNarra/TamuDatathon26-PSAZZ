import csv, numpy as np, itertools
K='/Users/avynarra/Documents/Datathon/R2_kit40/public/validation/'
def load(p): return {r['query_id']:np.array([float(r[c]) for c in ('p_0','p_1','p_2','p_5','p_20')]) for r in csv.DictReader(open(p))}
T,R=load(K+'targets.csv'),load(K+'reference.csv'); Bl=load('/Users/avynarra/Documents/Datathon/R2_members/blend_val.csv')
cond={r['query_id']:r['condition'] for r in csv.DictReader(open(K+'conditions.csv'))}
Q={r['query_id']:r for r in csv.DictReader(open(K+'queries.csv'))}
win={}
for r in csv.DictReader(open(K+'history.csv')): win[(r['machine_id'],int(r['tick']))]=float(r['payout'])>0
ids=sorted(T); E=np.array([Q[q]['episode_id'] for q in ids]); C=np.array([cond[q] for q in ids])
# condition-balanced folds (orch/common.folds)
ep_c={Q[q]['episode_id']:cond[q] for q in ids}; A=set()
for c in sorted(set(ep_c.values())):
  for i,e in enumerate(sorted(e for e in ep_c if ep_c[e]==c)):
    if i%2==0: A.add(e)
H=np.array([0 if e in A else 1 for e in E])
TT=np.array([T[q] for q in ids]); RR=np.array([R[q] for q in ids]); BB=np.array([Bl[q] for q in ids])
def peer(lag):
  out=[]
  for q in ids:
    mid=Q[q]['machine_id']; ep=Q[q]['episode_id']; t=int(Q[q]['cutoff'])-lag
    v=[win[(f"{ep}-m{m:03d}",t)] for m in range(8) if f"{ep}-m{m:03d}"!=mid and (f"{ep}-m{m:03d}",t) in win]
    out.append(np.mean(v) if v else 0.5)
  return np.array(out)
def pts(P,mask):
  r=[]
  for c in sorted(set(C)):
    m=mask&(C==c); f=lambda X: np.mean([((X[m&(E==e)]-TT[m&(E==e)])**2).sum(1).mean() for e in set(E[m])]); r.append(f(P)/f(RR))
  return 40/(1+np.mean(r))
pw=1-BB[:,0]; size=BB[:,1:]/BB[:,1:].sum(1,keepdims=True); z=np.log(np.clip(pw,1e-5,1-1e-5)/(1-np.clip(pw,1e-5,1-1e-5)))
def build(f,a,b):
  p=1/(1+np.exp(-(z+a+b*f))); P=np.column_stack([1-p,p[:,None]*size]); P=np.maximum(P,1e-4); return P/P.sum(1,keepdims=True)
print('base all %.4f A %.4f B %.4f'%(pts(BB,H>=0),pts(BB,H==0),pts(BB,H==1)))
grid=[(a,b) for a in np.arange(-0.6,0.21,0.05) for b in np.arange(-0.4,1.61,0.1)]
for lag in range(1,9):
  f=peer(lag); out=[]
  for h in (0,1):
    a,b=max(grid,key=lambda ab: pts(build(f,*ab),H==h)); o=1-h
    out.append((round(a,2),round(b,2),pts(build(f,a,b),H==o)-pts(BB,H==o)))
  print('lag',lag,'fitA->B %+.4f (a %.2f b %.2f)  fitB->A %+.4f (a %.2f b %.2f)'%(out[0][2],out[0][0],out[0][1],out[1][2],out[1][0],out[1][1]))
