"""Per machine: which lag feature sets the size mode? Fit on its own history (wins only), predict mode at cutoff, compare to truth."""
import numpy as np, collections, itertools
from common import *
v=load('validation'); T=v['targets']; Qd={q['query_id']:q for q in v['Q']}
BIG={'steady':np.array([.455,.436,.107,.002]),'balanced':np.array([.214,.547,.205,.034]),'jackpot':np.array([.121,.324,.416,.139])}
SM={'steady':np.array([.754,.197,.048,.001]),'balanced':np.array([.5,.348,.13,.022]),'jackpot':np.array([.335,.245,.315,.105])}
IDX={1:0,2:1,5:2,20:3}
def tmode(q):
  t=T[q]; s=t[1:]/(1-t[0]); ty=Qd[q]['machine_type']; return int(((s-BIG[ty])**2).sum()<((s-SM[ty])**2).sum())
FEATS={'win':lambda y:int(y>0),'big':lambda y:int(y>=5),'cls3':lambda y:0 if y==0 else (1 if y<=2 else 2),'cls':lambda y:y}
res=collections.Counter(); best_used=collections.Counter(); confs=[]
for q in v['Q']:
  qid=q['query_id']; ty=q['machine_type']; h=v['H'][q['machine_id']]; c=int(q['cutoff'])
  lb=np.log(BIG[ty]+1e-4); ls=np.log(SM[ty]+1e-4)
  best=None
  for k in range(1,9):
    for fn,f in FEATS.items():
      cells=collections.defaultdict(lambda:[0.,0.])  # per feature value: LL big, LL small
      for t,y in h.items():
        if t>=c or y==0 or (t-k) not in h: continue
        val=f(h[t-k]); cells[val][0]+=lb[IDX[y]]; cells[val][1]+=ls[IDX[y]]
      ll=sum(max(a,b) for a,b in cells.values()); nparam=len(cells)
      score=ll-0.5*nparam*np.log(max(2,sum(1 for t,y in h.items() if t<c and y>0)))
      if best is None or score>best[0]:
        best=(score,k,fn,{val:int(a>b) for val,(a,b) in cells.items()},{val:a-b for val,(a,b) in cells.items()})
  # constant-mode alternative
  a=sum(lb[IDX[y]] for t,y in h.items() if t<c and y>0); b=sum(ls[IDX[y]] for t,y in h.items() if t<c and y>0)
  const=max(a,b)-0.5*np.log(max(2,sum(1 for t,y in h.items() if t<c and y>0)))
  if const>=best[0]:
    pred=int(a>b); best_used['const']+=1; margin=a-b
  else:
    _,k,fn,mp,llr=best; best_used[(k,fn)]+=1
    y=h.get(c-k)
    if y is None: res['missing']+=1; continue
    val=FEATS[fn](y); pred=mp.get(val,int(a>b)); margin=llr.get(val,0)
  res['n']+=1; res['agree']+=pred==tmode(qid); confs.append((abs(margin),pred==tmode(qid)))
print(res, 'acc', round(res['agree']/res['n'],3))
print(best_used.most_common(12))
confs.sort(); arr=np.array([c[1] for c in confs]); n=len(arr)
for i in range(4): print('margin quartile',i,round(arr[i*n//4:(i+1)*n//4].mean(),3))
