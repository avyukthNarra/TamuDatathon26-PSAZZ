exec(open('pertype.py').read().split("types=sorted")[0])
# detect short history from history gaps (no labels used)
import collections
hist=collections.defaultdict(set)
for r in csv.DictReader(open(K+'history.csv')): hist[r['machine_id']].add(int(r['tick']))
SH=np.array([len(hist[Q[q]['machine_id']])<int(Q[q]['cutoff'])*0.6 for q in ids])
print('detected short', SH.sum(), 'true short', (C=='short_history').sum(), 'agree', (SH==(C=='short_history')).mean())
def build2(wl,ws,tl,ts):
  z=np.where(SH,(np.array(ws)[:,None]*Z).sum(0)*ts,(np.array(wl)[:,None]*Z).sum(0)*tl)
  pw=1/(1+np.exp(-z)); P=np.column_stack([1-pw,pw[:,None]*S]); P=np.maximum(P,1e-4); return P/P.sum(1,keepdims=True)
eq=(1/3,1/3,1/3); base=build2(eq,eq,1,1); print('default A %.4f B %.4f'%(pts(base,H==0),pts(base,H==1)))
grid=[tuple(np.array(w)/10) for w in itertools.product(range(11),repeat=3) if sum(w)==10]
temps=(0.8,0.9,1.0,1.1,1.2)
for h in (0,1):
  best=max(((w,t) for w in grid for t in temps),key=lambda x: pts(build2(eq,x[0],1,x[1]),H==h))
  o=1-h; print('short-only fit',"AB"[h],best,'-> held-out',"AB"[o],'%.4f vs %.4f'%(pts(build2(eq,best[0],1,best[1]),H==o),pts(base,H==o)))
for t in temps: P=build2(eq,eq,1,t); print('short T',t,'A %.4f B %.4f'%(pts(P,H==0),pts(P,H==1)))
