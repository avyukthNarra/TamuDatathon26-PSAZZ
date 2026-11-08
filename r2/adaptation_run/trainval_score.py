import csv, numpy as np, sys
def load(p): return {r['query_id']:np.array([float(r[c]) for c in ('p_0','p_1','p_2','p_5','p_20')]) for r in csv.DictReader(open(p))}
def pts(k):
    P,T,R=load(f'kits/{k}/pred.csv'),load(f'kits/{k}/_test_targets.csv'),load(f'kits/{k}/_test_reference.csv')
    C={r['query_id']:r['condition'] for r in csv.DictReader(open(f'kits/{k}/_test_conditions.csv'))}
    E={r['query_id']:r['episode_id'] for r in csv.DictReader(open(f'kits/{k}/_test_targets.csv'))}
    rat={}
    for c in sorted(set(C.values())):
        f=lambda X: np.mean([np.mean([((X[q]-T[q])**2).sum() for q in T if E[q]==e]) for e in {E[q] for q in T if C[q]==c}])
        rat[c]=f(P)/f(R)
    return 40/(1+np.mean(list(rat.values()))), rat
for h in 'AB':
    b,rb=pts('base_'+h); a,ra=pts('aug_'+h)
    print(f"held-out {'B' if h=='A' else 'A'}: base {b:.4f}  aug {a:.4f}  delta {a-b:+.4f} | "+' '.join(f"{c[:5]} {rb[c]:.4f}->{ra[c]:.4f}" for c in rb))
