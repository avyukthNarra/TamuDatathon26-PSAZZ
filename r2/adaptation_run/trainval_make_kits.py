"""Build paired kits: base_X (train=train) and aug_X (train=train+val half X); validation=half X; test=other half."""
import csv, json, shutil
from pathlib import Path
SRC=Path('/Users/avynarra/Documents/Datathon/R2_kit40/public'); OUT=Path('kits')
def rows(p):
    with open(p) as f: r=csv.DictReader(f); return r.fieldnames, list(r)
def write(p, fields, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(data)
_,vq=rows(SRC/'validation/queries.csv'); eps=sorted({r['episode_id'] for r in vq}); half={e:i%2 for i,e in enumerate(eps)}
def epof(r): return r.get('episode_id') or r['query_id'].rsplit('-m',1)[0]
qep={r['query_id']:r['episode_id'] for r in vq}
def sel(name, h):
    f,d=rows(SRC/'validation'/name)
    key=lambda r: half[r['episode_id']] if 'episode_id' in r else half[qep[r['query_id']]]
    return f,[r for r in d if key(r)==h]
for h in (0,1):
    o=1-h
    for kind in ('base','aug'):
        k=OUT/f'{kind}_{"AB"[h]}'
        if k.exists(): shutil.rmtree(k)
        (k).mkdir(parents=True); shutil.copy(SRC/'metadata.json',k/'metadata.json')
        for name in ('queries.csv','history.csv','labels.csv','conditions.csv','sample_submission.csv'):
            f,d=rows(SRC/'train'/name)
            if kind=='aug':
                f2,d2=sel(name,h); d=d+[{c:r.get(c,'') for c in f} for r in d2]
            write(k/'train'/name,f,d)
        for name in ('queries.csv','history.csv','labels.csv','conditions.csv','sample_submission.csv','targets.csv','reference.csv'):
            f,d=sel(name,h); write(k/'validation'/name,f,d)
        for name in ('queries.csv','history.csv','sample_submission.csv'):
            f,d=sel(name,o); write(k/'test'/name,f,d)
        f,d=sel('targets.csv',o); write(k/'_test_targets.csv',f,d)
        for name in ('conditions.csv','reference.csv'):
            f,d=sel(name,o); write(k/f'_test_{name}',f,d)
print('ok')
