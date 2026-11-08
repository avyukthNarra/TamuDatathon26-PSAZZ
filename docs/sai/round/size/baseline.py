"""Fit pinned 27d776d + original 20% rule model; write only assigned files."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'mainFiles'),str(ROOT/'mainFiles/combo'),str(ROOT/'docs/sai')]
from slot_forecasting.io import load_inputs,read_labels,read_targets
import mixer5_source as pm
import predict as combo
from harness import load_split,to_csv,evaluate


def main():
    start=time.time();data=ROOT/'mainFiles/public'
    tq,th=load_inputs(data/'train');labels=read_labels(data/'train/labels.csv',tq)
    vq,vh=load_inputs(data/'validation');targets=read_targets(data/'validation/targets.csv')
    modes=pm.size_modes_from_targets(targets,vq)
    prior=pm.win_prior_by_type_from_targets(targets,vq)
    model=pm.adapt(tq,th,labels,size_modes=modes,win_prior=prior)
    rules=combo.fit_rules(tq,th,labels)
    print('FIT_DONE',time.time()-start,flush=True)
    mixer=model.predict(vq,vh)
    rp=[r.predict(vq,vh) for r in rules]
    v=load_split('validation')
    mix=np.array([mixer[q] for q in v.qid])
    rule=np.mean([np.array([r[q] for q in v.qid]) for r in rp],axis=0)
    p=.2*rule+.8*mix;p/=p.sum(1,keepdims=True)
    to_csv(ROOT/'docs/sai/preds/R5_BASE_validation.csv',v,p)
    np.savez_compressed(OUT/'baseline_arrays.npz',mixer=mix,rule=rule,base=p,qid=v.qid)
    report={'id':'R5_BASE','source':'27d776d','source_sha256':hashlib.sha256((ROOT/'worktrees/_tm/mixer5/mainFiles/slot_forecasting/pair_mixer.py').read_bytes()).hexdigest(),
            'size_modes':modes,'win_prior':prior,'pairs':model.pairs,'cell_mode':model.cell_mode,
            'eta':model.eta,'size_keys':model.size_keys,'evaluation':evaluate(p,v,'dev'),
            'seconds':time.time()-start,'target_dependency':'Size modes and per-type win priors fitted on all validation targets, per requested baseline; all structure/lag selection fitted on train. No audit/all scoring.'}
    (OUT/'baseline.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BASELINE',report['evaluation'],'SECONDS',report['seconds'],flush=True)

if __name__=='__main__':main()
