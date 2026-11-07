"""Verify reusable v4 adapter, saved predictions, and determinism; no scoring."""
import json
import hashlib
import numpy as np
from experiments import HERE,ROOT,OUT
from improved_mixer_v4 import ResearchPairMixerV4
from slot_forecasting.io import load_inputs,read_submission


def main():
    metadata=json.loads((HERE/'combos/TM_MIXER4.json').read_text())
    results=json.loads((OUT/'v4_results.json').read_text())
    best=results['experiments']['IMP_M5V4']
    tq,th=load_inputs(ROOT/'mainFiles/public/train')
    vq,vh=load_inputs(ROOT/'mainFiles/public/validation')
    model=ResearchPairMixerV4(size_modes=metadata['size_modes'],win_prior=metadata['win_prior'],
        win_options=best['params']).fit(tq,th)
    a,b=model.predict(vq,vh),model.predict(vq,vh)
    rule=read_submission(HERE/'preds/TM_RULE_validation.csv')
    target=read_submission(HERE/'preds/IMP_M5V4_validation.csv')
    rw=best['rule_weight']
    error=max(abs(rw*rule[q][j]+(1-rw)*a[q][j]-target[q][j]) for q in a for j in range(5))
    deterministic=max(abs(a[q][j]-b[q][j]) for q in a for j in range(5))
    assert error<1e-12,error
    assert deterministic==0.,deterministic
    files={}
    for mid in results['experiments']:
        path=HERE/f'preds/{mid}_validation.csv'
        pred=read_submission(path,{q.query_id for q in vq})
        assert len(pred)==len(vq)
        files[path.name]={'rows':len(pred),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    output={'reusable_model_reproduction_max_abs_diff':error,'two_run_max_abs_diff':deterministic,
            'n_queries':len(vq),'files':files,'target_dependency':results['inherited_validation_target_dependency']}
    (OUT/'v4_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
