"""Check reusable IMP_M5 model against experiment CSV, twice, without scoring."""
import hashlib
import json
import sys
from pathlib import Path

from improved_mixer import ImprovedPairMixer
from experiments import ROOT,HERE
from slot_forecasting.io import load_inputs,read_targets,read_submission
from pranav_v3 import size_modes_from_targets


def main():
    tq,th = load_inputs(ROOT/'mainFiles/public/train')
    vq,vh = load_inputs(ROOT/'mainFiles/public/validation')
    # Inherited baseline size mode extraction ONLY; new win priors fit train.
    modes = size_modes_from_targets(read_targets(ROOT/'mainFiles/public/validation/targets.csv'),vq)
    model = ImprovedPairMixer(size_modes=modes).fit(tq,th)
    a,b = model.predict(vq,vh),model.predict(vq,vh)
    rule = read_submission(HERE/'preds/TM_RULE_validation.csv')
    ref = read_submission(HERE/'preds/IMP_M5_validation.csv')
    error = max(abs(.2*rule[q][j]+.8*a[q][j]-ref[q][j]) for q in a for j in range(5))
    deterministic = max(abs(a[q][j]-b[q][j]) for q in a for j in range(5))
    assert error < 1e-12, error
    assert deterministic == 0., deterministic
    report = {'n_queries':len(a),'IMP_M5_reproduction_max_abs_diff':error,
        'two_run_max_abs_diff':deterministic,'new_win_priors_read_validation_targets':False,
        'inherited_size_modes_read_validation_targets':True,'files':{}}
    for k in range(1,6):
        path = HERE/f'preds/IMP_M{k}_validation.csv'
        p = read_submission(path,{q.query_id for q in vq})
        assert len(p) == len(vq)
        report['files'][path.name] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(p)}
    (Path(__file__).resolve().parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
