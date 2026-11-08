"""Confirm deterministic train screening and coherent score/tally artifacts."""
import json
import numpy as np
from discover import OUT,main as discover_main


def main():
    old=json.loads((OUT/'train_discovery.json').read_text());old.pop('seconds',None)
    cache=np.load(OUT/'train_scores.npz');previous={k:cache[k].copy() for k in cache.files};cache.close()
    discover_main()
    new=json.loads((OUT/'train_discovery.json').read_text());new.pop('seconds',None)
    assert old==new,'Train discovery is not deterministic'
    now=np.load(OUT/'train_scores.npz')
    max_error=0.
    for k,a in previous.items():
        b=now[k]
        assert np.array_equal(a,b,equal_nan=True),k
        if np.issubdtype(a.dtype,np.floating):
            max_error=max(max_error,float(np.nanmax(abs(a-b))))
    count=0
    for kind,tally in new['tally'].items():
        assert sum(tally['bic'].values())==sum(tally['search_penalized'].values())
        count+=sum(tally['bic'].values())
    assert count==new['n_eligible_BIC_tally_machines']==730
    assert new['n_deterministic_rules']==368
    summary=json.loads((OUT/'summary.json').read_text())
    assert not summary['global_passed_train_gate']
    assert not any(summary['target_passed_train_gate'].values())
    result={'two_run_train_artifacts_identical':True,'score_array_max_abs_diff':max_error,
            'eligible_machines':count,'deterministic_rules':368,'families':len(new['families']),
            'dev_candidates_tuned':0,'train_gate_passed':False}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFIED',result)

if __name__=='__main__':main()
