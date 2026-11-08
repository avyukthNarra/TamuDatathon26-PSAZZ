"""Bounded T1 sparse-history experiments; dev-only evaluation and tuning."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT/'docs/sai'
sys.path.insert(0, str(HERE))
from harness import load_split, read_pred_csv, evaluate, diff_ci
from harness.scoring import scope_episodes
from sparse import forecast, gapped, PAIRS

OUT = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    v, train = load_split('validation'), load_split('train')
    # All new structure is the original train-discovered pairs plus the bounded
    # lag 1..8 dictionary. Inherited validation-derived priors are held fixed.
    prior = json.loads((HERE/'combos/TM_MIXER4.json').read_text())['win_prior']
    baseline = read_pred_csv(HERE/'preds/TM_MIXER4_validation.csv', v)
    rule = read_pred_csv(HERE/'preds/TM_RULE_validation.csv', v)
    ref = read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv', v)
    raw = np.load(HERE/'improve/model/v3_raw.npy')
    sizes = raw[:, 1:]/raw[:, 1:].sum(1, keepdims=True)
    control = forecast(v, baseline, sizes, prior)
    assert np.max(abs(control-baseline)) < 1e-10
    mask = gapped(v)
    assert np.array_equal(control[~mask], baseline[~mask])
    assert np.max(abs(.2*rule+.8*control-ref)) < 1e-10
    eps = scope_episodes(v, 'dev')
    folds = [[], []]
    for c in sorted(set(v.condition)):
        ce = [e for e in eps if v.condition[v.episode_rows(e)[0]] == c]
        ce.sort(key=lambda e: hashlib.sha256(f't1-v1|{e}'.encode()).hexdigest())
        for f in (0, 1):
            folds[f].extend(i for e in ce[f::2] for i in v.episode_rows(e))
    folds = [np.array(sorted(f)) for f in folds]
    original_lines = (HERE/'preds/TM_COMBINED4_validation.csv').read_text().splitlines(keepends=True)
    original_rows = {line.split(',',1)[0]:line for line in original_lines[1:]}
    def save(mid, p):
        lines = [original_lines[0]]
        for i, qid in enumerate(v.qid):
            if not mask[i] or np.array_equal(p[i], ref[i]):
                lines.append(original_rows[qid])
            else:
                lines.append(qid+','+','.join(f'{x:.17g}' for x in p[i])+'\n')
        (HERE/f'preds/{mid}_validation.csv').write_text(''.join(lines))
    def fs(p, f):
        ids = folds[f]
        return evaluate(p[ids], v.subset(ids), 'dev', bootstrap=0)['points']
    candidates = []
    settings = [('T1a', {'method':'expected', 'soft_weight':a}) for a in (.25, .5, 1.)]
    settings += [('T1b', {'method':'forward', 'soft_weight':a}) for a in (.25, .5, 1.)]
    settings += [('T1c', {'decay_missing':False, 'half_lives':h, 'eta':e})
                 for h in ((6.,20.),(12.,40.),(24.,80.)) for e in (4.,8.,16.)]
    settings += [('T1c', {'decay_missing':True, 'half_lives':h, 'eta':e})
                 for h,e in [((6.,20.),8.),((24.,80.),8.)]]
    settings += [('T1d', {'extra':extra, 'extra_mass':mass})
                 for extra in ('single','wide','both') for mass in (.1,.5)]
    family_indices = {}
    for number, (family, params) in enumerate(settings, 1):
        p = forecast(v, baseline, sizes, prior, **params)
        blend = .2*rule+.8*p
        # Assignment restores stored baseline bytes on dense rows. This is part
        # of the gate, rather than numerical roundoff hidden by a tolerance.
        assert np.array_equal(p[~mask], baseline[~mask])
        blend[~mask] = ref[~mask]
        assert blend[~mask].tobytes() == ref[~mask].tobytes()
        mid = f'BOOST_T1_{number:02d}'
        row = {'id':mid,'family':family,'params':params,
               'dev':evaluate(blend,v,'dev'),'difference':diff_ci(blend,ref,v,'dev'),
               'fold_scores':[fs(blend,f) for f in (0,1)],'dense_byte_identical':True}
        candidates.append((row,blend));family_indices.setdefault(family,[]).append(number-1)
        save(mid,blend)
        print(mid,family,params,row['dev']['points'],row['difference'],flush=True)
    reference = {'id':'TM_COMBINED4','family':'control','params':{},
                 'dev':evaluate(ref,v,'dev'),'difference':diff_ci(ref,ref,v,'dev'),
                 'fold_scores':[fs(ref,f) for f in (0,1)],'dense_byte_identical':True}
    candidates.append((reference,ref))
    summaries = {}
    def opposite_check(indices, tag):
        # Include the unchanged baseline so selection is allowed to decline a change.
        indices = indices + [len(candidates)-1]
        oof = ref.copy(); chosen = []
        for f in (0,1):
            index = max(indices,key=lambda i:candidates[i][0]['fold_scores'][1-f])
            row,p = candidates[index]
            oof[folds[f]] = p[folds[f]]
            chosen.append({'test_fold':f,'selected':row['id'],'params':row['params']})
        save(f'BOOST_T1_{tag}_OOF',oof)
        return {'dev':evaluate(oof,v,'dev'),'difference':diff_ci(oof,ref,v,'dev'),'choices':chosen}
    for family,indices in family_indices.items():
        best = max(indices,key=lambda i:candidates[i][0]['dev']['points'])
        summaries[family] = {'best':candidates[best][0]['id'],
                            'opposite_fold':opposite_check(indices,family)}
    best_index = max(range(len(candidates)),key=lambda i:candidates[i][0]['dev']['points'])
    selected, prediction = candidates[best_index]
    save('BOOST_T1_BEST',prediction)
    report = {'scope':'dev','reference':reference,
              'baseline_reproduction_max_abs_diff':float(abs(control-baseline).max()),
              'gapped_train_machines':int(gapped(train).sum()),'gapped_validation_machines':int(mask.sum()),
              'dense_byte_identity_all_variants':True,'inherited_prior':prior,
              'inherited_target_dependency':'V4 win prior and payout-size modes fit all 96 validation targets; unchanged. No audit/all evaluation.',
              'size_source':'Original v3/v4 shared raw size head: docs/sai/improve/model/v3_raw.npy',
              'lag_dictionary':{'base':PAIRS,'extra':'single lags 1..8, all 28 distinct pairs from 1..8'},
              'candidates':[r for r,_ in candidates],'families':summaries,'best':selected['id'],
              'global_opposite_fold':opposite_check(list(range(len(candidates)-1)),'GLOBAL'),
              'seconds':time.perf_counter()-started}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',report['best'],report['global_opposite_fold'],flush=True)


if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError(f'Set {key}=1')
    main()
