"""Fixed-settings replacement check, dev only. No further hyperparameter search."""
import json
import numpy as np
from experiments import HERE,OUT,TYPES,win_head
from harness import load_split,read_pred_csv,evaluate,diff_ci,to_csv


def main():
    train=load_split('train');v=load_split('validation')
    structure=json.loads((OUT/'train_structure.json').read_text())
    prior={k:np.asarray(x) for k,x in structure['type_pair_prior'].items()}
    cells={k:np.asarray(x) for k,x in structure['type_cell_mean'].items()}
    beta=structure['train_beta_levels']
    pop=json.loads((OUT/'v4_results.json').read_text())['inherited_population_prior']
    ref=read_pred_csv(HERE/'preds/TM_COMBINED4_validation.csv',v)
    rule=read_pred_csv(HERE/'preds/TM_RULE_validation.csv',v)
    raw=np.load(OUT/'v3_raw.npy');sizes=raw[:,1:]/raw[:,1:].sum(1,keepdims=True)
    def full(win):
        p=np.maximum(np.column_stack((1-win,win[:,None]*sizes)),1e-4)
        p/=p.sum(1,keepdims=True)
        return .2*rule+.8*p
    # Full replacement: TRAIN per-type means, same .45/.45/.1 weights,
    # strength 6, weak middle strength 2, and v4's learning rate 8.
    train_pw=win_head(v,prior,cells,beta,beta_strength=6.,beta_types=TYPES,beta_eta=8.,eta_short=8.,eta_long=8.)
    # Literal replacement using the original M4 jackpot low/high means for ALL types.
    jackpot_beta={k:beta['jackpot'] for k in TYPES}
    jackpot_all=win_head(v,prior,cells,jackpot_beta,beta_strength=6.,beta_types=TYPES,beta_eta=8.,eta_short=8.,eta_long=8.)
    # Partial replacement diagnostic: only jackpot cells use the train prior.
    baseline=win_head(v,prior,cells,beta,population_prior=pop,eta_short=8.,eta_long=8.)
    swap=baseline.copy();swap[v.mtype=='jackpot']=train_pw[v.mtype=='jackpot']
    candidates={'IMP_M4V4ALL':full(train_pw),'IMP_M4V4JACKPOT':full(jackpot_all),'IMP_M4V4SWAP':full(swap)}
    result={}
    for mid,p in candidates.items():
        to_csv(HERE/f'preds/{mid}_validation.csv',v,p)
        result[mid]={'evaluation':evaluate(p,v,'dev'),'difference':diff_ci(p,ref,v,'dev')}
    original=read_pred_csv(HERE/'preds/IMP_M4_validation.csv',v)
    result['original_M4_v3_jackpot_only']={'evaluation':evaluate(original,v,'dev'),'difference':diff_ci(original,ref,v,'dev')}
    output={'reference':'TM_COMBINED4','train_type_beta_levels':beta,'settings':'Fixed strength6, eta8, weights .45/.45/.1, middle strength2; no new tuning',
            'remaining_target_dependency':'Original validation-derived SIZE MODES retained; ALL and JACKPOT have NO validation-derived win prior. SWAP retains validation-derived win priors for non-jackpot machines.',
            'experiments':result}
    (OUT/'train_prior_replacement.json').write_text(json.dumps(output,indent=2)+'\n')
    for mid,r in result.items():print(mid,r['evaluation']['points'],r['difference'])

if __name__=='__main__':main()
