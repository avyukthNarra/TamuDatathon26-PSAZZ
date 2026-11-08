"""Predeclared 18-cell grid; smoke every cell, then full only if estimated <600s.
No validation targets train model parameters. Selection and reports scope=dev only.
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'mainFiles'),str(HERE/'methods')]
from harness import load_split, smoke_split, read_pred_csv, evaluate, diff_ci, blend_gain, to_csv
from harness.results import write_result
from harness.scoring import scope_episodes
from slot_forecasting.schema import Query, Play
from _team_mixer_source import adapt, CLASS_INDEX, MIN_PROBABILITY
from TEAM_VOLATILITY import VolatilityMixer

SETS=[(12.,40.),(8.,24.,80.),(16.,64.,1e9)]
GRID=[]
for h in SETS:
    opts=[{}, {'score_half_life':40.}, {'alpha':.01}, {'decay_missing':False}]
    if h==SETS[0]:opts += [{'score_half_life':100.},{'alpha':.05}]
    for o in opts: GRID.append(dict(half_life=h,**o))
for sensitivity in (1.,2.):
    for alpha in (.01,.05):GRID.append(dict(half_life=SETS[0],adaptive=True,sensitivity=sensitivity,alpha=alpha))
assert len(GRID)==18

def build_train(t):
    qs=[Query(str(t.qid[i]),str(t.episode[i]),str(t.machine[i]),str(t.mtype[i]),int(t.signal[i]),int(t.cutoff[i])) for i in range(t.n)]
    hs=[Play(str(t.episode[i]),str(t.machine[i]),tick,pay) for i in range(t.n) for tick,pay in t.seq(i).items()]
    labels={str(t.qid[i]):int(t.outcome[i]) for i in range(t.n)}
    fitted=adapt(qs,hs,labels,log=lambda *_:None)
    return dict(pairs=fitted.pairs,cell_mode=fitted.cell_mode,type_size_priors=fitted.type_size_priors)

def predict(v,params,trained):
    model=VolatilityMixer(**trained,**params)
    ans=[]
    for i in range(v.n):
        own=[None]*(int(v.cutoff[i])+1)
        for tick,pay in v.seq(i).items():own[tick]=CLASS_INDEX[pay]
        captured=[]
        model.walk(own,str(v.mtype[i]),int(v.cutoff[i]),lambda t,p:captured.append(p))
        p=np.maximum(captured[-1],MIN_PROBABILITY);ans.append(p/p.sum())
    return np.asarray(ans)

def hybrid(raw,rule,combined):
    pw=.8*(1-raw[:,0])+.2*(1-rule[:,0])
    sizes=combined[:,1:]/combined[:,1:].sum(axis=1,keepdims=True)
    return np.column_stack([1-pw,pw[:,None]*sizes])

def main():
    started=time.monotonic(); smoke_only='--smoke' in sys.argv
    v=load_split('validation');t=load_split('train');sm=smoke_split(v)
    trained=build_train(t)
    paths=lambda name: HERE/'preds'/f'{name}_validation.csv'
    mix=read_pred_csv(paths('TM_MIXER'),v);comb=read_pred_csv(paths('TM_COMBINED'),v);rule=read_pred_csv(paths('TM_RULE'),v)
    rows=np.asarray([list(v.qid).index(q) for q in sm.qid])
    decl={'grid':GRID,'source':'6a8098c06837d23b9f6b904ed5b8a9e0b5904569','selection':'total dev technical points; no audit evaluation','hybrid':'0.8 new mixer win + 0.2 cached rule win; cached TM_COMBINED severity','n_variants':len(GRID)}
    (HERE/'grids'/'TEAM_VOLATILITY_declared.json').write_text(json.dumps(decl,indent=1)+'\n')
    smoke=[]; st=time.monotonic()
    for j,p in enumerate(GRID):
        r=predict(sm,p,trained);h=hybrid(r,rule[rows],comb[rows]);e=evaluate(h,sm,scope='smoke',bootstrap=0)
        smoke.append({'id':j,'params':p,'points':e['points']})
        print('smoke',j,round(e['points'],4),flush=True)
    elapsed=time.monotonic()-st;estimate=elapsed*v.n/sm.n
    smoke_doc={'smoke_rows':sm.n,'seconds':elapsed,'estimated_full_grid_seconds':estimate,'grid':smoke}
    (HERE/'grids'/'TEAM_VOLATILITY_smoke.json').write_text(json.dumps(smoke_doc,indent=1)+'\n')
    print('SMOKE seconds',elapsed,'FULL estimate',estimate,flush=True)
    if smoke_only or estimate>500:
        print('STOP full budget estimate exceeds bounded allowance' if estimate>500 else 'smoke complete',flush=True);return
    full=[];best=None;control=None
    for j,p in enumerate(GRID):
        before=time.monotonic();r=predict(v,p,trained);h=hybrid(r,rule,comb);e=evaluate(h,v,scope='dev',bootstrap=0)
        full.append({'id':j,'params':p,'points':e['points'],'condition_ratios':e['condition_ratios'],'seconds':time.monotonic()-before})
        if j==0:control=(r,h)
        elif best is None or e['points']>best[0]:best=(e['points'],j,r,h)
        print('full',j,round(e['points'],4),flush=True)
    # Actual fresh model/traversal refits, no prediction cache reuse.
    p=GRID[best[1]];again=predict(v,p,build_train(t));det=float(np.max(np.abs(again-best[2])))
    base_again=predict(v,GRID[0],build_train(t));base_det=float(np.max(np.abs(base_again-control[0])))
    result={'predeclared_variants':len(GRID),'baseline_controls':3,'exact_zero_change_controls':1,'winner_id':best[1], 'grid':full,'elapsed_seconds':time.monotonic()-started,
            'baseline_raw_maxdiff_cached_mixer':float(np.max(np.abs(control[0]-mix))), 'baseline_hybrid_maxdiff_cached_combined':float(np.max(np.abs(control[1]-comb))),
            'baseline_det_maxdiff':base_det,'selected_det_maxdiff':det,'source_commit':decl['source'],'no_validation_targets_fit':True}
    (HERE/'grids'/'TEAM_VOLATILITY_grid.json').write_text(json.dumps(result,indent=1)+'\n')
    for name,pred,params,d in [('TEAM_VOLATILITY',best[3],p,det),('TEAM_VOLATILITY_CONTROL',control[1],GRID[0],base_det)]:
        to_csv(paths(name),v,pred)
        extra={'vs_TM_COMBINED':diff_ci(pred,comb,v,scope='dev'),'vs_TM_MIXER':diff_ci(pred,mix,v,scope='dev'),'source_commit':decl['source'],'grid_variants':len(GRID),'zero_change_controls':1,'memory_family_controls':3}
        gain=blend_gain(pred,comb,v,scope='dev');gain.pop('oof');extra['blend_gain']=gain
        write_result(name,'codex','Causal score memory and variance-normalized innovation gate; cached combined severity',params,v,pred,result['elapsed_seconds'],determinism=d,scope='dev',extra=extra)
    report=['# TEAM_VOLATILITY (dev only)',f'- Predeclared {len(GRID)} variants; smoke {sm.n} rows, {elapsed:.1f}s; extrapolated grid {estimate:.1f}s.',f'- Best nonbaseline cell {best[1]}: {p}.',f'- Cached mixer raw control max difference {result["baseline_raw_maxdiff_cached_mixer"]:.3g}; combined hybrid {result["baseline_hybrid_maxdiff_cached_combined"]:.3g}.',f'- Fresh trained refit determinism: selected {det:.3g}, control {base_det:.3g}.','- Source pinned 6a8098c; no size modes or validation targets fitted.','- Own-machine residual EWMA16/64 subtracts Bernoulli noise and normalizes variance before score forgetting/share adaptation.','- Gate changes act on the next observed update; no peer or condition inputs.','- Selection used total dev only; audit never evaluated.','- Results include paired CIs, condition ratios, episode-OOF blend contribution; grid contains all controls.']
    (HERE/'reports'/'TEAM_VOLATILITY.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='grid'}),flush=True)
if __name__=='__main__':main()
