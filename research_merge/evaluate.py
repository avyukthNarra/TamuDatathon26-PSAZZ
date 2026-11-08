"""Cross-fit smooth-reversal merge against exact, sysid, and v13 forecasts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'mainFiles'))
from research_exact.model import ExactModel, fit, size_predict
from research_merge.model import merged_predict
from research_exact.evaluate import Evaluator
from slot_forecasting.io import read_submission, write_submission

V13 = ROOT.parent/'_tm/v13/mainFiles/runs/v13/crossfit_validation_predictions.csv'
LEVEL_SD = {'steady': (.50,.50), 'balanced': (.59,.57), 'jackpot': (.67,.71)}

def combine(win, sizes):
    return {i: np.r_[1-p, p*np.asarray(sizes[i])] for i,p in win.items()}

def main():
    out = ROOT/'runs/merge'; out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(ROOT/'mainFiles/public')
    parpath = out/'params.json'
    if parpath.exists(): params=json.loads(parpath.read_text())
    else:
        params = fit(ROOT/'mainFiles/public', size=25)
        parpath.write_text(json.dumps(params, indent=2))
    candidates = {m: {} for m in ('merged','static','independent','shared','exact','robust_eps003')}
    diagnostics = {}
    cachepath=out/'win_predictions.json'
    cached=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    for fold in (0,1):
        p = json.loads(json.dumps(params))
        for kind in ('steady','balanced','jackpot'):
            # Retain train-fitted cell means and shared type slope. Replace the
            # excess-history covariance with the measured independent residuals.
            q = p['levels'][kind]
            sd0, sd1 = LEVEL_SD[kind]
            q['cov'] = [[sd0**2, 0.], [0., sd1**2]]
        model = ExactModel(p, 25)
        fold_machines = [m for m in ev.machines if ev.folds[m.episode_id] == fold]
        if str(fold) in cached:
            wins=cached[str(fold)]['wins']; diagnostics[str(fold)]=cached[str(fold)]['diagnostics']
            if 'exact' not in wins:
                wins['exact'], _ = model.predict_win(fold_machines, verbose=False)
                cached[str(fold)]['wins']=wins; cachepath.write_text(json.dumps(cached))
            if 'wins_eps003' not in cached[str(fold)]:
                wr, dr = merged_predict(model, fold_machines, mode='all', hypothesis_eps=.003)
                cached[str(fold)]['wins_eps003']=wr; cached[str(fold)]['diagnostics_eps003']=dr
                cachepath.write_text(json.dumps(cached))
            sizes, _ = size_predict(ev.machines, ev.atoms(fold), params['size_prior'], 'all')
            for mode in ('merged','static','independent','shared','exact'):
                candidates[mode].update(combine(wins[mode], sizes))
            candidates['robust_eps003'].update(combine(cached[str(fold)]['wins_eps003']['merged'], sizes))
            continue
        wins, d = merged_predict(model, fold_machines, mode='all')
        robust_wins, robust_diagnostics = merged_predict(model, fold_machines, mode='all', hypothesis_eps=.003)
        exact_win, _ = model.predict_win(fold_machines, verbose=False)
        wins['exact'] = exact_win
        diagnostics[str(fold)] = d
        cached[str(fold)]={'wins':wins,'diagnostics':d,'wins_eps003':robust_wins,
                           'diagnostics_eps003':robust_diagnostics}
        cachepath.write_text(json.dumps(cached))
        sizes, _ = size_predict(ev.machines, ev.atoms(fold), params['size_prior'], 'all')
        for mode in ('merged','static','independent','shared','exact'):
            candidates[mode].update(combine(wins[mode], sizes))
        candidates['robust_eps003'].update(combine(robust_wins['merged'], sizes))
    # Tune conditional-size source on the opposite fold as the exact protocol does.
    for mode, pred in candidates.items():
        cross = {}
        for f in (0,1):
            trainfold = 1-f
            # Reconstruct size candidates with the fixed merged win forecast.
            win = {i: 1-np.asarray(pred[i])[0] for i in pred}
            sizes_by_family = {fam:size_predict(ev.machines, ev.atoms(trainfold), params['size_prior'], fam)[0]
                               for fam in ('all','static','lag')}
            fulls = {fam:combine(win,sizes) for fam,sizes in sizes_by_family.items()}
            bad=[(i,np.asarray(fulls['all'][i]).tolist()) for i in ev.ids if not np.all(np.isfinite(fulls['all'][i])) or np.min(fulls['all'][i]) < 0 or np.max(fulls['all'][i]) > 1]
            if bad: raise ValueError(f'invalid prediction before scoring: {bad[:3]}')
            chosen = min(fulls, key=lambda name: ev.score(fulls[name], trainfold)['R'])
            for i in ev.ids:
                if ev.folds[ev.eid[i]] == f: cross[i] = fulls[chosen][i]
        candidates[mode] = cross
    # Select the independent-reversal contribution on the opposite fold.
    exact_independent = {}
    selected_mix = {}
    for f in (0,1):
        trainfold=1-f
        options={}
        for weight in (0.,.05,.1,.2,.4,.6,1.):
            full={}
            for i in ev.ids:
                a=np.asarray(candidates['exact'][i]); b=np.asarray(candidates['independent'][i])
                wa=1-a[0]; wb=1-b[0]
                wl=1/(1+np.exp(-((1-weight)*np.log(wa/(1-wa))+weight*np.log(wb/(1-wb)))))
                sa=a[1:]/wa; sb=b[1:]/wb
                s=(1-weight)*sa+weight*sb
                full[i]=np.r_[1-wl,wl*s]
            options[weight]=full
        weight=min(options,key=lambda x:ev.score(options[x],trainfold)['R'])
        selected_mix[str(f)]=weight
        for i in ev.ids:
            if ev.folds[ev.eid[i]]==f: exact_independent[i]=options[weight][i]
    merged = candidates['merged']
    write_submission(out/'validation_crossfit.csv', merged)
    v13 = read_submission(V13, ev.ids)
    def logit_blend(a,b):
        pa=1-np.asarray(a)[0]; pb=1-np.asarray(b)[0]
        wa=1/(1+np.exp(-((np.log(pa/(1-pa))+np.log(pb/(1-pb)))/2)))
        sa=np.asarray(a)[1:]/pa; sb=np.asarray(b)[1:]/pb
        s=(sa+sb)/2
        return np.r_[1-wa,wa*s]
    blend={i:logit_blend(merged[i],v13[i]) for i in merged}
    write_submission(out/'validation_crossfit_v13_blend.csv', blend)
    # Diagnostic attribution: apply sysid's train-estimated lag-3 peer coefficient
    # as a one-step log-odds correction to the exact cross-fit forecast.
    q_by_id={q.query_id:q for q in ev.queries}; m_by_id={m.query_id:m for m in ev.machines}
    by_episode={}
    for q in ev.queries: by_episode.setdefault(q.episode_id,[]).append(q)
    peer_exact={}
    for i,pred in candidates['exact'].items():
        q=q_by_id[i]; scores=[]
        for other in by_episode[q.episode_id]:
            if other.machine_id==q.machine_id: continue
            seq=m_by_id[other.query_id].seq; tick=q.cutoff-3
            if 0<=tick<len(seq) and seq[tick]>=0: scores.append(2*float(seq[tick]>0)-1)
        peer=float(np.mean(scores)) if scores else 0.
        a=np.asarray(pred); win=1-a[0]
        corrected=1/(1+np.exp(-(np.log(win/(1-win))+.48635*peer)))
        peer_exact[i]=np.r_[1-corrected,corrected*a[1:]/win]
    results={k:ev.score(v) for k,v in candidates.items()}
    results['selected_exact_plus_independent_mix']=ev.score(exact_independent)
    results['selected_independent_weight_by_fold']=selected_mix
    results['merged_ci95']=ev.ci(merged)
    results['robust_eps003']=ev.score(candidates['robust_eps003'])
    results['v13']=ev.score(v13)
    results['merged_v13_equal_logodds_blend']=ev.score(blend)
    results['exact_plus_train_peer_lag3_diagnostic']=ev.score(peer_exact)
    # Per-type win-probability MSE, by condition and overall.
    tm={t.query_id:np.asarray(t.probabilities) for t in ev.targets}
    bytype={}
    for kind in ('steady','balanced','jackpot'):
        bytype[kind]={}
        for con in ('stable','short_history','drift','shared_change'):
            ids=[i for i in ev.ids if ev.kind[i]==kind and ev.conditions[i]==con]
            bytype[kind][con]=float(np.mean([(1-merged[i][0]-(1-tm[i][0]))**2 for i in ids]))
        ids=[i for i in ev.ids if ev.kind[i]==kind]
        bytype[kind]['all']=float(np.mean([(1-merged[i][0]-(1-tm[i][0]))**2 for i in ids]))
    results['win_mse_by_type_condition']=bytype
    def win_mse(pred, ids):
        return float(np.mean([(1-pred[i][0]-(1-tm[i][0]))**2 for i in ids]))
    jackpot_ids=[i for i in ev.ids if ev.kind[i]=='jackpot']
    results['robust_win_mse']={
        'overall_all_types':win_mse(candidates['robust_eps003'],ev.ids),
        'baseline_overall_all_types':win_mse(merged,ev.ids),
        'jackpot_all':win_mse(candidates['robust_eps003'],jackpot_ids),
        'jackpot_by_condition':{c:win_mse(candidates['robust_eps003'],[i for i in jackpot_ids if ev.conditions[i]==c])
                                for c in ('stable','short_history','drift','shared_change')},
        'baseline_jackpot_all':win_mse(merged,jackpot_ids),
        'baseline_jackpot_by_condition':{c:win_mse(merged,[i for i in jackpot_ids if ev.conditions[i]==c])
                                          for c in ('stable','short_history','drift','shared_change')}}
    ptype={}
    for kind in ('steady','balanced','jackpot'):
        ptype[kind]={}
        for con in ('stable','short_history','drift','shared_change'):
            ids=[i for i in ev.ids if ev.kind[i]==kind and ev.conditions[i]==con]
            ptype[kind][con]=float(np.mean([(1-peer_exact[i][0]-(1-tm[i][0]))**2 for i in ids]))
    results['peer_diagnostic_win_mse_by_type_condition']=ptype
    true_size={}
    for i,q in tm.items():
        win=1-q[0]
        true_size[i]=q[1:]/win if win>1e-9 else np.full(4,.25)
    merged_true_sizes=combine({i:1-np.asarray(v)[0] for i,v in merged.items()},true_size)
    v13_true_sizes=combine({i:1-np.asarray(v13[i])[0] for i in ev.ids},true_size)
    results['merged_true_size_diagnostic']=ev.score(merged_true_sizes)
    results['v13_true_size_diagnostic']=ev.score(v13_true_sizes)
    results['level_prior']={'means_and_slopes':'train-fitted per-type dead/live means with shared per-type signal slope',
                           'independent_sd_dead_live':LEVEL_SD,
                           'drift_cutoff_swap':float(np.clip(params['schedule']['anchor']/params['schedule']['duration'],0,1))}
    results['diagnostics']=diagnostics
    (out/'results.json').write_text(json.dumps(results,indent=2))
    write_submission(out/'validation_crossfit_robust_eps003.csv', candidates['robust_eps003'])
    print(json.dumps({k:v for k,v in results.items() if k!='diagnostics'},indent=2))

if __name__=='__main__': main()
