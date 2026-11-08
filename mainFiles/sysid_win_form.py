"""Fold-cross-fitted target diagnostics for the functional form of P(win)."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sysid_study import ROOT,load_machines,folds,NAMES
from sysid_bayes import TYPES
from sysid_dynamic import dynamic_predict
from slot_forecasting.io import read_targets,read_submission,write_json
from slot_forecasting.contest import contest_score
from sysid_eval import bootstrap


def design(ms,hist_lp,model):
    q=[m.query for m in ms]; n=len(q)
    typ=np.column_stack([[x.machine_type==t for x in q] for t in TYPES]).astype(float)
    sig=np.array([x.signal for x in q])[:,None]
    base=np.column_stack([typ[:,:2],sig])
    names=['type_steady','type_balanced','signal']
    lp=logit(np.clip(np.asarray(hist_lp),1e-7,1-1e-7))[:,None]
    f=np.array([m.features[m.query.cutoff] for m in ms])
    parity=f[:,model['ruleids']]
    ep_ids=[x.episode_id for x in q]
    ep_mean=np.array([np.mean([hist_lp[j] for j in range(n) if ep_ids[j]==ep_ids[i]]) for i in range(n)])[:,None]
    peer_mean=np.array([np.mean([hist_lp[j] for j in range(n) if ep_ids[j]==ep_ids[i] and j!=i]) for i in range(n)])[:,None]
    peer_lag=f[:,74:75]
    candidates={
      'type_signal':(base,names),
      'type_signal_history_lp':(np.column_stack([base,lp]),names+['history_logit']),
      'type_signal_history_lp_by_type':(np.column_stack([base,lp,typ*lp]),names+['history_logit']+[f'{t}_history' for t in TYPES]),
      'type_signal_history_lp_by_type_signal':(np.column_stack([base,lp,typ*lp,sig*lp]),names+['history_logit']+[f'{t}_history' for t in TYPES]+['signal_history']),
      'type_signal_history_episode':(np.column_stack([base,lp,ep_mean]),names+['history_logit','episode_mean_history_logit']),
      'type_signal_history_peer':(np.column_stack([base,lp,peer_mean]),names+['history_logit','peer_mean_history_logit']),
      'type_signal_history_peer_lag':(np.column_stack([base,lp,peer_lag]),names+['history_logit','mean_peer_win_lag3']),
      'type_signal_parity':(np.column_stack([base,parity]),names+[NAMES[i] for i in model['ruleids']]),
      'type_signal_history_and_parity':(np.column_stack([base,lp,parity]),names+['history_logit']+[NAMES[i] for i in model['ruleids']]),
    }
    for half in [16,32,64,128]:
        stats=[]
        for m in ms:
            F=m.features
            ticks=m.ticks
            recent=ticks[(ticks<m.query.cutoff)&(ticks>=m.query.cutoff-half)]
            if len(recent):
                pay=m.payouts[recent]; wins=np.sum(pay>0)
                stats.append((wins+.5)/(len(pay)+1))
            else:stats.append(.5)
        frac=logit(np.clip(stats,1e-5,1-1e-5))[:,None]
        candidates[f'type_signal_recent{half}']=(np.column_stack([base,frac]),names+[f'recent_win_logit_{half}'])
        candidates[f'type_signal_history_recent{half}']=(np.column_stack([base,lp,frac]),names+['history_logit',f'recent_win_logit_{half}'])
    return candidates


def fit_predict(X,y,Xtest,ridge=0.):
    mu=X.mean(axis=0);sd=X.std(axis=0);sd[sd<1e-9]=1
    A=(X-mu)/sd;B=(Xtest-mu)/sd
    # Preserve three type indicators while ridge-shrinking all other effects.
    penalty=np.eye(A.shape[1])*ridge
    penalty[:2,:2]=0
    pen=np.sqrt(np.diag(penalty))
    augmented=np.vstack([A,np.diag(pen)])
    beta=np.linalg.lstsq(augmented,np.r_[y,np.zeros(A.shape[1])],rcond=1e-10)[0]
    return B@beta+np.mean(y)-np.mean(A@beta),beta


def metrics(z,y):
    p=expit(z);return {'logit_rmse':float(np.sqrt(np.mean((z-logit(y))**2))),'probability_rmse':float(np.sqrt(np.mean((p-y)**2)))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'public');ap.add_argument('--models',type=Path,default=ROOT/'runs/sysid-final');ap.add_argument('--out',type=Path,default=ROOT/'runs/sysid-win');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    # Freeze model family before reading validation true targets.
    spec=json.loads((args.models/'train_choice.json').read_text())['candidate'];wn,rn,sn=spec.split('/')
    from sysid_pipeline import WIN_FILES,REGIMES
    model=json.loads((args.models/WIN_FILES[wn]).read_text());regime=next(x for x in REGIMES if x[0]==rn);_,width,shared=regime
    machines,queries,_=load_machines(args.data,'validation');n=len(machines);cond,fold=folds(args.data,queries);fold=np.asarray(fold)
    win,_=dynamic_predict(model,machines,width=width,shared_width=shared)
    lp=np.array([win[m.query.query_id] for m in machines]);byid={m.query.query_id:i for i,m in enumerate(machines)}
    Xs=design(machines,lp,model); targets={t.query_id:t.probabilities for t in read_targets(args.data/'validation/targets.csv')}
    yt=np.array([1-targets[m.query.query_id][0] for m in machines]);yt=np.clip(yt,1e-8,1-1e-8);z=logit(yt)
    reference=read_submission(args.data/'validation/reference.csv');rows=[]
    # All candidates use identical predictor inputs; folds tune ridge within A/B.
    all_scores={}; probability_metrics={}; fold_prediction_store={}; fold_points={}
    sizes={qid:np.array(targets[qid])[1:]/(1-np.array(targets[qid])[0]) for qid in targets}
    references=read_submission(args.data/'validation/reference.csv')
    target_rows={t.query_id:t for t in read_targets(args.data/'validation/targets.csv')}
    for name,(X,xnames) in Xs.items():
      for ridge in [.1,1.,10.]:
        tag=f'{name}|ridge={ridge:g}';crosspred={};foldstats={}
        for tune in [0,1]:
          train=fold==tune;test=~train;zpred,_=fit_predict(X[train],z[train],X[test],ridge)
          held=np.where(test)[0];crosspred.update({machines[i].query.query_id:float(expit(v)) for i,v in zip(held,zpred)})
          foldstats['A' if tune==1 else 'B']=metrics(zpred,yt[test])
        fold_prediction_store[tag]=crosspred
        p={qid:np.r_[1-w,w*sizes[qid]] for qid,w in crosspred.items()}
        all_scores[tag]=contest_score(p,list(target_rows.values()),cond,references,max_points=60)
        fold_points[tag]={}
        for held in [0,1]:
          keep={machines[i].query.query_id for i in range(n) if fold[i]==held}
          fold_points[tag]['A' if held==0 else 'B']=contest_score({k:v for k,v in p.items() if k in keep},[v for k,v in target_rows.items() if k in keep],{k:v for k,v in cond.items() if k in keep},{k:v for k,v in references.items() if k in keep},max_points=60)['technical_points']
        probability_metrics[tag]=foldstats
    order=sorted(all_scores,key=lambda k:all_scores[k]['technical_points'],reverse=True)
    primary='type_signal_history_lp|ridge=1'
    primary_ci=bootstrap({qid:np.r_[1-w,w*sizes[qid]] for qid,w in fold_prediction_store[primary].items()},queries,list(target_rows.values()),cond,references,repeats=4000)
    best_tag=order[0];best_pred=fold_prediction_store[best_tag]
    best_full={k:np.r_[1-v,v*sizes[k]] for k,v in best_pred.items()}
    best_ci=bootstrap(best_full,queries,list(target_rows.values()),cond,references,repeats=4000)
    group_rmse={}
    for tag,pred in fold_prediction_store.items():
      group_rmse[tag]={}
      for group,values in [('type',{t:[i for i,m in enumerate(machines) if m.query.machine_type==t] for t in TYPES}),('condition',{c:[i for i,m in enumerate(machines) if cond[m.query.query_id]==c] for c in sorted(set(cond.values()))})]:
        group_rmse[tag][group]={name:float(np.sqrt(np.mean([(pred[machines[i].query.query_id]-yt[i])**2 for i in idx]))) for name,idx in values.items()}
    # Verify the discrete winning-size profiles independently of win targets.
    profile_report={}
    for typ in TYPES:
      rows=[targets[m.query.query_id] for m in machines if m.query.machine_type==typ]
      profiles=np.array([np.asarray(r[1:])/(1-r[0]) for r in rows])
      rounded,counts=np.unique(np.round(profiles,8),axis=0,return_counts=True)
      profile_report[typ]={'n_modes_rounded_1e-8':int(len(rounded)),'counts':counts.tolist(),'profiles':rounded.tolist()}
    report={'train_model':spec,'best_candidate':best_tag,'best_candidate_ci95_points':best_ci,'validation_size_profile_check':profile_report,'diagnostic':'Each fixed candidate is fitted on targets in A and scored on B, then fitted on B and scored on A. The win-only ceiling substitutes exact held-out conditional size probabilities; it is diagnostic, not a deployable forecast. No model is selected using the fold it predicts.',
      'candidate_win_only_scores':all_scores,'candidate_fold_points':fold_points,'candidate_probability_metrics':probability_metrics,
      'primary_candidate':primary,'primary_candidate_ci95_points':primary_ci,
      'crossfit_win_rmse_by_group':group_rmse,'crossfit_ranking':order,
      'target_win_deciles_all':np.quantile(yt,np.arange(.1,1,.1)).tolist(),
      'target_deciles_by_type':{t:np.quantile(yt[[m.query.machine_type==t for m in machines]],[.05,.1,.25,.5,.75,.9,.95]).tolist() for t in TYPES}}
    write_json(args.out/'win_form.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
