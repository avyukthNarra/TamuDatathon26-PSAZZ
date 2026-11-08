"""Cross-fit the measured dead/live logit-level prior for P(win)."""
import argparse,json
import copy
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sysid_study import ROOT,load_machines,folds
from sysid_bayes import TYPES
from sysid_dynamic import dynamic_predict
from sysid_pipeline import WIN_FILES,REGIMES
from sysid_eval import probability_distribution
from slot_forecasting.io import read_targets,read_submission,write_json
from slot_forecasting.contest import contest_score
from sysid_eval import bootstrap
from sysid_win_form import fit_predict


def fit_level_prior(indices,machines,truth):
    result={}
    for typ in TYPES:
        ix=[i for i in indices if machines[i].query.machine_type==typ]
        z=logit(np.clip([truth[machines[i].query.query_id] for i in ix],1e-8,1-1e-8))
        sig=np.array([machines[i].query.signal for i in ix])
        centers=np.quantile(z,[.25,.75])
        for _ in range(100):
            cls=np.argmin(abs(z[:,None]-centers[None,:]),axis=1)
            new=np.array([z[cls==j].mean() for j in range(2)])
            if np.max(abs(new-centers))<1e-12:break
            centers=new
        levels=[]
        for mode in [0,1]:
            y=z[cls==mode];s=sig[cls==mode]
            X=np.column_stack([np.ones(len(s)),s]);b=np.linalg.lstsq(X,y,rcond=None)[0]
            sd=float(np.sqrt(np.mean((y-X@b)**2)))
            levels.append({'intercept':float(b[0]),'signal':float(b[1]),'sd':sd,'n':int(len(y)),'center':float(centers[mode])})
        result[typ]={'dead':levels[0],'live':levels[1]}
    return result


def normal_logistic_mean(mu,sd,n=25):
    x,w=np.polynomial.hermite.hermgauss(n)
    return float(np.sum(w*expit(mu+np.sqrt(2)*sd*x))/np.sqrt(np.pi))


def remap(p,type_,signal,prior):
    d=prior[type_]['dead'];l=prior[type_]['live']
    qd=normal_logistic_mean(d['intercept']+d['signal']*signal,d['sd'])
    ql=normal_logistic_mean(l['intercept']+l['signal']*signal,l['sd'])
    r=np.clip((p-qd)/max(ql-qd,1e-8),0,1)
    return float((1-r)*qd+r*ql),float(r)


def physical_model(base,prior):
    """Translate the fold's two logit-level components into the simulator grid."""
    model=copy.deepcopy(base)
    mus=[];betas=[];sigmas=[];amps=[]
    for typ in TYPES:
        d=prior[typ]['dead'];l=prior[typ]['live']
        mus.append((d['intercept']+l['intercept'])/2)
        betas.append((d['signal']+l['signal'])/2)
        amps.append((l['intercept']-d['intercept'])/2)
        # Level residual is machine-specific. Pool dead/live variances as a
        # first-order Gaussian approximation for the persistent intercept.
        sigmas.append(np.sqrt((d['sd']**2+l['sd']**2)/2))
    model['mu']=mus;model['signal_by_type']=betas
    model['amplitude_by_type']=amps;model['sigma_by_type']=sigmas
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'public');ap.add_argument('--models',type=Path,default=ROOT/'runs/sysid-final');ap.add_argument('--out',type=Path,default=ROOT/'runs/sysid-win');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((args.models/'train_choice.json').read_text())['candidate'];wn,rn,_=spec.split('/');model=json.loads((args.models/WIN_FILES[wn]).read_text());_,width,shared=next(x for x in REGIMES if x[0]==rn)
    machines,queries,_=load_machines(args.data,'validation');cond,fold=folds(args.data,queries);fold=np.asarray(fold)
    base,_=dynamic_predict(model,machines,width=width,shared_width=shared);p=np.array([base[m.query.query_id] for m in machines])
    targets=read_targets(args.data/'validation/targets.csv');truth={t.query_id:1-t.probabilities[0] for t in targets};ref=read_submission(args.data/'validation/reference.csv')
    pred=np.empty(len(machines));modes=np.empty(len(machines));calibrated=np.empty(len(machines));priors={};physical_priors={}
    for tune in [0,1]:
        train=np.where(fold==tune)[0];test=np.where(fold!=tune)[0];prior=fit_level_prior(train,machines,truth);priors['A' if tune==0 else 'B']=prior
        tuned_model=physical_model(model,prior)
        tuned,_=dynamic_predict(tuned_model,machines,width=width,shared_width=shared)
        tuned_all=np.array([tuned[m.query.query_id] for m in machines])
        pred[test]=tuned_all[test]
        physical_priors['A' if tune==0 else 'B']={t:{'mu':tuned_model['mu'][TYPES.index(t)],'signal':tuned_model['signal_by_type'][TYPES.index(t)],'sigma':tuned_model['sigma_by_type'][TYPES.index(t)],'amplitude':tuned_model['amplitude_by_type'][TYPES.index(t)]} for t in TYPES}
        mapped=np.array([remap(p[i],machines[i].query.machine_type,machines[i].query.signal,prior)[0] for i in range(len(machines))])
        X=np.column_stack([np.array([m.query.machine_type=='steady' for m in machines],float),np.array([m.query.machine_type=='balanced' for m in machines],float),np.array([m.query.signal for m in machines]),logit(np.clip(mapped,1e-6,1-1e-6))])
        zfit=logit(np.clip([truth[machines[i].query.query_id] for i in train],1e-8,1-1e-8))
        zp,_=fit_predict(X[train],zfit,X[test],1.)
        calibrated[test]=expit(zp)
    qids=[m.query.query_id for m in machines];sizes={t.query_id:np.asarray(t.probabilities[1:])/(1-t.probabilities[0]) for t in targets}
    forecast={qid:probability_distribution(pred[i],sizes[qid]) for i,qid in enumerate(qids)}
    groups={qid:cond[qid] for qid in qids};score=contest_score(forecast,targets,groups,ref,max_points=60)
    fold_score={}
    for f in [0,1]:
        keep=set(qids[i] for i in range(len(qids)) if fold[i]==f)
        fold_score['A' if f==0 else 'B']=contest_score({k:v for k,v in forecast.items() if k in keep},[t for t in targets if t.query_id in keep],{k:v for k,v in groups.items() if k in keep},{k:v for k,v in ref.items() if k in keep},max_points=60)['technical_points']
    ci=bootstrap(forecast,queries,targets,groups,ref,repeats=4000)
    mapped_pred=np.array([remap(p[i],machines[i].query.machine_type,machines[i].query.signal,priors['A' if fold[i]!=0 else 'B'])[0] for i in range(len(machines))])
    mapped_forecast={qid:probability_distribution(mapped_pred[i],sizes[qid]) for i,qid in enumerate(qids)}
    mapped_score=contest_score(mapped_forecast,targets,groups,ref,max_points=60)
    mapped_ci=bootstrap(mapped_forecast,queries,targets,groups,ref,repeats=4000)
    cforecast={qid:probability_distribution(calibrated[i],sizes[qid]) for i,qid in enumerate(qids)}
    cscore=contest_score(cforecast,targets,groups,ref,max_points=60);cci=bootstrap(cforecast,queries,targets,groups,ref,repeats=4000)
    report={'candidate':'History posterior mode under opposite-fold fitted type x signal dead/live logit level priors','points':score['technical_points'],'ci95_points':ci,'fold_points':fold_score,'condition_ratios':{k:v['loss_ratio'] for k,v in score['by_condition'].items()},'direct_remap_points':mapped_score['technical_points'],'direct_remap_ci95_points':mapped_ci,'direct_remap_condition_ratios':{k:v['loss_ratio'] for k,v in mapped_score['by_condition'].items()},'calibrated_points':cscore['technical_points'],'calibrated_ci95_points':cci,'calibrated_condition_ratios':{k:v['loss_ratio'] for k,v in cscore['by_condition'].items()},'calibrated_win_rmse_by_type':{t:float(np.sqrt(np.mean([(calibrated[i]-truth[qids[i]])**2 for i,m in enumerate(machines) if m.query.machine_type==t]))) for t in TYPES},'win_rmse_by_type':{t:float(np.sqrt(np.mean([(pred[i]-truth[qids[i]])**2 for i,m in enumerate(machines) if m.query.machine_type==t]))) for t in TYPES},'win_rmse_by_condition':{c:float(np.sqrt(np.mean([(pred[i]-truth[qids[i]])**2 for i,m in enumerate(machines) if cond[qids[i]]==c]))) for c in sorted(set(cond.values()))},'priors':priors,'physical_priors':physical_priors,'base_history_score':None,'diagnostic':'For each tune fold, logit targets are clustered into dead/live components by type, then each component is regressed on signal. The resulting midpoint, half-gap and pooled residual SD replace type intercept, signal, parity amplitude and Gaussian machine intercept SD in the history likelihood; only the other fold is scored. Exact held-out sizes are used only for the win-only ceiling.'}
    write_json(args.out/'level_prior.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
