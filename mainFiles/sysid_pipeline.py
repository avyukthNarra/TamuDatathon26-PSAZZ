"""Reproduce the frozen system-identification experiment and forecasts.

python3 mainFiles/sysid_pipeline.py --out mainFiles/runs/sysid-final
This trains on train only, saves an answer-free candidate catalog, cross-fits
candidate selection, bootstraps episodes, and predicts test with both the
train-only configuration and the configuration selected on all validation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from sysid_study import ROOT,load_machines,folds
from sysid_bayes import train_win,condition_map,complete_data,static_predict
from sysid_dynamic import dynamic_predict
from sysid_size import train_size,size_predict,size_data
from sysid_eval import probability_distribution,official_score,bootstrap,select_cross_fitted
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import read_labels,read_targets,read_submission,write_json,write_submission

WIN_MODELS=(('no_common',False),('basic',True),('peers_only','peers_only'),('all_peers','all_peers'),('strongest_peer','strongest_peer'))
WIN_FILES={'no_common':'win_model_no_common.json','basic':'win_model.json','peers_only':'win_model_peers_only.json','all_peers':'win_model_all_peers.json','strongest_peer':'win_model_strongest_peer.json'}
SIZE_MODELS=(('full',False,False),('screened',True,False),('constrained',True,True))
SIZE_FILES={'full':'size_model.json','screened':'size_model_screened.json','constrained':'size_model_screened_constrained.json'}
REGIMES=(('dyn0',0.,0.),('dyn01',.1,.1),('dyn02',.2,.2),('dyn01shared0',.1,0.))


def load_json(path):
    return json.loads(Path(path).read_text())


def prepare(data,out):
    start=time.time();out.mkdir(parents=True,exist_ok=True)
    train,tq,th=load_machines(data,'train');labels=read_labels(data/'train/labels.csv',tq);conditions=condition_map(data,'train')
    models={};sizes={}
    for name,shared in WIN_MODELS:
        models[name]=train_win(train,labels,conditions,out,shared_features=shared)
        print('fit win',name,models[name]['stable_train_log_marginal'],flush=True)
    for name,screened,constrained in SIZE_MODELS:
        sizes[name]=train_size(train,labels,conditions,out,screened=screened,constrained=constrained)
        print('fit size',name,sizes[name]['marginal_loglik'],flush=True)
    # BIC is a train-only preference over shared physical nuisance effects.
    nwin=sum(len(complete_data(m,labels)[1]) for m in train if conditions[m.query.query_id]=='stable')
    win_bic={name:-2*model['stable_train_log_marginal']+(6+len(model['commonids'])+len(model['ruleids'])-1)*np.log(nwin) for name,model in models.items()}
    selected_win=min(win_bic,key=win_bic.get)
    nsize=sum(len(size_data(m,labels)[1]) for m in train if conditions[m.query.query_id]=='stable')
    size_bic={name:-2*model['marginal_loglik']+((10 if name=='constrained' else 18)+len(model['all_rule_weights'])-1)*np.log(nsize) for name,model in sizes.items()}
    selected_size=min(size_bic,key=size_bic.get)
    # Choose reversal width from TRAIN history marginal likelihood, without
    # using a validation target or supplying test condition labels.
    train_likelihood={};episode_condition={m.query.episode_id:conditions[m.query.query_id] for m in train}
    for name,width,shared_width in REGIMES:
        _,r=dynamic_predict(models[selected_win],train,width=width,shared_width=shared_width)
        train_likelihood[name]={'total':sum(v['log_evidence'] for v in r.values()),
            'by_condition':{c:sum(v['log_evidence'] for ep,v in r.items() if episode_condition[ep]==c) for c in sorted(set(conditions.values()))},
            'static_total':sum(v['component_log_evidences'][0] for v in r.values())}
        print('train regime',name,train_likelihood[name]['total'],flush=True)
    selected_regime=max(train_likelihood,key=lambda k:train_likelihood[k]['total'])
    selected=f'{selected_win}/{selected_regime}/{selected_size}'
    train_choice={'candidate':selected,'win_bic':win_bic,'size_bic':size_bic,'regime_train_likelihood':train_likelihood,
        'stable_train_win_observations':nwin,'stable_train_positive_observations':nsize}
    write_json(out/'train_choice.json',train_choice)
    # Freeze the complete selection family before reading any target file.
    catalog={}
    for wn,_ in WIN_MODELS:
        for rn,w,sw in REGIMES:
            for sn,_,_ in SIZE_MODELS:catalog[f'{wn}/{rn}/{sn}']={'win':wn,'regime':rn,'width':w,'shared_width':sw,'size':sn}
    for rn,half in [('static',None),('decay16',16),('decay32',32)]:
        for sn,_,_ in SIZE_MODELS:catalog[f'basic/{rn}/{sn}']={'win':'basic','regime':rn,'half_life':half,'size':sn}
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('sysid_*.py')}
    write_json(out/'frozen_catalog.json',{'catalog':catalog,'train_only_choice':selected,'source_hashes':source_hashes,
        'declaration':'No validation targets or labels were loaded by prepare(). All candidates are fixed before scoring.'})
    val,vq,vh=load_machines(data,'validation');ids=[q.query_id for q in vq]
    szpred={name:{m.query.query_id:size_predict(model,m)[0] for m in val} for name,model in sizes.items()}
    wpred={}
    for wn,model in models.items():
        for rn,width,shared_width in REGIMES:
            pp,_=dynamic_predict(model,val,width=width,shared_width=shared_width)
            wpred[wn,rn]=pp
            print('predict',wn,rn,flush=True)
    for rn,half in [('static',None),('decay16',16),('decay32',32)]:wpred['basic',rn]={m.query.query_id:static_predict(models['basic'],m,half)[0] for m in val}
    predictions=[];names=[]
    for name,spec in catalog.items():
        win=wpred[spec['win'],spec['regime']];sz=szpred[spec['size']]
        predictions.append([probability_distribution(win[k],sz[k]) for k in ids]);names.append(name)
    base=FrequencyForecaster('pooled').fit(tq,th,labels);pooled=base.predict(vq,vh)
    predictions.append([pooled[k] for k in ids]);names.append('pooled')
    np.savez_compressed(out/'validation_candidates.npz',predictions=np.asarray(predictions),names=np.asarray(names),query_ids=np.asarray(ids))
    report={'runtime_seconds':time.time()-start,'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy']},'n_candidates':len(catalog),'train_only_choice':selected}
    write_json(out/'prepare.json',report);print('PREPARED',report,flush=True)


def score(data,out,make_test=True):
    start=time.time();archive=np.load(out/'validation_candidates.npz',allow_pickle=False)
    names=archive['names'].tolist();ids=archive['query_ids'].tolist()
    candidates={name:{k:p.tolist() for k,p in zip(ids,pred)} for name,pred in zip(names,archive['predictions'])}
    _,vq,_=load_machines(data,'validation');targets=read_targets(data/'validation/targets.csv');groups,fold=folds(data,vq)
    reference=read_submission(data/'validation/reference.csv');foldids=[{q.query_id for q,f in zip(vq,fold) if f==i} for i in range(2)]
    scores={}
    for name,pred in candidates.items():
        scores[name]={'overall':official_score(pred,vq,targets,groups,reference),
            'fold_A':official_score(pred,vq,targets,groups,reference,foldids[0]),
            'fold_B':official_score(pred,vq,targets,groups,reference,foldids[1])}
    eligible=[n for n in names if n!='pooled']
    cf,selection=select_cross_fitted({n:candidates[n] for n in eligible},vq,fold,{n:[scores[n]['fold_A']['technical_points'],scores[n]['fold_B']['technical_points']] for n in eligible})
    selectA=selection['tune_A_predict_B'];selectB=selection['tune_B_predict_A']
    train_choice=load_json(out/'train_choice.json')['candidate'];best=max(eligible,key=lambda n:scores[n]['overall']['technical_points'])
    result={'selection':{'tune_A_predict_B':selectA,'tune_B_predict_A':selectB,'train_only':train_choice,'select_all_validation_for_test':best},
        'catalog_scores':scores,'results':{}}
    final_predictions={'cross_fitted':cf,'train_only':candidates[train_choice],'pooled':candidates['pooled']}
    truth={t.query_id:np.asarray(t.probabilities) for t in targets}
    for tag,pred in list(final_predictions.items()):
        if tag=='pooled':continue
        pw={};ps={}
        for k,v in pred.items():
            p=np.asarray(v);w=1-p[0];size=p[1:]/w;t=truth[k];tw=1-t[0];ts=t[1:]/tw
            pw[k]=probability_distribution(tw,size).tolist();ps[k]=probability_distribution(w,ts).tolist()
        final_predictions[tag+'_oracle_win']=pw;final_predictions[tag+'_oracle_size']=ps
    for name,pred in final_predictions.items():
        r=official_score(pred,vq,targets,groups,reference);r['ci95_points']=bootstrap(pred,vq,targets,groups,reference)
        r['fold_A_points']=official_score(pred,vq,targets,groups,reference,foldids[0])['technical_points']
        r['fold_B_points']=official_score(pred,vq,targets,groups,reference,foldids[1])['technical_points']
        result['results'][name]=r
    write_submission(out/'validation_predictions.csv',cf);write_submission(out/'validation_train_only.csv',candidates[train_choice])
    write_submission(out/'validation_selected_all.csv',candidates[best])
    write_json(out/'final_results.json',result)
    for k,r in result['results'].items():print(k,r['technical_points'],r['ci95_points'],{c:v['loss_ratio'] for c,v in r['by_condition'].items()},flush=True)
    print('SELECTION',result['selection'],flush=True)
    if make_test:
        test,tq,_=load_machines(data,'test');catalog=load_json(out/'frozen_catalog.json')['catalog']
        for name,file in [(train_choice,'submission_train_only.csv'),(best,'submission.csv')]:
            spec=catalog[name];wm=load_json(out/WIN_FILES[spec['win']]);sm=load_json(out/SIZE_FILES[spec['size']])
            if spec['regime'].startswith('dyn'):
                win,_=dynamic_predict(wm,test,width=spec['width'],shared_width=spec['shared_width'])
            else:win={m.query.query_id:static_predict(wm,m,spec['half_life'])[0] for m in test}
            size={m.query.query_id:size_predict(sm,m)[0] for m in test}
            pred={k:probability_distribution(w,size[k]) for k,w in win.items()};write_submission(out/file,pred)
    write_json(out/'score_runtime.json',{'runtime_seconds':time.time()-start})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'public');p.add_argument('--out',type=Path,default=ROOT/'runs/sysid-final');p.add_argument('--phase',choices=['prepare','score','all'],default='all');args=p.parse_args()
    if args.phase in ('prepare','all'):prepare(args.data,args.out)
    if args.phase in ('score','all'):score(args.data,args.out)


if __name__=='__main__':main()
