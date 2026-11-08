"""Describe observed-history gaps on train and DEV only, without target fitting."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
from harness import load_split
from harness.scoring import scope_episodes
from sparse import PAIRS


def main():
    result={}
    for name in ('train','validation'):
        s=load_split(name)
        rows=np.arange(s.n) if name=='train' else np.concatenate([
            s.episode_rows(e) for e in scope_episodes(s,'dev')])
        report={}
        for condition in sorted(set(s.condition)):
            details=[]
            for i in rows[s.condition[rows]==condition]:
                observed=np.flatnonzero(np.isfinite(s.grid[i,:s.cutoff[i]]))
                missing=~np.isfinite(s.grid[i,observed[0]:s.cutoff[i]])
                complete=[sum(t>=max(a,b) and np.isfinite(s.grid[i,t-a]) and
                              np.isfinite(s.grid[i,t-b]) for t in observed)/len(observed)
                          for a,b in PAIRS]
                details.append([len(observed),int(observed[-1]-observed[0]+1),
                                int(missing.sum()),int(np.sum(np.diff(
                                    np.r_[False,missing,False].astype(int))==1)),float(np.mean(complete))])
            data=np.array(details)
            report[condition]={'machines':len(data),'columns':['observed','span','missing_to_cutoff','gap_runs','pair_coverage'],
                               'median':np.median(data,axis=0).tolist(),'min':data.min(0).tolist(),'max':data.max(0).tolist()}
        result[name]=report
    Path(__file__).with_name('history_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
