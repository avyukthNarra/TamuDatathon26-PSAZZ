import volfeat as V, numpy as np
from scipy.stats import spearmanr
v,names,F,dev,rows,arows,w,fold=V.setup()
M=["TM_COMBINED","C2","TM_E7","TM_MIXER","TM_RULE","TM_E6"]
P={m:V.read_pred_csv(V.HERE/"preds"/f"{m}_validation.csv",v) for m in M}
E={m:((P[m]-v.truth)**2).sum(1) for m in M}
print("corr(err_member, feature) spearman on dev; ratio=err/TMC err in low/high tercile")
feats=["n_obs","obs_frac","ewv_w16","vy16","vol_of_vol","abs_gap_z","garch_h","since_shift","peer_mean_ewv_w16","peer_max_ewv_w16","agree_sd_gap_z","peer_gap_gap_z","peer_mean_vol_of_vol"]
for f in feats:
    x=F[rows,names.index(f)]
    q=np.quantile(x,[1/3,2/3]); b=np.digitize(x,q)
    line=f"{f:22s}"
    for m in M:
        c=spearmanr(x,E[m][rows])[0]; line+=f" {m[3:8]:>6s}:{c:+.2f}"
    print(line)
    for t in range(3):
        s=b==t; base=E["TM_COMBINED"][rows][s].mean()
        print(f"   tercile{t} errTMC={base:.4f} "+" ".join(f"{m[3:8]}={E[m][rows][s].mean()/base:.3f}" for m in M[1:]), "best=",min(M,key=lambda m:E[m][rows][s].mean()))
