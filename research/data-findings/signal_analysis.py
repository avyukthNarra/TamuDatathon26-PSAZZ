import sys, numpy as np, pandas as pd
BASE = "/Users/avynarra/Documents/Datathon/worktrees/r0-baseline/mainFiles/public"
PAIRS = [(1,3),(2,5),(4,7)]

def load(split):
    q = pd.read_csv(f"{BASE}/{split}/queries.csv")
    h = pd.read_csv(f"{BASE}/{split}/history.csv")
    return q, h

def machine_stats(q, h):
    rows = []
    hg = {k: g for k, g in h.groupby(["episode_id","machine_id"])}
    for _, r in q.iterrows():
        g = hg.get((r.episode_id, r.machine_id))
        if g is None: continue
        g = g[g.tick < r.cutoff].sort_values("tick")
        tick = g.tick.values
        pay = g.payout.values
        w = dict(zip(tick, (pay > 0).astype(int)))
        pm = dict(zip(tick, pay))
        # per pair: collect (target tick t, agree?, win at t) with t-a, t-b present
        res = {}
        for (a,b) in PAIRS:
            ag_w, ag_n, dg_w, dg_n = 0,0,0,0
            ag_w1, ag_n1, dg_w1, dg_n1 = 0,0,0,0  # first half
            ag_w2, ag_n2, dg_w2, dg_n2 = 0,0,0,0
            ts = [t for t in tick if (t-a) in w and (t-b) in w]
            if not ts: res[(a,b)] = None; continue
            mid = r.cutoff / 2
            for t in ts:
                agree = w[t-a] == w[t-b]
                y = w[t]
                if agree: ag_w += y; ag_n += 1
                else: dg_w += y; dg_n += 1
                half = 1 if t < mid else 2
                if half == 1:
                    if agree: ag_w1 += y; ag_n1 += 1
                    else: dg_w1 += y; dg_n1 += 1
                else:
                    if agree: ag_w2 += y; ag_n2 += 1
                    else: dg_w2 += y; dg_n2 += 1
            res[(a,b)] = dict(n=len(ts), ag_n=ag_n, dg_n=dg_n,
                ag=ag_w/ag_n if ag_n else np.nan, dg=dg_w/dg_n if dg_n else np.nan,
                ag1=ag_w1/ag_n1 if ag_n1 else np.nan, dg1=dg_w1/dg_n1 if dg_n1 else np.nan,
                ag2=ag_w2/ag_n2 if ag_n2 else np.nan, dg2=dg_w2/dg_n2 if dg_n2 else np.nan)
        # strongest pair: largest |diff| among pairs with both groups >=5
        best, bestdiff, bestidx = None, -1, None
        for i,p in enumerate(PAIRS):
            s = res.get(p)
            if s is None or s["ag_n"] < 5 or s["dg_n"] < 5: continue
            d = s["ag"] - s["dg"]
            if abs(d) > bestdiff: bestdiff, best, bestidx = abs(d), d, i+1
        wins = pay[pay>0]
        rows.append(dict(
            episode_id=r.episode_id, machine_id=r.machine_id, machine_type=r.machine_type,
            signal=int(r.signal), n_ticks=len(tick),
            win_rate=(pay>0).mean() if len(pay) else np.nan,
            direction=(np.sign(best) if best is not None else np.nan),
            strongest_pair=bestidx if bestidx else np.nan,
            strength=bestdiff if bestidx else np.nan,
            mean_win_payout=wins.mean() if len(wins) else np.nan,
            share20=(wins==20).mean() if len(wins) else np.nan,
            # first vs second half for strongest pair
            d_first=(res[PAIRS[bestidx-1]]["ag1"]-res[PAIRS[bestidx-1]]["dg1"]) if bestidx else np.nan,
            d_second=(res[PAIRS[bestidx-1]]["ag2"]-res[PAIRS[bestidx-1]]["dg2"]) if bestidx else np.nan,
            wr_first=(pd.Series(pay[tick < r.cutoff/2]>0).mean() if (tick < r.cutoff/2).any() else np.nan),
            wr_second=(pd.Series(pay[tick >= r.cutoff/2]>0).mean() if (tick >= r.cutoff/2).any() else np.nan),
        ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    q, h = load("train")
    tr = machine_stats(q, h)
    tr.to_csv("/Users/avynarra/Documents/Datathon/worktrees/_coord/signal/train_machine_stats.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("train machines:", len(tr), "signal counts:\n", tr.signal.value_counts().to_string())
    tr["pair_str"] = tr.strongest_pair.map(lambda x: "" if pd.isna(x) else str(int(x)))
    g = tr.groupby(["machine_type","signal"])
    out = pd.DataFrame({
        "count": g.size(),
        "mean_win": g.win_rate.mean(),
        "share_dir_plus": g.direction.apply(lambda s: (s[s.notna()]>0).mean()),
        "n_dir": g.direction.apply(lambda s: s.notna().sum()),
        "mean_strength": g.strength.apply(lambda s: s.abs().mean()),
        "mean_win_payout": g.mean_win_payout.mean(),
        "share_20": g.share20.mean(),
    })
    print(out.round(3).to_string())
    print("\nstrongest pair distribution by type x signal (share of machines):")
    ct = pd.crosstab([tr.machine_type, tr.signal], tr.pair_str, normalize="index").round(2)
    print(ct.to_string())
    # drift: first vs second half
    print("\nDRIFT (train): mean win first half, second half, and strongest-pair diff first/second, sign flips")
    tr["flip"] = np.sign(tr.d_first) != np.sign(tr.d_second)
    tr["dd"] = tr.d_second - tr.d_first
    tr["wr_change"] = tr.wr_second - tr.wr_first
    d = tr.groupby(["machine_type","signal"]).agg(
        n=("win_rate","size"),
        wr_first=("wr_first","mean"), wr_second=("wr_second","mean"),
        wr_change=("wr_change","mean"),
        dir_flip_share=("flip", "mean"),
        mean_dd=("dd","mean"),
    )
    print(d.round(3).to_string())
    # per-signal only
    print("\nBy signal only:")
    print(tr.groupby("signal").agg(n=("win_rate","size"), mean_win=("win_rate","mean"),
        share_dir_plus=("direction", lambda s: (s[s.notna()]>0).mean()),
        mean_strength=("strength", lambda s: s.abs().mean()),
        flip=("flip","mean"), wr_change=("wr_change","mean")).round(3).to_string())

    # validation
    qv, hv = load("validation")
    va = machine_stats(qv, hv)
    tg = pd.read_csv(f"{BASE}/validation/targets.csv")
    va = va.merge(tg[["query_id","machine_id"] if False else ["episode_id","p_0"]].drop_duplicates(), how="left", on="episode_id") if False else va
    tq = pd.read_csv(f"{BASE}/validation/targets.csv")[["query_id","p_0"]]
    qv2 = qv.merge(tq, on="query_id")
    qv2["true_win"] = 1 - qv2.p_0
    print("\nVALIDATION true win prob (1-p_0) by type x signal:")
    vt = qv2.groupby(["machine_type","signal"]).agg(n=("true_win","size"), mean_true_win=("true_win","mean"), sd=("true_win","std"))
    print(vt.round(3).to_string())
    print("by signal only:", qv2.groupby("signal").true_win.agg(["size","mean"]).round(3).to_dict())
    va.to_csv("/Users/avynarra/Documents/Datathon/worktrees/_coord/signal/val_machine_stats.csv", index=False)
