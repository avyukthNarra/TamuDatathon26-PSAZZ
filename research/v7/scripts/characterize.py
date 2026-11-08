"""Decompose the train-selected size rule and check the paired gap against v6.

Run with: python characterize.py
Imports investigate.py. Reads public query/history/target/condition tables
(tick is an integer index; payout is 0, 1, 2, 5, or 20) and the v6 probability
file. Writes predictions/bma_ge2_124_noxor.csv and bma_ge2_124_prior.csv
with columns query_id, p_0, p_1, p_2, p_5, p_20.
"""

import csv
import math
from collections import defaultdict

import investigate as inv
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_queries, read_submission, read_targets
from slot_forecasting.scoring import distribution_error

MAIN = inv.MAIN
HERE = inv.HERE
MEMBERS = ["const0", "ge2@1", "ge2@2", "ge2@4", "ge2^1+2+4"]
NO_XOR = ["const0", "ge2@1", "ge2@2", "ge2@4"]


def bma(ys, feat_rows, qfeats, loge, prior=None):
    jn = len(qfeats)
    ls = [[0.0, 0.0] for _ in range(jn)]
    lp = [0.0] * jn if prior is None else [math.log(p) for p in prior]
    oll = 0.0
    for n, y in enumerate(ys):
        joints = []
        for j in range(jn):
            f = feat_rows[j][n]
            if f is None:
                mix = inv.logsumexp([math.log(0.5) + loge[0][y], math.log(0.5) + loge[1][y]])
                joints.append(lp[j] + mix)
            else:
                joints.append(
                    lp[j]
                    + inv.logsumexp([ls[j][s] + loge[s ^ f][y] for s in (0, 1)])
                    - inv.logsumexp(ls[j])
                )
        oll += inv.logsumexp(joints) - inv.logsumexp(lp)
        base = inv.logsumexp(joints)
        for j in range(jn):
            f = feat_rows[j][n]
            if f is not None:
                ls[j][0] += loge[0 ^ f][y]
                ls[j][1] += loge[1 ^ f][y]
            lp[j] = joints[j] - base
    m = max(lp)
    raw = [math.exp(x - m) for x in lp]
    z = sum(raw)
    feat_post = [x / z for x in raw]
    signs = []
    atom_terms = [[], []]
    for j, f in enumerate(qfeats):
        sm = max(ls[j])
        ps = [math.exp(ls[j][0] - sm), math.exp(ls[j][1] - sm)]
        sz = sum(ps)
        sign = 0 if ps[0] >= ps[1] else 1
        signs.append(sign)
        if f is None:
            for a in (0, 1):
                atom_terms[a].append(lp[j] + math.log(0.5))
        else:
            for s in (0, 1):
                atom_terms[s ^ f].append(lp[j] + math.log(ps[s] / sz))
    raw_w = [inv.logsumexp(atom_terms[a]) for a in (0, 1)]
    m = max(raw_w)
    w = [math.exp(raw_w[0] - m), math.exp(raw_w[1] - m)]
    z = sum(w)
    return oll, [w[0] / z, w[1] / z], feat_post, signs


def columns(pay, wins, cutoff, member_names, name_to_spec):
    cols = inv.feature_columns(pay, wins, cutoff, [name_to_spec[n] for n in member_names])
    return [c[0] for c in cols], [c[1] for c in cols]


def pack(queries, hist):
    rows = []
    for q in queries:
        pay, wins, cutoff, _ = inv.series_of(q, hist)
        rows.append((q, pay, wins, cutoff))
    return rows


def main():
    vq, vc, vh, vt = inv.load_split("validation")
    tq, _, th, _ = inv.load_split("train")
    atoms, true_atom, _ = inv.atoms_from_targets(vq, vt)
    loge = {k: [[math.log(max(x, 1e-15)) for x in v] for v in vecs] for k, vecs in atoms.items()}
    name_to_spec = {inv.spec_name(s): s for s in inv.build_specs()}
    vrows = pack(vq, vh)
    trows = pack(tq, th)

    def train_ll(member_names, prior=None):
        total = 0.0
        for q, pay, wins, cutoff in trows:
            ys = [y for _, y in wins]
            feats, qfeats = columns(pay, wins, cutoff, member_names, name_to_spec)
            total += bma(ys, feats, qfeats, loge[q["machine_type"]], prior)[0]
        return total

    print("train LL", flush=True)
    ll_full = train_ll(MEMBERS)
    ll_noxor = train_ll(NO_XOR)
    print(f"  with xor {ll_full:.1f}")
    print(f"  no xor   {ll_noxor:.1f}  (delta {ll_noxor - ll_full:+.1f})")
    prior_grid = []
    for p_static in (0.2, 0.35, 0.5, 0.65, 0.8):
        rest = (1.0 - p_static) / 3.0
        prior = [p_static, rest, rest, rest]
        ll = train_ll(NO_XOR, prior)
        prior_grid.append((ll, p_static, prior))
        print(f"  no-xor P(static)={p_static:.2f}  LL {ll:.1f}  delta vs xor-family {ll - ll_full:+.1f}")
    best_ll, best_p, best_prior = max(prior_grid)
    print(f"train-best of this grid: P(static)={best_p:.2f}")

    print("\nvalidation decomposition", flush=True)
    by_map = defaultdict(list)
    conf_bins = defaultdict(list)
    cond_rows = defaultdict(list)
    type_rows = defaultdict(list)
    mass = [0.0] * len(MEMBERS)
    sign_counts = [0, 0]
    hard_hit = hard_n = 0
    examples = []
    for q, pay, wins, cutoff in vrows:
        kind = q["machine_type"]
        ys = [y for _, y in wins]
        feats, qfeats = columns(pay, wins, cutoff, MEMBERS, name_to_spec)
        _, w, feat_post, signs = bma(ys, feats, qfeats, loge[kind])
        y = true_atom[q["query_id"]]
        rec = (w, y)
        cond_rows[vc[q["query_id"]]].append(rec)
        type_rows[kind].append(rec)
        cond_rows["all"].append(rec)
        for i, p in enumerate(feat_post):
            mass[i] += p
        j = max(range(len(MEMBERS)), key=lambda i: feat_post[i])
        by_map[MEMBERS[j]].append((feat_post[j], w, y, signs[j], qfeats[j], kind))
        conf = max(w)
        bname = "<0.6" if conf < 0.6 else "0.6-0.75" if conf < 0.75 else "0.75-0.9" if conf < 0.9 else ">=0.9"
        conf_bins[bname].append(rec)
        if j != 0 and qfeats[j] is not None:
            sign_counts[signs[j]] += 1
            if feat_post[j] >= 0.5:
                hard_n += 1
                hard_hit += int((signs[j] ^ qfeats[j]) == y)
        if len(examples) < 6 and feat_post[j] > 0.8 and j != 0 and qfeats[j] is not None:
            examples.append((q["query_id"], MEMBERS[j], signs[j], qfeats[j], signs[j] ^ qfeats[j], y, feat_post[j], w[y]))

    print(f"acc {inv.acc_of(cond_rows['all']):.3f}  P(true) {inv.mean_true(cond_rows['all']):.3f}")
    for cond in ("stable", "short_history", "drift", "shared_change"):
        print(f"  {cond:<16} acc {inv.acc_of(cond_rows[cond]):.3f}  P(true) {inv.mean_true(cond_rows[cond]):.3f}")
    for kind in ("steady", "balanced", "jackpot"):
        print(f"  {kind:<16} acc {inv.acc_of(type_rows[kind]):.3f}  P(true) {inv.mean_true(type_rows[kind]):.3f}")
    print("mean feature mass", {m: round(mass[i] / len(vrows), 3) for i, m in enumerate(MEMBERS)})
    print(f"{'MAP':<16}{'n':>5}{'meanP':>8}{'acc':>8}{'P(true)':>9}")
    for name in MEMBERS:
        bucket = by_map[name]
        rr = [(w, y) for _, w, y, _, _, _ in bucket]
        mean_p = sum(p for p, *_ in bucket) / len(bucket)
        print(f"{name:<16}{len(bucket):5d}{mean_p:8.3f}{inv.acc_of(rr):8.3f}{inv.mean_true(rr):9.3f}")
    print(f"MAP sign among non-static (0: atom=ge2 bit, 1: atom=1-bit): {sign_counts}")
    print(f"hard sign^bit when MAP lag mass>=0.5: {hard_hit}/{hard_n} = {hard_hit / hard_n:.3f}")
    print("calibration")
    for bname in ("<0.6", "0.6-0.75", "0.75-0.9", ">=0.9"):
        rr = conf_bins[bname]
        print(f"  {bname:<10} n {len(rr):4d} acc {inv.acc_of(rr):.3f} P(true) {inv.mean_true(rr):.3f} "
              f"mean conf {sum(max(w) for w, _ in rr) / len(rr):.3f}")
    print("examples (id, map, sign, bit, predicted atom, true atom, map mass, P(true))")
    for row in examples:
        print(" ", row)

    def val_weights(member_names, prior=None):
        out = {}
        for q, pay, wins, cutoff in vrows:
            kind = q["machine_type"]
            ys = [y for _, y in wins]
            feats, qfeats = columns(pay, wins, cutoff, member_names, name_to_spec)
            _, w, _, _ = bma(ys, feats, qfeats, loge[kind], prior)
            out[q["query_id"]] = (w, atoms[kind])
        return out

    base_rows = list(csv.DictReader(open(inv.V6)))
    print("\nscores", flush=True)
    for name, mp in (
        ("bma_ge2_124_noxor", val_weights(NO_XOR)),
        ("bma_ge2_124_prior", val_weights(NO_XOR, best_prior)),
    ):
        path = inv.write_preds(f"{name}.csv", base_rows, mp)
        pt, cond = inv.fmt_score(inv.score_file(path))
        print(f"  {name:<24} {pt:8.3f}  {cond}")

    queries = read_queries(MAIN / "public/validation/queries.csv")
    qmap = {q.query_id: q for q in queries}
    targets = {t.query_id: t for t in read_targets(MAIN / "public/validation/targets.csv")}
    groups = {r["query_id"]: r["condition"] for r in csv.DictReader(open(MAIN / "public/validation/conditions.csv"))}
    ref = read_submission(MAIN / "public/validation/reference.csv", qmap)

    def read_pred(path):
        return read_submission(path, qmap)

    pred = read_pred(HERE / "predictions" / "bma_ge2_124+static.csv")
    v6p = read_pred(inv.V6)
    ep_ids = defaultdict(list)
    for q in vq:
        ep_ids[q["episode_id"]].append(q["query_id"])
    diffs = defaultdict(list)
    for ep, ids in ep_ids.items():
        e_v6 = sum(distribution_error(v6p[k], targets[k].probabilities) for k in ids) / len(ids)
        e_new = sum(distribution_error(pred[k], targets[k].probabilities) for k in ids) / len(ids)
        diffs[vc[ids[0]]].append(e_v6 - e_new)
    all_d = [d for xs in diffs.values() for d in xs]
    mean = sum(all_d) / len(all_d)
    se = math.sqrt(sum((d - mean) ** 2 for d in all_d) / (len(all_d) - 1) / len(all_d))
    print(f"\npaired episode SE, v6 - bma: mean {mean:.5f} se {se:.5f} "
          f"better {sum(1 for d in all_d if d > 0)}/{len(all_d)}")
    print(f"95% interval [{mean - 1.96 * se:.5f}, {mean + 1.96 * se:.5f}]")
    for cond in ("stable", "short_history", "drift", "shared_change"):
        xs = diffs[cond]
        print(f"  {cond:<16} {sum(xs)/len(xs):.5f}  better {sum(1 for d in xs if d > 0)}/{len(xs)}")

    eps_by = defaultdict(list)
    for q in vq:
        if q["episode_id"] not in eps_by[vc[q["query_id"]]]:
            eps_by[vc[q["query_id"]]].append(q["episode_id"])
    halves = {"even": set(), "odd": set()}
    for eps in eps_by.values():
        for i, ep in enumerate(sorted(eps)):
            halves["even" if i % 2 == 0 else "odd"].add(ep)
    q_ep = {q["query_id"]: q["episode_id"] for q in vq}

    def half_points(path, episodes):
        sub = {k: v for k, v in read_pred(path).items() if q_ep[k] in episodes}
        ts = tuple(t for t in targets.values() if t.query_id in sub)
        ref_h = {k: v for k, v in ref.items() if k in sub}
        grp = {k: groups[k] for k in sub}
        return contest_score(sub, ts, grp, ref_h, max_points=60)["technical_points"]

    for label, episodes in halves.items():
        a = half_points(inv.V6, episodes)
        b = half_points(HERE / "predictions" / "bma_ge2_124+static.csv", episodes)
        print(f"  {label}: v6 {a:.3f}  bma {b:.3f}  delta {b - a:+.3f}")


if __name__ == "__main__":
    main()
