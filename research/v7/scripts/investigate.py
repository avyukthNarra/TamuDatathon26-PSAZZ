"""What sets the payout-size atom at the query tick?

Atoms are the two exact conditional mixes (p1, p2, p5, p20) / p_win per type,
taken from validation targets. Rules are scored without the query label:
train online log score selects them; validation labels only measure them.

Reads public CSVs and runs/combined-v6/validation_predictions.csv.
Writes predictions/*.csv (query_id, p_0, p_1, p_2, p_5, p_20) and results.json.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

MAIN = Path("/Users/avynarra/Documents/Datathon/worktrees/pranav-5629/mainFiles")
sys.path.insert(0, str(MAIN))
from slot_forecasting.contest import score_contest_files  # noqa: E402

HERE = Path("/Users/avynarra/Documents/Datathon/research_pf/g5-size")
PRED = HERE / "predictions"
PRED.mkdir(exist_ok=True)
V6 = MAIN / "runs/combined-v6/validation_predictions.csv"
PAY = ("1", "2", "5", "20")
IDX = {1: 0, 2: 1, 5: 2, 20: 3}
CLASS_ID = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
PAIRS = [(a, b) for a in range(1, 9) for b in range(a + 1, 9)]
CANON = ((1, 3), (2, 5), (4, 7))
BITS = {
    "win": lambda v: 1 if v > 0 else 0,
    "ge2": lambda v: 1 if v >= 2 else 0,
    "ge5": lambda v: 1 if v >= 5 else 0,
    "eq20": lambda v: 1 if v == 20 else 0,
}


def logsumexp(xs):
    m = max(xs)
    if m < -1e200:
        return -1e300
    return m + math.log(sum(math.exp(x - m) for x in xs))


def load_split(name):
    base = MAIN / "public" / name
    queries = list(csv.DictReader(open(base / "queries.csv")))
    cond = {}
    cond_path = base / "conditions.csv"
    if cond_path.exists():
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(cond_path))}
    hist = defaultdict(dict)
    for r in csv.DictReader(open(base / "history.csv")):
        hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
    targets = {}
    tpath = base / "targets.csv"
    if tpath.exists():
        targets = {r["query_id"]: r for r in csv.DictReader(open(tpath))}
    return queries, cond, hist, targets


def atoms_from_targets(queries, targets):
    q = {r["query_id"]: r for r in queries}
    found = defaultdict(list)
    assign = {}
    worst = 0.0
    for k, r in targets.items():
        win = 1.0 - float(r["p_0"])
        if win <= 0:
            continue
        vec = tuple(float(r["p_" + p]) / win for p in PAY)
        kind = q[k]["machine_type"]
        hit = None
        for i, atom in enumerate(found[kind]):
            dev = max(abs(a - b) for a, b in zip(vec, atom))
            if dev < 1e-8:
                hit = i
                worst = max(worst, dev)
                break
        if hit is None:
            found[kind].append(vec)
            hit = len(found[kind]) - 1
        assign[k] = (kind, hit)
    canon, relabel = {}, {}
    for kind, vecs in found.items():
        rank = sorted(range(len(vecs)), key=lambda i: -vecs[i][0])
        canon[kind] = [vecs[i] for i in rank]
        remap = {old: new for new, old in enumerate(rank)}
        for k, (knd, hit) in list(assign.items()):
            if knd == kind:
                relabel[k] = remap[hit]
    return canon, relabel, worst


def series_of(q, hist):
    cutoff = int(q["cutoff"])
    raw = hist[(q["episode_id"], q["machine_id"])]
    pay = [-1] * cutoff
    gaps = 0
    for t in range(cutoff):
        v = raw.get(t, -1)
        pay[t] = v
        gaps += int(v < 0)
    wins = [(t, IDX[pay[t]]) for t in range(cutoff) if pay[t] > 0]
    return pay, wins, cutoff, gaps


def bit_at(pay, t, lag, bit):
    j = t - lag
    if j < 0 or pay[j] < 0:
        return None
    return bit(pay[j])


def mix_probs(p0, weights, atoms, floor=1e-4):
    pw = 1.0 - p0
    size = [weights[0] * atoms[0][i] + weights[1] * atoms[1][i] for i in range(4)]
    probs = [1.0 - pw] + [pw * s for s in size]
    probs = [max(x, floor) for x in probs]
    z = sum(probs)
    return [x / z for x in probs]


def static_posterior(wins, loge):
    ll = [0.0, 0.0]
    for _, y in wins:
        ll[0] += loge[0][y]
        ll[1] += loge[1][y]
    m = max(ll)
    w = [math.exp(ll[0] - m), math.exp(ll[1] - m)]
    z = w[0] + w[1]
    return [w[0] / z, w[1] / z], ll


def window_posterior(wins, loge, last_n=None, min_tick=None):
    chosen = wins
    if min_tick is not None:
        chosen = [(t, y) for t, y in wins if t >= min_tick]
    if last_n is not None:
        chosen = chosen[-last_n:]
    return static_posterior(chosen, loge)[0]


def hmm_posterior(pay, loge, eps):
    """P(state at cutoff | payouts before cutoff). Losses and gaps emit nothing."""
    log_stay = math.log(1.0 - eps) if eps < 1.0 else -1e300
    log_sw = math.log(eps) if eps > 0.0 else -1e300
    a = [math.log(0.5), math.log(0.5)]
    for y in pay:
        if y > 0:
            i = IDX[y]
            a[0] += loge[0][i]
            a[1] += loge[1][i]
        a = [
            logsumexp([a[0] + log_stay, a[1] + log_sw]),
            logsumexp([a[0] + log_sw, a[1] + log_stay]),
        ]
    margin = logsumexp(a)
    m = max(a)
    w = [math.exp(a[0] - m), math.exp(a[1] - m)]
    z = w[0] + w[1]
    return [w[0] / z, w[1] / z], margin


def hmm_wins_posterior(wins, loge, eps):
    log_stay = math.log(1.0 - eps) if eps < 1.0 else -1e300
    log_sw = math.log(eps) if eps > 0.0 else -1e300
    a = [math.log(0.5), math.log(0.5)]
    for _, y in wins:
        a[0] += loge[0][y]
        a[1] += loge[1][y]
        a = [
            logsumexp([a[0] + log_stay, a[1] + log_sw]),
            logsumexp([a[0] + log_sw, a[1] + log_stay]),
        ]
    margin = logsumexp(a)
    m = max(a)
    w = [math.exp(a[0] - m), math.exp(a[1] - m)]
    z = w[0] + w[1]
    return [w[0] / z, w[1] / z], margin


def changepoint_posterior(wins, loge, cutoff):
    """Constant atom, or one flip on a 5-tick grid. P(flip) = 0.5."""
    if not wins:
        return [0.5, 0.5]
    n = len(wins)
    p0 = [0.0] * (n + 1)
    p1 = [0.0] * (n + 1)
    for i, (_, y) in enumerate(wins):
        p0[i + 1] = p0[i] + loge[0][y]
        p1[i + 1] = p1[i] + loge[1][y]
    ticks = [t for t, _ in wins]
    grid = list(range(20, cutoff, 5))

    def n_before(c):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if ticks[mid] < c:
                lo = mid + 1
            else:
                hi = mid
        return lo

    logs = [(math.log(0.25) + p0[n], 1.0), (math.log(0.25) + p1[n], 0.0)]
    if grid:
        lp = math.log(0.5 / len(grid))
        for c in grid:
            k = n_before(c)
            logs.append((lp + p0[k] + (p1[n] - p1[k]), 0.0))
            logs.append((lp + p1[k] + (p0[n] - p0[k]), 1.0))
    m = max(L for L, _ in logs)
    num = [0.0, 0.0]
    for L, w0 in logs:
        e = math.exp(L - m)
        num[0] += e * w0
        num[1] += e * (1.0 - w0)
    z = num[0] + num[1]
    return [num[0] / z, num[1] / z]


def sign_predict(ys, feats, qfeat, loge):
    """Latent sign. Online log score falls back to the static posterior when the bit is missing."""
    ls = [0.0, 0.0]
    st = [0.0, 0.0]
    oll = 0.0
    for y, f in zip(ys, feats):
        if f is None:
            oll += logsumexp([st[0] + loge[0][y], st[1] + loge[1][y]]) - logsumexp(st)
            st[0] += loge[0][y]
            st[1] += loge[1][y]
            continue
        oll += logsumexp([ls[s] + loge[s ^ f][y] for s in (0, 1)]) - logsumexp(ls)
        ls[0] += loge[0 ^ f][y]
        ls[1] += loge[1 ^ f][y]
        st[0] += loge[0][y]
        st[1] += loge[1][y]
    if qfeat is None:
        m = max(st)
        w = [math.exp(st[0] - m), math.exp(st[1] - m)]
        z = w[0] + w[1]
        return oll, [w[0] / z, w[1] / z]
    m = max(ls)
    ps = [math.exp(ls[0] - m), math.exp(ls[1] - m)]
    z = ps[0] + ps[1]
    weights = [0.0, 0.0]
    for s in (0, 1):
        weights[s ^ qfeat] += ps[s] / z
    return oll, weights


def cell_predict(ys, cells, qcell, loge, ncells):
    ls = [[0.0, 0.0] for _ in range(ncells)]
    st = [0.0, 0.0]
    oll = 0.0
    for y, c in zip(ys, cells):
        if c is None:
            oll += logsumexp([st[0] + loge[0][y], st[1] + loge[1][y]]) - logsumexp(st)
            st[0] += loge[0][y]
            st[1] += loge[1][y]
            continue
        oll += logsumexp([ls[c][0] + loge[0][y], ls[c][1] + loge[1][y]]) - logsumexp(ls[c])
        ls[c][0] += loge[0][y]
        ls[c][1] += loge[1][y]
        st[0] += loge[0][y]
        st[1] += loge[1][y]
    src = st if qcell is None else ls[qcell]
    m = max(src)
    w = [math.exp(src[0] - m), math.exp(src[1] - m)]
    z = w[0] + w[1]
    return oll, [w[0] / z, w[1] / z]


def bma_predict(ys, feat_rows, qfeats, loge):
    """Uniform prior over binary features, each with a latent sign."""
    jn = len(qfeats)
    ls = [[0.0, 0.0] for _ in range(jn)]
    lp = [0.0] * jn
    oll = 0.0
    for n, y in enumerate(ys):
        joints = []
        for j in range(jn):
            f = feat_rows[j][n]
            if f is None:
                mix = logsumexp([math.log(0.5) + loge[0][y], math.log(0.5) + loge[1][y]])
                joints.append(lp[j] + mix)
            else:
                joints.append(lp[j] + logsumexp([ls[j][s] + loge[s ^ f][y] for s in (0, 1)]) - logsumexp(ls[j]))
        oll += logsumexp(joints) - logsumexp(lp)
        base = logsumexp(joints)
        for j in range(jn):
            f = feat_rows[j][n]
            if f is not None:
                ls[j][0] += loge[0 ^ f][y]
                ls[j][1] += loge[1 ^ f][y]
            lp[j] = joints[j] - base
    atom_terms = [[], []]
    for j, f in enumerate(qfeats):
        m = max(ls[j])
        ps = [math.exp(ls[j][0] - m), math.exp(ls[j][1] - m)]
        zs = ps[0] + ps[1]
        if f is None:
            for a in (0, 1):
                atom_terms[a].append(lp[j] + math.log(0.5))
        else:
            for s in (0, 1):
                atom_terms[s ^ f].append(lp[j] + math.log(ps[s] / zs))
    weights = [logsumexp(atom_terms[a]) for a in (0, 1)]
    m = max(weights)
    w = [math.exp(weights[0] - m), math.exp(weights[1] - m)]
    z = w[0] + w[1]
    return oll, [w[0] / z, w[1] / z]


def best_win_pair(pay):
    best, best_score = None, -1.0
    n = len(pay)
    start = max(0, n - 80)
    for a, b in CANON:
        w = [0, 0]
        c = [0, 0]
        for t in range(max(start, b), n):
            if pay[t] < 0 or pay[t - a] < 0 or pay[t - b] < 0:
                continue
            agree = int((pay[t - a] > 0) == (pay[t - b] > 0))
            c[agree] += 1
            w[agree] += int(pay[t] > 0)
        if c[0] < 8 or c[1] < 8:
            continue
        tot = c[0] + c[1]
        mean = (w[0] + w[1]) / tot
        score = sum(c[i] * (w[i] / c[i] - mean) ** 2 for i in (0, 1)) / tot
        if score > best_score:
            best_score = score
            live_agree = 1 if w[1] / c[1] >= w[0] / c[0] else 0
            best = (a, b, live_agree)
    return best


def acc_of(rows):
    if not rows:
        return float("nan")
    s = 0.0
    for w, y in rows:
        if abs(w[0] - w[1]) < 1e-12:
            s += 0.5
        else:
            s += float((w[1] > w[0]) == bool(y))
    return s / len(rows)


def mean_true(rows):
    if not rows:
        return float("nan")
    return sum(w[y] for w, y in rows) / len(rows)


def build_specs():
    specs = [("const",)]
    for bname in BITS:
        for lag in range(1, 9):
            specs.append(("lag", bname, lag))
        for a, b in PAIRS:
            for op in ("xor", "and", "or"):
                specs.append((op, bname, a, b))
        for k in range(2, 9):
            specs.append(("xorset", bname, tuple(range(1, k + 1))))
        specs.append(("xorset", bname, (1, 2, 4)))
        specs.append(("lastwin", bname))
    return specs


def spec_name(spec):
    kind = spec[0]
    if kind == "const":
        return "const0"
    if kind == "lag":
        return f"{spec[1]}@{spec[2]}"
    if kind in ("xor", "and", "or"):
        op = {"xor": "^^", "and": "&", "or": "|"}[kind]
        return f"{spec[1]}{op}{spec[2]},{spec[3]}"
    if kind == "xorset":
        return spec[1] + "^" + "+".join(str(x) for x in spec[2])
    if kind == "lastwin":
        return f"{spec[1]}@lastwin"
    return str(spec)


def eval_spec_at(pay, t, spec, row_by_bit):
    kind = spec[0]
    if kind == "const":
        return 0
    if kind == "lastwin":
        bit = BITS[spec[1]]
        for lag in range(1, 9):
            j = t - lag
            if j < 0 or pay[j] < 0:
                return None
            if pay[j] > 0:
                return bit(pay[j])
        return None
    row = row_by_bit[spec[1]]
    if kind == "lag":
        return row[spec[2] - 1]
    if kind == "xorset":
        x = 0
        for lag in spec[2]:
            b = row[lag - 1]
            if b is None:
                return None
            x ^= b
        return x
    x, y = row[spec[2] - 1], row[spec[3] - 1]
    if x is None or y is None:
        return None
    if kind == "xor":
        return x ^ y
    if kind == "and":
        return x & y
    return x | y


def rows_for(pay, ticks, bitname):
    bit = BITS[bitname]
    return [[bit_at(pay, t, lag, bit) for lag in range(1, 9)] for t in ticks]


def feature_columns(pay, wins, cutoff, specs):
    ticks = [t for t, _ in wins] + [cutoff]
    by_bit = {name: rows_for(pay, ticks, name) for name in BITS}
    cols = []
    for spec in specs:
        feats = [eval_spec_at(pay, ticks[i], spec, {b: by_bit[b][i] for b in BITS}) for i in range(len(wins))]
        qf = eval_spec_at(pay, cutoff, spec, {b: by_bit[b][-1] for b in BITS})
        cols.append((feats, qf))
    return cols


def cell_columns(pay, wins, cutoff):
    """4-cell maps of a bit pair, plus the exact payout class at lags 1, 2 and 4."""
    out = {}
    ticks = [t for t, _ in wins] + [cutoff]
    for bname, bit in BITS.items():
        if bname == "eq20":
            continue
        rows = rows_for(pay, ticks, bname)
        for a, b in PAIRS:
            cells = []
            for row in rows:
                xa, xb = row[a - 1], row[b - 1]
                cells.append(None if xa is None or xb is None else xa * 2 + xb)
            out[f"cell4:{bname}@{a},{b}"] = (cells[:-1], cells[-1], 4)
    for lag in (1, 2, 4):
        cells = []
        for t in ticks:
            j = t - lag
            if j < 0 or pay[j] < 0:
                cells.append(None)
            else:
                cells.append(CLASS_ID[pay[j]])
        out[f"class@{lag}"] = (cells[:-1], cells[-1], 5)
    return out


def acc_mean(hit, pt, n):
    if n == 0:
        return float("nan"), float("nan")
    return hit / n, pt / n


def write_preds(name, base_rows, weights_by_q):
    path = PRED / name
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for r in base_rows:
            p0 = float(r["p_0"])
            weights, atoms = weights_by_q[r["query_id"]]
            probs = mix_probs(p0, weights, atoms)
            w.writerow([r["query_id"]] + [f"{x:.10f}" for x in probs])
    return path


def score_file(path):
    val = MAIN / "public" / "validation"
    return score_contest_files(
        path, val / "targets.csv", val / "queries.csv", val / "conditions.csv",
        val / "reference.csv", max_points=60,
    )


def fmt_score(s):
    cond = {c: round(v["loss_ratio"], 4) for c, v in s["by_condition"].items()}
    return round(s["technical_points"], 3), cond


def theoretical_error(atom_a, atom_b, n):
    """Normal approximation to P(static Bayes picks the wrong atom) after n wins from A."""
    mu = 0.0
    var = 0.0
    for i in range(4):
        lr = math.log(atom_a[i] / atom_b[i])
        mu += atom_a[i] * lr
        var += atom_a[i] * lr * lr
    var -= mu * mu
    if var <= 0 or n <= 0:
        return 0.5
    z = -mu * math.sqrt(n) / math.sqrt(var)
    # Phi(z)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main():
    print("loading", flush=True)
    vq, vc, vh, vt = load_split("validation")
    tq, _, th, _ = load_split("train")
    atoms, true_atom, worst = atoms_from_targets(vq, vt)
    print("within-atom max deviation", f"{worst:.3e}")
    for kind, vecs in atoms.items():
        print(f"  {kind}: {len(vecs)} atoms")
        for i, vec in enumerate(vecs):
            print("   ", i, ", ".join(f"{x:.10f}" for x in vec))
        print(f"    KL(0||1)={sum(vecs[0][i]*math.log(vecs[0][i]/vecs[1][i]) for i in range(4)):.4f} "
              f"KL(1||0)={sum(vecs[1][i]*math.log(vecs[1][i]/vecs[0][i]) for i in range(4)):.4f}")
        for n in (10, 30, 60):
            e0 = theoretical_error(vecs[0], vecs[1], n)
            e1 = theoretical_error(vecs[1], vecs[0], n)
            print(f"    if static, P(wrong) after {n} wins ≈ {(e0+e1)/2:.4f}")
    loge = {k: [[math.log(max(x, 1e-15)) for x in v] for v in vecs] for k, vecs in atoms.items()}
    qmap = {r["query_id"]: r for r in vq}
    base_rows = list(csv.DictReader(open(V6)))

    def pack(queries, hist):
        rows = []
        for q in queries:
            pay, wins, cutoff, gaps = series_of(q, hist)
            rows.append((q, pay, wins, cutoff, gaps))
        return rows

    print("packing series", flush=True)
    vrows = pack(vq, vh)
    trows = pack(tq, th)
    print(f"train machines {len(trows)} mean wins {sum(len(w) for _,_,w,_,_ in trows)/len(trows):.1f} "
          f"mean gaps {sum(g for *_, g in trows)/len(trows):.1f}")
    print(f"val machines {len(vrows)} mean wins {sum(len(w) for _,_,w,_,_ in vrows)/len(vrows):.1f} "
          f"mean gaps {sum(g for *_, g in vrows)/len(vrows):.1f}")
    lag_ok = {lag: 0 for lag in (1, 2, 4, 8)}
    for q, pay, wins, cutoff, _ in vrows:
        for lag in lag_ok:
            lag_ok[lag] += int(cutoff - lag >= 0 and pay[cutoff - lag] >= 0)
    print("val queries with lag observed", {k: v for k, v in lag_ok.items()})

    # ---------- static, windows, diagnostics ----------
    print("\n== static posterior ==", flush=True)
    static_w = {}
    groups = defaultdict(list)
    odds = []
    for q, pay, wins, cutoff, _ in vrows:
        kind = q["machine_type"]
        w, ll = static_posterior(wins, loge[kind])
        y = true_atom[q["query_id"]]
        static_w[q["query_id"]] = (w, atoms[kind])
        rec = (w, y)
        cond = vc[q["query_id"]]
        groups[("all", "all")].append(rec)
        groups[("cond", cond)].append(rec)
        groups[("type", kind)].append(rec)
        groups[("ct", cond + "/" + kind)].append(rec)
        n = len(wins)
        bin_name = "0" if n == 0 else "1-9" if n < 10 else "10-29" if n < 30 else "30-59" if n < 60 else "60+"
        groups[("bin", bin_name)].append(rec)
        odds.append(ll[y] - ll[1 - y])
    print(f"{'slice':<28}{'n':>6}{'acc':>8}{'P(true)':>10}{'conf':>8}{'sure':>7}{'sure_wrong':>11}")
    order = [("all", "all")]
    order += [("cond", c) for c in ("stable", "short_history", "drift", "shared_change")]
    order += [("type", t) for t in ("steady", "balanced", "jackpot")]
    order += [("bin", b) for b in ("0", "1-9", "10-29", "30-59", "60+")]
    order += [("ct", k) for _, k in sorted(kk for kk in groups if kk[0] == "ct")]
    for key in order:
        rows = groups[key]
        n = len(rows)
        conf = sum(max(w) for w, _ in rows) / n
        sure = [(w, y) for w, y in rows if max(w) >= 0.8]
        wrong = sum(1 for w, y in sure if (w[1] > w[0]) != bool(y) and abs(w[0] - w[1]) >= 1e-12)
        print(f"{key[1]:<28}{n:6d}{acc_of(rows):8.3f}{mean_true(rows):10.3f}{conf:8.3f}{len(sure):7d}{wrong:11d}")
    neg = sum(1 for x in odds if x < 0)
    print(f"LLR(true): median {sorted(odds)[len(odds)//2]:.2f}  frac<0 {neg/len(odds):.3f}  "
          f"frac<-2 {sum(1 for x in odds if x < -2)/len(odds):.3f}  "
          f"frac>2 {sum(1 for x in odds if x > 2)/len(odds):.3f}")

    print("\n== windows ==", flush=True)
    window_specs = {
        "last10wins": lambda wins, c: window_posterior(wins, None, last_n=10),
    }
    window_w = {}
    for label, kwargs in (
        ("last10wins", {"last_n": 10}),
        ("last20wins", {"last_n": 20}),
        ("last40wins", {"last_n": 40}),
        ("last30ticks", {"min_tick": "c-30"}),
        ("last60ticks", {"min_tick": "c-60"}),
        ("early_half", {"min_tick": 0, "max_tick": "c/2"}),
        ("late_half", {"min_tick": "c/2"}),
    ):
        store = {}
        rows_by = defaultdict(list)
        for q, pay, wins, cutoff, _ in vrows:
            kind = q["machine_type"]
            kw = {}
            if "last_n" in kwargs:
                kw["last_n"] = kwargs["last_n"]
            if kwargs.get("min_tick") == "c-30":
                kw["min_tick"] = cutoff - 30
            elif kwargs.get("min_tick") == "c-60":
                kw["min_tick"] = cutoff - 60
            elif kwargs.get("min_tick") == "c/2":
                kw["min_tick"] = cutoff / 2
            chosen_min = kw.get("min_tick")
            chosen = wins
            if chosen_min is not None:
                chosen = [(t, y) for t, y in wins if t >= chosen_min]
            if kwargs.get("max_tick") == "c/2":
                chosen = [(t, y) for t, y in wins if t < cutoff / 2]
            if "last_n" in kw:
                chosen = wins[-kw["last_n"]:]
            w = static_posterior(chosen, loge[kind])[0]
            y = true_atom[q["query_id"]]
            store[q["query_id"]] = (w, atoms[kind])
            rows_by["all"].append((w, y))
            rows_by[vc[q["query_id"]]].append((w, y))
        window_w[label] = store
        bits = "  ".join(f"{c[:6]} {acc_of(rows_by[c]):.3f}" for c in
                         ("stable", "short_history", "drift", "shared_change"))
        print(f"  {label:<14} acc {acc_of(rows_by['all']):.3f}  P(true) {mean_true(rows_by['all']):.3f}  {bits}")

    # early vs late disagreement
    print("\nearly-half ML atom vs late-half ML atom:")
    for cond in ("all", "stable", "short_history", "drift", "shared_change"):
        dis = same_late = n = 0
        for q, pay, wins, cutoff, _ in vrows:
            if cond != "all" and vc[q["query_id"]] != cond:
                continue
            ew = window_w["early_half"][q["query_id"]][0]
            lw = window_w["late_half"][q["query_id"]][0]
            if abs(ew[0] - ew[1]) < 1e-9 or abs(lw[0] - lw[1]) < 1e-9:
                continue
            n += 1
            e_ml, l_ml = int(ew[1] > ew[0]), int(lw[1] > lw[0])
            y = true_atom[q["query_id"]]
            if e_ml != l_ml:
                dis += 1
                same_late += int(l_ml == y)
        print(f"  {cond:<16} comparable {n:4d}  disagree {dis:4d} ({dis/n if n else float('nan'):.3f})  "
              f"when disagree, late matches query {same_late}/{dis}")

    # ---------- HMM ----------
    print("\n== HMM, train marginal likelihood ==", flush=True)
    eps_grid = (0.0, 0.0003, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
    hmm_train = []
    for eps in eps_grid:
        total = 0.0
        total_w = 0.0
        for q, pay, wins, cutoff, _ in trows:
            L = loge[q["machine_type"]]
            total += hmm_posterior(pay, L, eps)[1]
            total_w += hmm_wins_posterior(wins, L, eps)[1]
        hmm_train.append((eps, total, total_w))
        print(f"  eps {eps:<7} tickLL {total:.1f}  winLL {total_w:.1f}", flush=True)
    best_eps = max(hmm_train, key=lambda r: r[1])[0]
    best_eps_w = max(hmm_train, key=lambda r: r[2])[0]
    print(f"selected tick eps {best_eps}  win-step eps {best_eps_w}")
    hmm_w, win_hmm_w = {}, {}
    for q, pay, wins, cutoff, _ in vrows:
        kind = q["machine_type"]
        y = true_atom[q["query_id"]]
        w, _ = hmm_posterior(pay, loge[kind], best_eps)
        hmm_w[q["query_id"]] = (w, atoms[kind])
        ww, _ = hmm_wins_posterior(wins, loge[kind], best_eps_w)
        win_hmm_w[q["query_id"]] = (ww, atoms[kind])
    rows = [(hmm_w[q["query_id"]][0], true_atom[q["query_id"]]) for q, *_ in vrows]
    print(f"HMM tick val acc {acc_of(rows):.3f} P(true) {mean_true(rows):.3f}")
    rows = [(win_hmm_w[q["query_id"]][0], true_atom[q["query_id"]]) for q, *_ in vrows]
    print(f"HMM win-step val acc {acc_of(rows):.3f} P(true) {mean_true(rows):.3f}")
    print("val accuracy by tick eps (diagnostic only):")
    best_val_eps, best_val_acc = None, -1
    eps_store = {}
    for eps in eps_grid:
        store = {}
        rr = []
        for q, pay, wins, cutoff, _ in vrows:
            kind = q["machine_type"]
            w, _ = hmm_posterior(pay, loge[kind], eps)
            store[q["query_id"]] = (w, atoms[kind])
            rr.append((w, true_atom[q["query_id"]]))
        aa = acc_of(rr)
        print(f"  eps {eps:<7} acc {aa:.3f} P(true) {mean_true(rr):.3f}")
        if aa > best_val_acc:
            best_val_acc, best_val_eps, eps_store = aa, eps, store
    print(f"val-oracle eps {best_val_eps} acc {best_val_acc:.3f} (not used as the selected model)")

    print("\n== changepoint ==", flush=True)
    cp_w = {}
    rr = defaultdict(list)
    for q, pay, wins, cutoff, _ in vrows:
        kind = q["machine_type"]
        w = changepoint_posterior(wins, loge[kind], cutoff)
        cp_w[q["query_id"]] = (w, atoms[kind])
        y = true_atom[q["query_id"]]
        rr["all"].append((w, y))
        rr[vc[q["query_id"]]].append((w, y))
    print(f"changepoint acc {acc_of(rr['all']):.3f} P(true) {mean_true(rr['all']):.3f}")
    for cond in ("stable", "short_history", "drift", "shared_change"):
        print(f"  {cond:<16} {acc_of(rr[cond]):.3f}  P(true) {mean_true(rr[cond]):.3f}")

    # ---------- rule search ----------
    print("\n== rule search ==", flush=True)
    specs = build_specs()
    names = [spec_name(s) for s in specs]
    print(f"{len(specs)} binary specs", flush=True)
    train_oll = defaultdict(float)
    train_gll = defaultdict(lambda: [0.0, 0.0])
    train_static = 0.0
    n_train_wins = 0
    for i, (q, pay, wins, cutoff, _) in enumerate(trows):
        L = loge[q["machine_type"]]
        ys = [y for _, y in wins]
        st = [0.0, 0.0]
        for y in ys:
            train_static += logsumexp([st[0] + L[0][y], st[1] + L[1][y]]) - logsumexp(st)
            st[0] += L[0][y]
            st[1] += L[1][y]
            n_train_wins += 1
        cols = feature_columns(pay, wins, cutoff, specs)
        for name, (feats, qf) in zip(names, cols):
            oll, _ = sign_predict(ys, feats, qf, L)
            train_oll[name] += oll
            for y, f in zip(ys, feats):
                if f is None:
                    continue
                train_gll[name][0] += L[0 ^ f][y]
                train_gll[name][1] += L[1 ^ f][y]
        if i % 200 == 0:
            print(f"  train {i}/{len(trows)}", flush=True)
    print(f"train static online LL {train_static:.1f} on {n_train_wins} wins")
    ranked = sorted(names, key=lambda n: train_oll[n], reverse=True)
    print(f"{'feature':<24}{'gain':>10}{'globalSign':>12}")
    for name in ranked[:20]:
        sign = 0 if train_gll[name][0] >= train_gll[name][1] else 1
        print(f"{name:<24}{train_oll[name]-train_static:10.1f}{sign:12d}")

    cell_names_order = None
    cell_oll = defaultdict(float)
    for i, (q, pay, wins, cutoff, _) in enumerate(trows):
        L = loge[q["machine_type"]]
        ys = [y for _, y in wins]
        cells = cell_columns(pay, wins, cutoff)
        if cell_names_order is None:
            cell_names_order = list(cells)
        for name, (cs, qc, nc) in cells.items():
            cell_oll[name] += cell_predict(ys, cs, qc, L, nc)[0]
        if i % 200 == 0:
            print(f"  train cells {i}/{len(trows)}", flush=True)
    cranked = sorted(cell_names_order, key=lambda n: cell_oll[n], reverse=True)
    print("cell models, train gain vs static:")
    for name in cranked[:12]:
        print(f"  {name:<28} {cell_oll[name]-train_static:8.1f}")

    print("\nvalidation pass", flush=True)
    # val_hit uses half-credit ties; val_pt sums P(true); ghit[s] counts a fixed global sign
    val_hit = defaultdict(float)
    val_pt = defaultdict(float)
    val_n = defaultdict(int)
    val_cond = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    ghit = defaultdict(lambda: [0, 0])
    gn = defaultdict(int)
    for i, (q, pay, wins, cutoff, _) in enumerate(vrows):
        kind = q["machine_type"]
        L = loge[kind]
        ys = [y for _, y in wins]
        ytrue = true_atom[q["query_id"]]
        cond = vc[q["query_id"]]
        cols = feature_columns(pay, wins, cutoff, specs)
        for name, (feats, qf) in zip(names, cols):
            _, w = sign_predict(ys, feats, qf, L)
            if abs(w[0] - w[1]) < 1e-12:
                add = 0.5
            else:
                add = float((w[1] > w[0]) == bool(ytrue))
            val_hit[name] += add
            val_pt[name] += w[ytrue]
            val_n[name] += 1
            bucket = val_cond[name][cond]
            bucket[0] += add
            bucket[1] += w[ytrue]
            bucket[2] += 1
            if qf is not None:
                ghit[name][0] += int(qf == ytrue)
                ghit[name][1] += int((1 ^ qf) == ytrue)
                gn[name] += 1
        if i % 200 == 0:
            print(f"  val {i}/{len(vrows)}", flush=True)

    val_ranked = sorted(names, key=lambda n: val_hit[n], reverse=True)
    print(f"\n{'feature':<24}{'valAcc':>8}{'P(true)':>9}{'globAcc':>9}{'trainGain':>11}")
    for name in val_ranked[:25]:
        acc, pt = acc_mean(val_hit[name], val_pt[name], val_n[name])
        sign = 0 if train_gll[name][0] >= train_gll[name][1] else 1
        gacc = ghit[name][sign] / gn[name] if gn[name] else float("nan")
        print(f"{name:<24}{acc:8.3f}{pt:9.3f}{gacc:9.3f}{train_oll[name]-train_static:11.1f}")
    print("train-selected top 8 on validation:")
    for name in ranked[:8]:
        acc, pt = acc_mean(val_hit[name], val_pt[name], val_n[name])
        print(f"  {name:<24} valAcc {acc:.3f} P(true) {pt:.3f} gain {train_oll[name]-train_static:.1f}")
        for cond in ("stable", "short_history", "drift", "shared_change"):
            h, p, n = val_cond[name][cond]
            print(f"      {cond:<16} {h/n:.3f}  P(true) {p/n:.3f}")

    print("\ncell validation (second pass, top 8 by train and we also track all for the max)")
    cell_val = {}
    # Score every cell model. 84 models * 768 * ~80 is acceptable once.
    cell_hit = defaultdict(float)
    cell_pt = defaultdict(float)
    for i, (q, pay, wins, cutoff, _) in enumerate(vrows):
        L = loge[q["machine_type"]]
        ys = [y for _, y in wins]
        ytrue = true_atom[q["query_id"]]
        cells = cell_columns(pay, wins, cutoff)
        for name, (cs, qc, nc) in cells.items():
            _, w = cell_predict(ys, cs, qc, L, nc)
            if abs(w[0] - w[1]) < 1e-12:
                cell_hit[name] += 0.5
            else:
                cell_hit[name] += float((w[1] > w[0]) == bool(ytrue))
            cell_pt[name] += w[ytrue]
        if i % 200 == 0:
            print(f"  val cells {i}", flush=True)
    cell_by_val = sorted(cell_names_order, key=lambda n: cell_hit[n], reverse=True)
    print("best cells by validation accuracy (label-aware ranking, an upper bound):")
    for name in cell_by_val[:8]:
        print(f"  {name:<28} acc {cell_hit[name]/len(vrows):.3f} P(true) {cell_pt[name]/len(vrows):.3f} "
              f"trainGain {cell_oll[name]-train_static:.1f}")
    print("train-selected cells:")
    for name in cranked[:5]:
        print(f"  {name:<28} acc {cell_hit[name]/len(vrows):.3f} P(true) {cell_pt[name]/len(vrows):.3f} "
              f"trainGain {cell_oll[name]-train_static:.1f}")

    # ---------- BMA families + win-pair features ----------
    print("\n== BMA and win-pair features ==", flush=True)
    families = {
        "bma_ge5_lags+static": ["const0"] + [f"ge5@{k}" for k in range(1, 9)],
        "bma_ge2_lags+static": ["const0"] + [f"ge2@{k}" for k in range(1, 9)],
        "bma_win_lags+static": ["const0"] + [f"win@{k}" for k in range(1, 9)],
        "bma_ge5_xor+static": ["const0"] + [f"ge5^^{a},{b}" for a, b in PAIRS],
        "bma_ge2_xor+static": ["const0"] + [f"ge2^^{a},{b}" for a, b in PAIRS],
        "bma_win_xor+static": ["const0"] + [f"win^^{a},{b}" for a, b in PAIRS],
        "bma_ge5_canon+static": ["const0"] + [f"ge5^^{a},{b}" for a, b in CANON],
        "bma_ge2_124+static": ["const0", "ge2@1", "ge2@2", "ge2@4", "ge2^1+2+4"],
        "bma_ge5_124+static": ["const0", "ge5@1", "ge5@2", "ge5@4", "ge5^1+2+4"],
    }
    name_to_spec = {spec_name(s): s for s in specs}

    def columns_named(pay, wins, cutoff, member_names):
        member_specs = [name_to_spec[n] for n in member_names]
        cols = feature_columns(pay, wins, cutoff, member_specs)
        return [c[0] for c in cols], [c[1] for c in cols]

    bma_w = {}
    for fname, members in families.items():
        toll = 0.0
        for q, pay, wins, cutoff, _ in trows:
            L = loge[q["machine_type"]]
            ys = [y for _, y in wins]
            feat_rows, qfeats = columns_named(pay, wins, cutoff, members)
            toll += bma_predict(ys, feat_rows, qfeats, L)[0]
        weights = {}
        rr = []
        for q, pay, wins, cutoff, _ in vrows:
            kind = q["machine_type"]
            ys = [y for _, y in wins]
            feat_rows, qfeats = columns_named(pay, wins, cutoff, members)
            _, w = bma_predict(ys, feat_rows, qfeats, loge[kind])
            weights[q["query_id"]] = (w, atoms[kind])
            rr.append((w, true_atom[q["query_id"]]))
        bma_w[fname] = weights
        print(f"  {fname:<24} gain {toll-train_static:8.1f} acc {acc_of(rr):.3f} P(true) {mean_true(rr):.3f}",
              flush=True)

    print("win-pair (pair chosen from recent win/loss separation, then a size bit on that pair):")
    winpair_w = {}
    for bname in ("win", "ge2", "ge5"):
        bit = BITS[bname]
        weights = {}
        rr = []
        toll = 0.0
        for split, bucket, dest in ((trows, True, None), (vrows, False, weights)):
            for q, pay, wins, cutoff, _ in split:
                kind = q["machine_type"]
                L = loge[kind]
                pair = best_win_pair(pay)
                ys = [y for _, y in wins]
                if pair is None:
                    feats = [None] * len(wins)
                    qf = None
                else:
                    a, b, _ = pair
                    feats = []
                    for t, _y in wins:
                        xa, xb = bit_at(pay, t, a, bit), bit_at(pay, t, b, bit)
                        feats.append(None if xa is None or xb is None else xa ^ xb)
                    xa, xb = bit_at(pay, cutoff, a, bit), bit_at(pay, cutoff, b, bit)
                    qf = None if xa is None or xb is None else xa ^ xb
                oll, w = sign_predict(ys, feats, qf, L)
                if bucket:
                    toll += oll
                else:
                    dest[q["query_id"]] = (w, atoms[kind])
                    rr.append((w, true_atom[q["query_id"]]))
        # The loop above mixes train and val incorrectly because dest is only set on the second
        # iteration variable. Rewrite cleanly below.
        winpair_w[bname] = (toll, weights, rr)
    # clean rewrite
    winpair_w = {}
    for bname in ("win", "ge2", "ge5"):
        bit = BITS[bname]

        def run(split, bit=bit):
            total = 0.0
            weights = {}
            rr = []
            for q, pay, wins, cutoff, _ in split:
                kind = q["machine_type"]
                L = loge[kind]
                pair = best_win_pair(pay)
                ys = [y for _, y in wins]
                if pair is None:
                    feats = [None] * len(wins)
                    qf = None
                else:
                    a, b, _ = pair
                    feats = []
                    for t, _y in wins:
                        xa, xb = bit_at(pay, t, a, bit), bit_at(pay, t, b, bit)
                        feats.append(None if xa is None or xb is None else xa ^ xb)
                    xa, xb = bit_at(pay, cutoff, a, bit), bit_at(pay, cutoff, b, bit)
                    qf = None if xa is None or xb is None else xa ^ xb
                oll, w = sign_predict(ys, feats, qf, L)
                total += oll
                weights[q["query_id"]] = (w, atoms[kind])
                if q["query_id"] in true_atom:
                    rr.append((w, true_atom[q["query_id"]]))
            return total, weights, rr

        tr_ll, _, _ = run(trows)
        _, weights, rr = run(vrows)
        winpair_w[bname] = weights
        print(f"  winpair {bname:<6} trainGain {tr_ll-train_static:8.1f} acc {acc_of(rr):.3f} P(true) {mean_true(rr):.3f}")

    # live bit vs atom
    print("\n== live, signal, index ==")
    live_tab = defaultdict(list)
    sig_tab = defaultdict(lambda: [0, 0])
    idx_tab = defaultdict(lambda: [0, 0])
    for q, pay, wins, cutoff, _ in vrows:
        pair = best_win_pair(pay)
        y = true_atom[q["query_id"]]
        sig_tab[(q["machine_type"], int(q["signal"]))][y] += 1
        idx_tab[int(q["machine_id"][-3:])][y] += 1
        if pair is None:
            continue
        a, b, live_agree = pair
        if pay[cutoff - a] < 0 or pay[cutoff - b] < 0:
            continue
        agree = int((pay[cutoff - a] > 0) == (pay[cutoff - b] > 0))
        live = int(agree == live_agree)
        live_tab[q["machine_type"]].append((live, y))
        live_tab["all"].append((live, y))
    for kind, rows in live_tab.items():
        match = sum(1 for live, y in rows if live == y) / len(rows)
        print(f"  P(atom==live | {kind}) {match:.3f}  best orient {max(match, 1-match):.3f}  n={len(rows)}")
    for kind in ("steady", "balanced", "jackpot"):
        for sig in (-1, 0, 1):
            c0, c1 = sig_tab[(kind, sig)]
            tot = c0 + c1
            print(f"  {kind} signal {sig:+d}: P(atom1) {c1/tot:.3f} n={tot}")
    for mid in range(8):
        c0, c1 = idx_tab[mid]
        print(f"  m{mid} P(atom1) {c1/(c0+c1):.3f}")

    # neighbours: global orientation, and per-machine sign for a short list
    print("\n== neighbours ==")
    by_ep = defaultdict(list)
    for item in vrows:
        by_ep[item[0]["episode_id"]].append(item)
    ep_pay = {}
    ep_mids = {}
    for ep, items in by_ep.items():
        ep_mids[ep] = sorted(it[0]["machine_id"] for it in items)
        ep_pay[ep] = {it[0]["machine_id"]: it[1] for it in items}
    neigh_global = []
    for bname, bit in BITS.items():
        if bname == "eq20":
            continue
        for lag in (1, 2, 4):
            for offset in range(1, 8):
                agree = n = 0
                for q, pay, wins, cutoff, _ in vrows:
                    mids = ep_mids[q["episode_id"]]
                    other = mids[(mids.index(q["machine_id"]) + offset) % 8]
                    f = bit_at(ep_pay[q["episode_id"]][other], cutoff, lag, bit)
                    if f is None:
                        continue
                    agree += int(f == true_atom[q["query_id"]])
                    n += 1
                if n:
                    neigh_global.append((max(agree, n - agree) / n, f"off{offset}:{bname}@{lag}", n))
    neigh_global.sort(reverse=True)
    print(f"  best global neighbour acc {neigh_global[0][0]:.3f} ({neigh_global[0][1]})  "
          f"median {sorted(a for a,_,_ in neigh_global)[len(neigh_global)//2]:.3f}")
    for row in neigh_global[:8]:
        print(f"    {row[0]:.3f}  {row[1]} n={row[2]}")

    t_by_ep = defaultdict(list)
    for item in trows:
        t_by_ep[item[0]["episode_id"]].append(item)
    neigh_models = {}
    print("per-machine neighbour sign:")
    for bname in ("ge2", "ge5", "win"):
        bit = BITS[bname]
        for lag in (1, 2, 4):
            for offset in (1, 4):
                toll = 0.0
                for ep, items in t_by_ep.items():
                    mids = sorted(it[0]["machine_id"] for it in items)
                    pays = {it[0]["machine_id"]: it[1] for it in items}
                    for q, pay, wins, cutoff, _ in items:
                        other = mids[(mids.index(q["machine_id"]) + offset) % 8]
                        ys = [y for _, y in wins]
                        feats = [bit_at(pays[other], t, lag, bit) for t, _ in wins]
                        qf = bit_at(pays[other], cutoff, lag, bit)
                        toll += sign_predict(ys, feats, qf, loge[q["machine_type"]])[0]
                weights = {}
                rr = []
                for q, pay, wins, cutoff, _ in vrows:
                    mids = ep_mids[q["episode_id"]]
                    other = mids[(mids.index(q["machine_id"]) + offset) % 8]
                    ys = [y for _, y in wins]
                    feats = [bit_at(ep_pay[q["episode_id"]][other], t, lag, bit) for t, _ in wins]
                    qf = bit_at(ep_pay[q["episode_id"]][other], cutoff, lag, bit)
                    _, w = sign_predict(ys, feats, qf, loge[q["machine_type"]])
                    weights[q["query_id"]] = (w, atoms[q["machine_type"]])
                    rr.append((w, true_atom[q["query_id"]]))
                name = f"neigh{offset}:{bname}@{lag}"
                neigh_models[name] = (toll - train_static, weights, acc_of(rr), mean_true(rr))
                print(f"  {name:<20} gain {toll-train_static:8.1f} acc {acc_of(rr):.3f} P(true) {mean_true(rr):.3f}")

    # v6's own size mix
    print("\n== v6 size mix vs atoms ==")
    v6_stats = defaultdict(list)
    for r in base_rows:
        k = r["query_id"]
        kind = qmap[k]["machine_type"]
        pw = 1.0 - float(r["p_0"])
        if pw < 1e-8:
            continue
        mix = [float(r["p_" + p]) / pw for p in PAY]
        A, B = atoms[kind]
        num = sum((mix[i] - B[i]) * (A[i] - B[i]) for i in range(4))
        den = sum((A[i] - B[i]) ** 2 for i in range(4))
        wA = min(1.0, max(0.0, num / den))
        y = true_atom[k]
        dA = sum((mix[i] - A[i]) ** 2 for i in range(4))
        dB = sum((mix[i] - B[i]) ** 2 for i in range(4))
        nearest = 0 if dA <= dB else 1
        rec = (nearest == y, wA if y == 0 else 1.0 - wA)
        v6_stats["all"].append(rec)
        v6_stats[vc[k]].append(rec)
    for key, rows in v6_stats.items():
        print(f"  {key:<16} nearest {sum(a for a,_ in rows)/len(rows):.3f}  "
              f"weight on true {sum(b for _,b in rows)/len(rows):.3f}")

    # ---------- rebuild weights for the models we score ----------
    print("\n== contest scores ==", flush=True)

    def weights_for_named(member_names, how):
        """how is 'sign' for one feature or 'bma' for a list. member_names is a list of spec names."""
        out = {}
        member_specs = [name_to_spec[n] for n in member_names]
        for q, pay, wins, cutoff, _ in vrows:
            kind = q["machine_type"]
            ys = [y for _, y in wins]
            cols = feature_columns(pay, wins, cutoff, member_specs)
            if how == "sign":
                _, w = sign_predict(ys, cols[0][0], cols[0][1], loge[kind])
            else:
                _, w = bma_predict(ys, [c[0] for c in cols], [c[1] for c in cols], loge[kind])
            out[q["query_id"]] = (w, atoms[kind])
        return out

    def weights_for_cell(cell_name):
        out = {}
        for q, pay, wins, cutoff, _ in vrows:
            kind = q["machine_type"]
            ys = [y for _, y in wins]
            cs, qc, nc = cell_columns(pay, wins, cutoff)[cell_name]
            _, w = cell_predict(ys, cs, qc, loge[kind], nc)
            out[q["query_id"]] = (w, atoms[kind])
        return out

    pair_ranked = [n for n in ranked if "^^" in n or "&" in n or "|" in n]
    models = {
        "static": static_w,
        "hmm_train_eps": hmm_w,
        "hmm_winstep": win_hmm_w,
        "hmm_val_oracle_eps": eps_store,
        "changepoint": cp_w,
        "half": {q["query_id"]: ([0.5, 0.5], atoms[q["machine_type"]]) for q, *_ in vrows},
        "oracle": {
            q["query_id"]: ([1.0, 0.0] if true_atom[q["query_id"]] == 0 else [0.0, 1.0], atoms[q["machine_type"]])
            for q, *_ in vrows
        },
        "last20wins": window_w["last20wins"],
        "last60ticks": window_w["last60ticks"],
        "late_half": window_w["late_half"],
        "train_best_rule": None,  # filled below
        "train_best_pair": None,
        "val_best_rule_TUNED": None,
        "train_best_cell": None,
    }
    print("rebuilding scored rules:", ranked[0], pair_ranked[0], val_ranked[0], cranked[0], flush=True)
    models["train_best_rule"] = weights_for_named([ranked[0]], "sign")
    models["train_best_pair"] = weights_for_named([pair_ranked[0]], "sign")
    models["val_best_rule_TUNED"] = weights_for_named([val_ranked[0]], "sign")
    models["train_best_cell"] = weights_for_cell(cranked[0])
    models["val_best_cell_TUNED"] = weights_for_cell(cell_by_val[0])
    for fname, weights in bma_w.items():
        models[fname] = weights
    for bname, weights in winpair_w.items():
        models[f"winpair_{bname}"] = weights
    # best neighbour by train gain
    best_neigh = max(neigh_models, key=lambda n: neigh_models[n][0])
    models["train_best_neigh"] = neigh_models[best_neigh][1]
    print("train-best neighbour", best_neigh)

    # tempered static, alpha 0.5, as a calibration check (not selected)
    tempered = {}
    for q, pay, wins, cutoff, _ in vrows:
        kind = q["machine_type"]
        ll = [0.0, 0.0]
        for _, y in wins:
            ll[0] += 0.5 * loge[kind][0][y]
            ll[1] += 0.5 * loge[kind][1][y]
        m = max(ll)
        w = [math.exp(ll[0] - m), math.exp(ll[1] - m)]
        z = sum(w)
        tempered[q["query_id"]] = ([w[0] / z, w[1] / z], atoms[kind])
    models["static_alpha0.5"] = tempered

    results = {}
    s = score_file(V6)
    pt, cond = fmt_score(s)
    print(f"{'v6':<28} {pt:8.3f}  {cond}")
    results["v6"] = {"points": pt, "by_condition": cond}
    for name, mp in models.items():
        path = write_preds(f"{name}.csv", base_rows, mp)
        s = score_file(path)
        pt, cond = fmt_score(s)
        print(f"{name:<28} {pt:8.3f}  {cond}", flush=True)
        results[name] = {"points": pt, "by_condition": cond}

    payload = {
        "atoms": atoms,
        "worst_deviation": worst,
        "hmm_eps_tick": best_eps,
        "hmm_eps_win": best_eps_w,
        "hmm_train": [{"eps": e, "tickLL": a, "winLL": b} for e, a, b in hmm_train],
        "train_static_online_ll": train_static,
        "train_best_binary": ranked[0],
        "train_best_binary_gain": train_oll[ranked[0]] - train_static,
        "train_best_pair": pair_ranked[0],
        "train_best_pair_gain": train_oll[pair_ranked[0]] - train_static,
        "val_best_binary": val_ranked[0],
        "val_best_binary_acc": val_hit[val_ranked[0]] / len(vrows),
        "train_best_cell": cranked[0],
        "train_best_cell_gain": cell_oll[cranked[0]] - train_static,
        "val_best_cell": cell_by_val[0],
        "val_best_cell_acc": cell_hit[cell_by_val[0]] / len(vrows),
        "top_val_binary": [
            {
                "feature": n,
                "acc": val_hit[n] / len(vrows),
                "p_true": val_pt[n] / len(vrows),
                "train_gain": train_oll[n] - train_static,
            }
            for n in val_ranked[:20]
        ],
        "top_train_binary": [
            {"feature": n, "train_gain": train_oll[n] - train_static} for n in ranked[:20]
        ],
        "scores": results,
        "notes": {
            "train_best_rule": ranked[0],
            "train_best_pair": pair_ranked[0],
            "val_best_rule_TUNED": val_ranked[0],
            "train_best_cell": cranked[0],
            "val_best_cell_TUNED": cell_by_val[0],
            "train_best_neigh": best_neigh,
        },
    }
    (HERE / "results.json").write_text(json.dumps(payload, indent=2))
    print("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
