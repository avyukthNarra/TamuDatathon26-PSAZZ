#!/usr/bin/env python3
"""One-command round-two build and ranking.

    python3 r2/run_round2.py --kit /path/to/kit/public --out /path/to/OUT [--extra NAME=ENTRY_DIR ...]

Builds (max 3 at once) v17-guards, v7c (origin/avyukth/v7c-size-flip), plain v6 (its tooling/fallback-v6),
mixer6, and any --extra entries; scores validation (cross-fitted if present, else in-sample) with the
official formula, runs paired comparisons vs v17-guards, checks each test CSV, and writes OUT/RANKING.md.
Sources are read-only: v17g/mixer6 are copied, v7c/v6 are extracted with git archive. Nothing is uploaded.
"""
import argparse, csv, hashlib, io, itertools, json, shutil, subprocess, sys, tarfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "orch"))
from common import PCOLS, load, points, folds  # noqa: E402
import paired as pairedmod  # noqa: E402

TM = Path("/Users/saisreed/Desktop/Coding/TamuDatathon26-PSAZZ/worktrees/_tm")
V17G = TM / "v17g/mainFiles"
MIXER6 = TM / "mixer6/mainFiles"
V7C_REF = "origin/avyukth/v7c-size-flip"
V6_SUB = "mainFiles/round2/tooling/fallback-v6"
BLEND_DIR = Path(__file__).resolve().parents[1] / "adapted_entry_v17g"  # default pick: adapted blend, v17-guards first (PR #22)
BLEND_V13_DIR = Path(__file__).resolve().parents[1] / "adapted_entry"   # same blend with v13 first, kept as a challenger
DEFAULT = "blend"  # upload first if its test CSV is valid
FALLBACK = ("blend", "v17-guards", "v7c", "v6", "mixer6")  # zhill v18 is not built here; add it with --extra if wanted
MARGIN = 0.1  # a challenger must beat the default on both halves and on the mean by more than this
SAFE = "v17-guards"
ORIG_PAIRS = [(1, 3), (2, 5), (4, 7)]
BAR = "!" * 78


def say(msg=""):
    print(msg, flush=True)


def check_kit(kit):
    for split in ("train", "validation", "test"):
        if not (kit / split).is_dir():
            raise SystemExit(f"ERROR: {kit} has no {split}/ folder (pass the public/ folder itself)")
    if not (kit / "metadata.json").exists():
        raise SystemExit(f"ERROR: {kit}/metadata.json missing")
    for f in ("targets.csv", "conditions.csv", "reference.csv"):
        if not (kit / "validation" / f).exists():
            raise SystemExit(f"ERROR: validation/{f} missing from {kit}")
    meta = json.loads((kit / "metadata.json").read_text())
    rnd = meta.get("round")
    warn = []
    if rnd != "adaptation":
        warn.append(f"metadata round is {rnd!r}, not 'adaptation'")
    if rnd == "main":
        warn.append("THIS IS THE MAIN-ROUND KIT")
    if warn:
        say(BAR)
        for w in warn:
            say(f"!!! WARNING: {w}")
        say(BAR)
    say(f"kit: {kit} round={rnd} season={meta.get('season_id')} max_points={meta.get('max_points', 40)}")
    return meta, warn


def structure_screen(kit, out):
    """Walsh lag-subset screen (orch/r2screen) plus the pair census (orch/priors logic) on this kit."""
    r = subprocess.run([sys.executable, "-B", str(REPO / "orch/r2screen/walsh_screen.py"), str(kit)],
                       capture_output=True, text=True)
    (out / "screen_walsh.txt").write_text(r.stdout + r.stderr)
    lines = r.stdout.splitlines()
    if "### Overall" in r.stdout:
        i = next(k for k, l in enumerate(lines) if l.startswith("### Overall"))
        say("Walsh screen, overall lag subsets (full output in screen_walsh.txt):")
        for l in lines[i + 2:i + 10]:
            say("  " + l)
    say(pair_census(kit, out))


def _ll2(w, n):
    p = np.clip(w / np.maximum(n, 1e-9), 1e-6, 1 - 1e-6)
    return (w * np.log(p) + (n - w) * np.log(1 - p)).sum()


def pair_census(kit, out):
    pairs = list(itertools.combinations(range(1, 9), 2))
    cnt, per_split = Counter(), {}
    for split in ("train", "validation", "test"):
        d = load(split, pub=kit)
        conf = tot = 0
        for q in d["Q"]:
            h = d["H"][q["machine_id"]]
            if len([t for t in h if t < 96]) < 90:
                continue
            tot += 1
            W = {t: int(h[t] > 0) for t in range(96) if t in h}
            res = []
            for a, b in pairs:
                w = np.zeros(2); n = np.zeros(2)
                for t in range(8, 96):
                    if t in W and (t - a) in W and (t - b) in W:
                        k = int(W[t - a] == W[t - b]); w[k] += W[t]; n[k] += 1
                res.append((_ll2(w, n), (a, b)))
            res.sort(key=lambda r: -r[0])
            if res[0][0] - res[1][0] >= 3:
                conf += 1; cnt[res[0][1]] += 1
        per_split[split] = (tot, conf)
    total = sum(cnt.values())
    orig = sum(cnt[p] for p in ORIG_PAIRS)
    top = ", ".join(f"({a},{b}):{c}" for (a, b), c in cnt.most_common(6))
    share = orig / total if total else float("nan")
    verdict = "structure MOVED from (1,3),(2,5),(4,7)" if share < 0.5 else "structure still (1,3),(2,5),(4,7)-like"
    (out / "pair_census.json").write_text(json.dumps({"per_split": per_split,
        "pairs": {f"{a},{b}": c for (a, b), c in cnt.most_common()}, "orig_share": share}, indent=1))
    return (f"Pair census (confident machines per split {per_split}): top pairs {top}; "
            f"original-pair share {share:.2f} -> {verdict}")


def snapshot(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("public", "runs", "__pycache__", "*.pyc", ".git"))
    return dst


def git_extract(ref, subpath, dst, drop=()):
    if subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q", ref], capture_output=True).returncode:
        raise RuntimeError(f"git ref {ref} not found; run git fetch origin")
    data = subprocess.run(["git", "-C", str(REPO), "archive", ref, subpath], capture_output=True, check=True).stdout
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dst, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    src = dst / subpath
    for d in drop:
        shutil.rmtree(src / d, ignore_errors=True)
    return src


def timed_run(cmd, cwd, log, timeout=3600):
    t0 = time.monotonic()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return rc, time.monotonic() - t0


def build_job(name, src, kit, out, script):
    out.mkdir(parents=True, exist_ok=True)
    if script is None:  # --extra entry: predict.py writes the test CSV directly
        cmd = [sys.executable, "-B", str(src / "predict.py"), "--data", str(kit), "--out", str(out / "submission.csv")]
        rc, secs = timed_run(cmd, src, out / "build.log")
        if rc == 0:  # optional: validation predictions if the entry supports --split (cross-fit if --protocol)
            text = (src / "predict.py").read_text()
            vcmd = [sys.executable, "-B", str(src / "predict.py"), "--data", str(kit), "--split", "validation"]
            vname = "validation_predictions.csv"
            if "--protocol" in text:
                vcmd += ["--protocol", "crossfit"]; vname = "crossfit_validation_predictions.csv"
            if "--split" in text:
                timed_run(vcmd + ["--out", str(out / vname)], src, out / "validation.log")
    else:
        cmd = [sys.executable, "-B", script, "--data", str(kit), "--out", str(out), "--cv-val"]
        rc, secs = timed_run(cmd, src, out / "build.log")
    return {"name": name, "rc": rc, "secs": secs, "out": out}


def load_preds(path):
    return {r["query_id"]: np.array([float(r[c]) for c in PCOLS]) for r in csv.DictReader(open(path, newline=""))}


def score_val(pred, data, mp):
    total, ratios = points(pred, data, max_points=mp)
    ep = {q["query_id"]: q["episode_id"] for q in data["Q"]}
    A, B = folds(data)
    half = lambda S: points(pred, data, qids=[q for q in data["targets"] if ep[q] in S], max_points=mp)[0]
    return {"points": total, "halfA": half(A), "halfB": half(B), "ratios": ratios}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_test(checker, csv_path, kit):
    r = subprocess.run([sys.executable, "-B", str(checker), str(csv_path), "--data", str(kit), "--split", "test"],
                       capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", required=True, type=Path, help="the kit's public/ folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=ENTRY_DIR",
                    help="entry folder with predict.py (--data --out [--split]); repeatable")
    ap.add_argument("--blend", default=str(BLEND_DIR),
                    help="blend entry folder, added as candidate 'blend' unless empty or already given via --extra")
    ap.add_argument("--v7c-ref", default=V7C_REF)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    t_all = time.monotonic()
    kit = args.kit.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    meta, warnings = check_kit(kit)
    mp = meta.get("max_points", 40)

    # 1. structure screen
    structure_screen(kit, out)

    # 2. sources (read-only: copies or git archive extracts under OUT/_src)
    src_dir = out / "_src"
    jobs, unavailable = [], {}
    for name, fn in [(SAFE, lambda: snapshot(V17G, src_dir / "v17g")),
                     ("v7c", lambda: git_extract(args.v7c_ref, "mainFiles", src_dir / "v7c", drop=("public", "runs"))),
                     ("v6", lambda: git_extract(args.v7c_ref, V6_SUB, src_dir / "v6")),
                     ("mixer6", lambda: snapshot(MIXER6, src_dir / "mixer6"))]:
        try:
            src = fn()
        except Exception as e:  # noqa: BLE001 - record and continue with the others
            unavailable[name] = str(e)
            say(f"candidate {name} unavailable: {e}")
            continue
        script = "run_strongest.py" if name == SAFE else "run_pair_mixer.py"
        jobs.append((name, src, script))
    extras = list(args.extra)
    if args.blend and Path(args.blend).is_dir() and not any(e.startswith("blend=") for e in extras):
        extras.insert(0, f"blend={args.blend}")
    if BLEND_V13_DIR.is_dir() and not any(e.startswith("blend_v13=") for e in extras):
        extras.append(f"blend_v13={BLEND_V13_DIR}")
    for spec in extras:
        name, _, d = spec.partition("=")
        if not name or not d:
            raise SystemExit(f"--extra needs NAME=ENTRY_DIR, got {spec!r}")
        jobs.append((name, Path(d).resolve(), None))

    # 3. builds, max --jobs at once
    say(f"building {len(jobs)} candidates, {args.jobs} at a time")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {n: ex.submit(build_job, n, s, kit, out / n, sc) for n, s, sc in jobs}
        results = {n: f.result() for n, f in futs.items()}

    # 4. score, check, compare
    rows, preds = [], {}
    for name, _, _ in jobs:
        r = results[name]
        row = {"name": name, "secs": r["secs"], "status": "ok" if r["rc"] == 0 else f"build rc={r['rc']}",
               "points": None, "halfA": None, "halfB": None, "ratios": {}, "scored": "-", "insample": None,
               "test_ok": False, "test_msg": "", "sha": "", "delta": None, "ci": None, "pgain": None}
        cf, vp = r["out"] / "crossfit_validation_predictions.csv", r["out"] / "validation_predictions.csv"
        if cf.exists() or vp.exists():
            path = cf if cf.exists() else vp
            row["scored"] = path.name
            pred = load_preds(path)
            preds[name] = pred
            if len(pred) == len(load("validation", pub=kit)["targets"]):
                s = score_val(pred, load("validation", pub=kit), mp)
                row.update(points=s["points"], halfA=s["halfA"], halfB=s["halfB"], ratios=s["ratios"])
            else:
                row["status"] += "; validation predictions incomplete"
        vin = r["out"] / "validation_predictions.csv"  # in-sample view, always shown when present
        if vin.exists() and len(load_preds(vin)) == len(load("validation", pub=kit)["targets"]):
            row["insample"] = score_val(load_preds(vin), load("validation", pub=kit), mp)["points"]
        test = r["out"] / "submission.csv"
        if test.exists():
            own = src_dir / name / "check_submission.py"  # each candidate's own checker, else v17g's
            checker = own if own.exists() else V17G / "check_submission.py"
            row["test_ok"], row["test_msg"] = check_test(checker, test, kit)
            row["sha"] = sha(test)[:16]
        elif r["rc"] == 0:
            row["status"] += "; no submission.csv"
        rows.append(row)

    # paired vs the default candidate (the adapted blend); v17-guards if the blend has no validation predictions
    ref = DEFAULT if DEFAULT in preds else SAFE
    base = preds.get(ref)
    for row in rows:
        if row["name"] == ref or base is None or row["name"] not in preds:
            continue
        d, ci, pg = pairedmod.paired(base, preds[row["name"]], kit=kit, max_points=mp)
        row.update(delta=d, ci=ci, pgain=pg)

    order = {n: i for i, n in enumerate(FALLBACK)}
    rows.sort(key=lambda r: (order.get(r["name"], 9), r["name"]))
    safe_row = next((r for r in rows if r["name"] == SAFE), None)
    safe_ok = bool(safe_row and safe_row["test_ok"])
    valid = [r for r in rows if r["test_ok"]]
    # first upload: the first valid candidate in FALLBACK order (default = adapted blend), else any valid one
    safe_pick = next((r for n in FALLBACK for r in rows if r["name"] == n and r["test_ok"]), valid[0] if valid else None)
    def beats(r, d):  # challenger rule: both halves and the mean higher by more than MARGIN
        return all(r[k] is not None and d[k] is not None and r[k] - d[k] > MARGIN for k in ("points", "halfA", "halfB"))
    alt = max((r for r in valid if safe_pick and r["name"] != safe_pick["name"] and r["points"] is not None),
              key=lambda r: r["points"], default=None)
    lines = []
    lines.append("# Round-two ranking\n")
    if warnings:
        lines.append("> **WARNING:** " + "; ".join(warnings) + "\n")
    lines.append(f"Kit: `{kit}` round={meta.get('round')} season={meta.get('season_id')} max_points={mp}\n")
    lines.append("| candidate | scored file | points | in-sample points | half A | half B | "
                 + " | ".join(f"ratio {c}" for c in ("stable", "short_history", "drift", "shared_change"))
                 + f" | paired delta vs {ref} [95% CI] | P(gain>0) | runtime s | test CSV valid | sha256[:16] | status |")
    lines.append("|" + "---|" * 16)
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    for r in rows:
        ratio_cells = [f"{r['ratios'][c]:.3f}" if c in r["ratios"] else "-" for c in ("stable", "short_history", "drift", "shared_change")]
        d = "(reference)" if r["name"] == ref else ("-" if r["delta"] is None else f"{r['delta']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]")
        pg = "-" if r["pgain"] is None else f"{r['pgain']:.2f}"
        lines.append(f"| {r['name']} | {r['scored']} | {fmt(r['points'])} | {fmt(r['insample'])} | {fmt(r['halfA'])} | {fmt(r['halfB'])} | "
                     + " | ".join(ratio_cells) + f" | {d} | {pg} | {r['secs']:.0f} | "
                     + f"{'yes' if r['test_ok'] else 'NO'} | {r['sha'] or '-'} | {r['status']} |")
    for name, why in unavailable.items():
        lines.append(f"| {name} | " + "- | " * 14 + f"unavailable: {why} |")
    lines.append("")
    lines.append("Points use the scored file (cross-fitted when it exists); in-sample points use validation_predictions.csv. "
                 "Halves are the brief's A/B episode split, scored on the scored file. "
                 "Paired deltas use the same bootstrap as orch/paired.py (positive = candidate better).\n")
    lines.append("## Recommendation\n")
    if safe_pick is None:
        lines.append("No candidate produced a valid test CSV. Do not upload; read the build logs.")
    else:
        lines.append(f"1. Upload first: **{safe_pick['name']}** (`{out / safe_pick['name'] / 'submission.csv'}`)"
                     + ("" if safe_pick["name"] == DEFAULT else f" - {DEFAULT} was not valid or not built; next in fallback order."))
        if alt is None:
            lines.append("2. No other valid candidate with a score; nothing further to upload.")
        elif safe_pick["points"] is not None and beats(alt, safe_pick):
            lines.append(f"2. Then upload **{alt['name']}** (`{out / alt['name'] / 'submission.csv'}`): beats {safe_pick['name']} "
                         f"by > {MARGIN} on the mean and both halves ({alt['points']:.2f} vs {safe_pick['points']:.2f}).")
        else:
            lines.append(f"2. Keep {safe_pick['name']}: best challenger {alt['name']} ({alt['points']:.2f}) does not beat it by > {MARGIN} "
                         f"on the mean and both halves.")
    total = time.monotonic() - t_all
    lines.append(f"\nTotal wall time: {total / 60:.1f} min ({total:.0f} s).")
    (out / "RANKING.md").write_text("\n".join(lines) + "\n")
    say("\n".join(lines))
    say(f"\nwrote {out / 'RANKING.md'}")


if __name__ == "__main__":
    main()
