"""Round-adaptive member gate for the adapted blend (idea from Pranav's v18 add-on gates).

    python3 r2/gate_members.py --entry adapted_entry_v17g --kit /path/to/kit/public

Runs the entry's members on the kit's VALIDATION split with --protocol crossfit (each half
predicted with priors from the other half), then compares the equal log-odds blend of all
members with every blend that drops one win-only member. A member is dropped only when the
blend without it is better on BOTH validation halves (the kit's episode_folds) and by at
least --margin points overall; otherwise every member stays. The size member (hooks.SIZE_MEMBER)
is never dropped, and at least two members are always kept.

Also copies the kit's whole validation/ folder to <entry>/bundled/validation/, which
predict.py (adapted_entry_v17g_rules) reads when /data has no validation answers or no validation/
folder at all (the eval suite's organizer-style kit-test mounts only train/ and test/), so the
forecast is byte-identical either way (checked offline, --network none, read-only, 2 CPUs).

Writes <entry>/gate.json ({"win_members": [...], ...}); predict.py reads it, so the decision is
made once at build time and the graded rerun (with or without validation targets) uses it.
Without gate.json the entry runs every member, exactly as before.

Checked (cross-fitted, adapted_entry_v17g): main and three simulated round-two kits keep all
three members (no change); with one member replaced by the pooled reference, the gate drops it:
main 55.04 -> 58.00, sim "four" 35.63 -> 36.92 (exact) and 35.70 -> 37.07 (sysid).
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mainFiles"))
from slot_forecasting.contest import contest_score  # noqa: E402
from slot_forecasting.experiments import episode_folds  # noqa: E402
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets  # noqa: E402
from slot_forecasting.rounds import VERSION, round_metadata  # noqa: E402


def logit(p):
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def blend(members, names, size):
    out = {}
    for qid, s in size.items():
        z = sum(logit(1.0 - members[n][qid][0]) for n in names) / len(names)
        win = 1.0 / (1.0 + math.exp(-z))
        total = math.fsum(s[1:])
        split = [x / total for x in s[1:]] if total > 0 else [0.25] * 4
        raw = [max(v, 1e-4) for v in [1.0 - win] + [win * x for x in split]]
        scale = math.fsum(raw)
        out[qid] = tuple(v / scale for v in raw)
    return out


class Scorer:
    def __init__(self, kit):
        v = kit / "validation"
        self.max_points = round_metadata(kit)["max_points"]
        self.queries = load_inputs(v)[0]
        self.targets = read_targets(v / "targets.csv")
        self.groups = {r["query_id"]: r["condition"] for r in read_rows(v / "conditions.csv", ("query_id", "condition"))}
        self.reference = read_submission(v / "reference.csv", {q.query_id for q in self.queries})
        self.halves = episode_folds(self.targets, self.groups)

    def points(self, pred, episodes=None):
        ts = [t for t in self.targets if episodes is None or t.episode_id in episodes]
        ids = {t.query_id for t in ts}
        return contest_score({k: pred[k] for k in ids}, ts, {k: self.groups[k] for k in ids},
                             {k: self.reference[k] for k in ids}, max_points=self.max_points,
                             version=VERSION)["technical_points"]

    def all(self, pred):
        return (self.points(pred),) + tuple(self.points(pred, h) for h in self.halves)


def decide(members, size_member, scorer, margin, log=print):
    names = list(members)
    size = members[size_member]
    base = scorer.all(blend(members, names, size))
    log(f"all members {names}: {base[0]:.3f} (halves {base[1]:.2f} / {base[2]:.2f})")
    choice, best = names, base
    if len(names) > 2:
        for drop in names:
            if drop == size_member:
                continue
            keep = [n for n in names if n != drop]
            s = scorer.all(blend(members, keep, size))
            wins = s[1] > base[1] and s[2] > base[2] and s[0] >= base[0] + margin
            log(f"  without {drop}: {s[0]:.3f} (halves {s[1]:.2f} / {s[2]:.2f}){'  <- clear win' if wins else ''}")
            if wins and s[0] > best[0]:
                choice, best = keep, s
    return choice, base, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", required=True, type=Path)
    ap.add_argument("--kit", required=True, type=Path, help="the kit's public/ folder (with validation answers)")
    ap.add_argument("--members", type=Path, default=None,
                    help="folder with crossfit validation member CSVs already made (skips the member runs)")
    ap.add_argument("--margin", type=float, default=0.1, help="minimum overall gain in points to drop a member")
    args = ap.parse_args()
    entry, kit = args.entry.resolve(), args.kit.resolve()
    sys.path.insert(0, str(entry))
    from members.hooks import REGISTRY, SIZE_MEMBER  # noqa: E402

    hold = tempfile.TemporaryDirectory(prefix="gate-")
    member_dir = args.members.resolve() if args.members else Path(hold.name)
    if not args.members:
        subprocess.run([sys.executable, str(entry / "predict.py"), "--data", str(kit), "--split", "validation",
                        "--protocol", "crossfit", "--out", str(member_dir / "_blend.csv"),
                        "--member-dir", str(member_dir), "--all-members"], cwd=entry, check=True)
    scorer = Scorer(kit)
    ids = {q.query_id for q in scorer.queries}
    members = {spec["name"]: read_submission(member_dir / f"{spec['name']}.csv", ids) for spec in REGISTRY}
    choice, base, best = decide(members, SIZE_MEMBER, scorer, args.margin)
    gate = {"win_members": choice, "size_member": SIZE_MEMBER, "margin": args.margin,
            "all_members_points": round(base[0], 6), "chosen_points": round(best[0], 6),
            "kit_round": round_metadata(kit)["round"]}
    (entry / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    # bundle this round's validation folder (public kit data) so predict.py gives the same
    # forecast when the rerun mounts /data without its answers or without validation/ at all
    # (members use the validation answers as priors)
    bundled = entry / "bundled"
    shutil.rmtree(bundled, ignore_errors=True)
    bundled.mkdir()
    shutil.copytree(kit / "validation", bundled / "validation")
    print(f"gate: keep {choice} ({best[0]:.3f} vs all {base[0]:.3f}); wrote {entry / 'gate.json'}")


if __name__ == "__main__":
    main()
