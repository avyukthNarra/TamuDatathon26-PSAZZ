#!/usr/bin/env bash
# Score one candidate on kits under kits/K*.
# Usage: bench.sh [--quick|--full] [--score-only] <candidate-name> <build-command-template>
#   --quick      K1, K2, K6, K8 only
#   --full       every kit (default)
#   --score-only append --score-only when the template runs run_pair_mixer.py
#                (skips the entry zip and the fresh-process rerun)
# The template is a shell command. {kit} is the kit directory (parent of
# public/) and {out} is the run directory. At most 5 kits run at once.
# Cross-fitted halves are read from the build's stdout:
#   Cross-fitted validation: tuning X / 40; audit Y / 40
# Writes bench_<name>.csv and bench_<name>.md in this directory.

set -u

TIER="full"
SCORE_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick) TIER="quick"; shift ;;
    --full) TIER="full"; shift ;;
    --score-only) SCORE_ONLY=1; shift ;;
    --) shift; break ;;
    -*)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
    *) break ;;
  esac
done

if [ "$#" -ne 2 ]; then
  echo "usage: bench.sh [--quick|--full] [--score-only] <candidate-name> <build-command-template>" >&2
  exit 2
fi

NAME="$1"
TEMPLATE="$2"
WALL_START="$(date +%s)"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNS="$ROOT/runs/$NAME"
mkdir -p "$RUNS"

kits=()
for dir in "$ROOT"/kits/K*/; do
  [ -d "$dir" ] || continue
  base="$(basename "${dir%/}")"
  if [ "$TIER" = "quick" ]; then
    case "$base" in
      K1|K2|K6|K8) ;;
      *) continue ;;
    esac
  fi
  kits+=("${dir%/}")
done
if [ "${#kits[@]}" -eq 0 ]; then
  echo "no kits in $ROOT/kits (tier $TIER)" >&2
  exit 1
fi
echo "tier $TIER  score_only $SCORE_ONLY  kits ${#kits[@]}"

run_kit() {
  local kitdir="$1"
  local kit
  kit="$(basename "$kitdir")"
  if [ -f "$RUNS/$kit.result" ] && awk -F, 'NR==1 && $2=="ok" {ok=1} END{exit !ok}' "$RUNS/$kit.result"; then
    echo "$kit: already ok, skipping"
    return 0
  fi
  local out="$RUNS/$kit"
  local log="$RUNS/$kit.log"
  mkdir -p "$out"
  local qkit qout cmd
  qkit="$(printf '%q' "$kitdir")"
  qout="$(printf '%q' "$out")"
  cmd="$TEMPLATE"
  cmd="${cmd//\{kit\}/$qkit}"
  cmd="${cmd//\{out\}/$qout}"
  if [ "$SCORE_ONLY" = 1 ]; then
    case "$TEMPLATE" in
      *run_pair_mixer.py*) cmd="$cmd --score-only" ;;
      *) echo "$kit: build has no --score-only; running the full command" ;;
    esac
  fi
  local start rc
  start="$(date +%s)"
  bash -lc "$cmd" >"$log" 2>&1
  rc=$?
  python3 - "$log" "$RUNS/$kit.result" "$kit" "$rc" "$(( $(date +%s) - start ))" <<'PY'
import re, sys
log, dest, kit, rc, seconds = sys.argv[1:]
text = open(log, errors="replace").read()
halves = re.search(
    r"Cross-fitted validation: tuning ([0-9.]+) / [0-9.]+; audit ([0-9.]+)",
    text,
)
full = re.search(r"->\s*([0-9.]+)\s*/\s*[0-9.]+\s*points", text)
tuning = halves.group(1) if halves else ""
audit = halves.group(2) if halves else ""
full_points = full.group(1) if full else ""
status = "ok" if rc == "0" and tuning and audit else ("crash" if rc != "0" else "no-crossfit")
if tuning and audit:
    mean = f"{(float(tuning) + float(audit)) / 2:.4f}"
else:
    mean = ""
with open(dest, "w", encoding="utf-8") as handle:
    handle.write(",".join([kit, status, rc, tuning, audit, mean, full_points, seconds]) + "\n")
print(f"{kit}: {status} exit {rc} tuning {tuning or '-'} audit {audit or '-'} ({seconds}s)", flush=True)
PY
}

pids=()
alive() {
  local n=0 pid
  # bash 3.2 + set -u errors on an empty "${pids[@]}".
  if [ "${#pids[@]}" -eq 0 ]; then
    echo 0
    return
  fi
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

for kitdir in "${kits[@]}"; do
  while [ "$(alive)" -ge 5 ]; do
    sleep 0.4
  done
  run_kit "$kitdir" &
  pids+=("$!")
done
wait

python3 - "$NAME" "$ROOT" "$RUNS" "$TIER" "$SCORE_ONLY" "$(( $(date +%s) - WALL_START ))" <<'PY'
import csv, sys
from pathlib import Path
name, root, runs, tier, score_only, wall = sys.argv[1:]
root, runs = Path(root), Path(runs)
rows = []
for path in sorted(runs.glob("K*.result"), key=lambda p: int(p.name[1:].split(".")[0])):
    text = path.read_text(encoding="utf-8").strip()
    kit, status, rc, tuning, audit, mean, full, seconds = text.split(",")
    rows.append({
        "kit": kit, "status": status, "exit_code": rc,
        "tuning": tuning, "audit": audit, "crossfit_mean": mean,
        "full_points": full, "seconds": seconds,
    })
header = ["kit", "status", "exit_code", "tuning", "audit", "crossfit_mean", "full_points", "seconds"]
csv_path = root / f"bench_{name}.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

def num(row, key):
    return float(row[key]) if row[key] else None

scored = [row for row in rows if num(row, "crossfit_mean") is not None]
crashed = [row["kit"] for row in rows if row["status"] == "crash"]
missing = [row["kit"] for row in rows if row["status"] == "no-crossfit"]
mean = sum(num(row, "crossfit_mean") for row in scored) / len(scored) if scored else float("nan")
worst = min(scored, key=lambda row: num(row, "crossfit_mean")) if scored else None

lines = [
    f"# {name}",
    "",
    f"Tier `{tier}`. Score-only `{score_only}` (1 skips zip and the fresh rerun when the build accepts `--score-only`).",
    f"Wall clock {wall}s. Per-kit `seconds` is that kit's own build, and kits run up to 5 at a time.",
    "",
    "Cross-fitted validation points out of 40. `crossfit_mean` is the average of the two halves.",
    "",
    "| kit | status | tuning | audit | mean | full | seconds |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['kit']} | {row['status']} | {row['tuning'] or ''} | {row['audit'] or ''} | "
        f"{row['crossfit_mean'] or ''} | {row['full_points'] or ''} | {row['seconds']} |"
    )
lines.append("")
if scored:
    kit_seconds = sum(int(row["seconds"]) for row in rows if row["seconds"])
    lines.append(f"Mean over scored kits: {mean:.2f} / 40 ({len(scored)} kits).")
    lines.append(f"Timing: wall {wall}s, sum of per-kit builds {kit_seconds}s.")
    lines.append(
        f"Worst kit: {worst['kit']} at {num(worst, 'crossfit_mean'):.2f} "
        f"(tuning {worst['tuning']}, audit {worst['audit']})."
    )
else:
    lines.append("No kit printed both cross-fitted halves.")
if crashed:
    lines.append("Crashed: " + ", ".join(crashed) + ".")
if missing:
    lines.append("Finished without a cross-fit line: " + ", ".join(missing) + ".")
lines.append("")
md = "\n".join(lines)
(root / f"bench_{name}.md").write_text(md, encoding="utf-8")
print(md)
PY
