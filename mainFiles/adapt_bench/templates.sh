#!/usr/bin/env bash
# Build-command templates for bench.sh. {kit} and {out} are filled in by the runner.
# Run from this folder (mainFiles/adapt_bench/):  bash templates.sh v6|v7c|v15 [--quick|--full] [--score-only]
# Or pass a template yourself:  ./bench.sh <name> '<command>'
#
# Candidate checkouts are found through environment variables. Each default is
# a sibling worktree of this repo (worktrees/<name>/mainFiles); set the
# variable to your own checkout when it lives elsewhere.
#   V7C_DIR   v7c tree with run_pair_mixer.py. Not in this repo, so the default
#             assumes a sibling worktree named v7c.
#   V15_DIR   h12-v15 tree with run_strongest.py. Not in this repo; default
#             assumes research_pf/h12-v15/wt/mainFiles under the Datathon folder.

ROOT="$(cd "$(dirname "$0")" && pwd)"
V7C_DIR="${V7C_DIR:-$ROOT/../../../v7c/mainFiles}"
V15_DIR="${V15_DIR:-$ROOT/../../../../research_pf/h12-v15/wt/mainFiles}"

# v7c, flags at their entry_config defaults (size, size gate, flip, round2 all on).
V7C="cd \"$V7C_DIR\" && python3 run_pair_mixer.py --data {kit} --out {out} --cv-val"

# Plain v6: same tree with every v7c flag off.
V6="cd \"$V7C_DIR\" && python3 run_pair_mixer.py --data {kit} --out {out} --cv-val --no-size --no-size-gate --no-flip --no-round2"

# Plain v6 fallback, if the flag build is not the one you want.
V6_FALLBACK="cd \"$V7C_DIR/round2/tooling/fallback-v6\" && python3 run_pair_mixer.py --data {kit} --out {out} --cv-val"

# v15.
V15="cd \"$V15_DIR\" && python3 run_strongest.py --data {kit} --out {out} --cv-val"

NAME="${1:-}"
shift || true
case "$NAME" in
  v6) exec "$ROOT/bench.sh" "$@" v6 "$V6" ;;
  v6-fallback) exec "$ROOT/bench.sh" "$@" v6-fallback "$V6_FALLBACK" ;;
  v7c) exec "$ROOT/bench.sh" "$@" v7c "$V7C" ;;
  v15) exec "$ROOT/bench.sh" "$@" v15 "$V15" ;;
  *)
    echo "usage: templates.sh v6|v6-fallback|v7c|v15 [--quick|--full] [--score-only]" >&2
    echo "v7c: $V7C"
    echo "v6: $V6"
    echo "v6-fallback: $V6_FALLBACK"
    echo "v15: $V15"
    exit 2
    ;;
esac
