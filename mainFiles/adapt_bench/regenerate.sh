#!/usr/bin/env bash
# Regenerate the ten simulated round-two kits (K1..K10) from the fixed seeds
# in simulator.py. Output is deterministic: the same code gives the same bytes.
# Usage: ./regenerate.sh [OUT_DIR]     (default: kits/ next to this script)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$DIR/kits}"

python3 "$DIR/simulator.py" --out "$OUT"

echo "kit sizes in $OUT:"
for kit in K1 K2 K3 K4 K5 K6 K7 K8 K9 K10; do
  printf '  %-4s %s\n' "$kit" "$(du -sh "$OUT/$kit" | cut -f1)"
done
