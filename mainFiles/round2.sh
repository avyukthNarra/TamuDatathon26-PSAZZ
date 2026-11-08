#!/usr/bin/env bash
# Refit this round's model from a newly released public/ directory.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_SOURCE="$ROOT/v7c-mainFiles"
FALLBACK_SOURCE="$ROOT/tooling/fallback-v6"
PYTHON="${PYTHON:-python3}"
BUILD_SECONDS="${ROUND2_BUILD_SECONDS:-360}"
FALLBACK_SECONDS="${ROUND2_FALLBACK_SECONDS:-180}"

usage() {
  cat >&2 <<'EOF'
Usage: ./round2.sh /path/to/new-kit/public [--source /path/to/candidate/mainFiles]

Copies the source to a fresh isolated workspace, runs its round-aware build,
checks the generated test CSV against the supplied kit, and publishes only the
successful pair of files in ./SUBMIT_R2/.
EOF
}
fail() { echo "round2: ERROR: $*" >&2; exit 2; }

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage; exit 2; }
PUBLIC="$(cd -- "$1" 2>/dev/null && pwd -P)" || fail "public folder not found: $1"
shift
SOURCE="$DEFAULT_SOURCE"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) [[ $# -ge 2 ]] || fail "--source needs a path"; SOURCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done
[[ -d "$PUBLIC/train" && -d "$PUBLIC/validation" && -d "$PUBLIC/test" ]] || fail "expected train/, validation/, and test/ under $PUBLIC"
[[ -f "$PUBLIC/metadata.json" ]] || fail "metadata.json missing from $PUBLIC"
[[ -f "$FALLBACK_SOURCE/run_pair_mixer.py" && -f "$FALLBACK_SOURCE/check_submission.py" ]] || fail "fallback v6 source unavailable: $FALLBACK_SOURCE"
command -v "$PYTHON" >/dev/null || fail "Python executable not found: $PYTHON"

WORK_ROOT="$ROOT/.round2-work"
mkdir -p "$WORK_ROOT"
export TMPDIR="$WORK_ROOT/tmp"
mkdir -p "$TMPDIR"
WORK="$(mktemp -d "$WORK_ROOT/run.XXXXXXXX")"
PUBLISHED=0
if [[ -e "$ROOT/SUBMIT_R2" ]]; then
  mv "$ROOT/SUBMIT_R2" "$WORK/previous-SUBMIT_R2"
fi
cleanup() {
  status=$?
  if [[ $PUBLISHED -eq 0 && -d "$WORK" ]]; then
    echo "round2: failed run preserved for inspection at $WORK" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1

run_timed() {
  local seconds="$1"; shift
  "$PYTHON" -B -c '
import os, signal, subprocess, sys
seconds = float(sys.argv[1]); command = sys.argv[2:]
proc = subprocess.Popen(command, start_new_session=True)
try:
    raise SystemExit(proc.wait(timeout=seconds))
except subprocess.TimeoutExpired:
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait()
    print(f"round2: timed out after {seconds:g}s: {command[0]}", file=sys.stderr)
    raise SystemExit(124)
' "$seconds" "$@"
}

copy_source() {
  local from="$1" to="$2"
  mkdir -p "$to" || return 1
  # Explicitly allowlist runtime code and templates. Never copy the bundled
  # main-round public data, run outputs, symlinks, or repository metadata.
  local name
  for name in run_pair_mixer.py check_submission.py slot_forecasting entry_template; do
    [[ ! -L "$from/$name" && -e "$from/$name" ]] || { echo "round2: missing/non-regular source item: $from/$name" >&2; return 1; }
    if [[ -d "$from/$name" ]]; then
      mkdir -p "$to/$name" || return 1
      (cd "$from/$name" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0) |
        while IFS= read -r -d '' rel; do
          [[ ! -L "$from/$name/${rel#./}" ]] || { echo "round2: refusing source symlink: $from/$name/$rel" >&2; return 1; }
          mkdir -p "$to/$name/$(dirname -- "$rel")" || return 1
          cp "$from/$name/$rel" "$to/$name/$rel" || return 1
        done || return 1
    else
      cp "$from/$name" "$to/$name" || return 1
    fi
  done
  # The stock runner records a git SHA for descriptive metadata. Replace that
  # optional lookup in the disposable copy so the build does not invoke git.
  "$PYTHON" -B - "$to/run_pair_mixer.py" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
s = s.replace('commit = "local build"', 'commit = "isolated"')
s, count = re.subn(r'commit = subprocess\.run\(\["git", "-C", str\(HERE\), "rev-parse", "--short", "HEAD"\],\s*\n\s*capture_output=True, text=True\)\.stdout\.strip\(\) or "unknown"', 'commit = "isolated"', s)
if "rev-parse" in s:
    raise SystemExit("could not safely neutralize optional git metadata lookup")
p.write_text(s, encoding="utf-8")
PY
  [[ $? -eq 0 ]] || return 1
}

run_candidate() {
  local label="$1" source="$2"
  local base="$WORK/$label"
  local out="$base/output"
  local budget="$BUILD_SECONDS"
  [[ "$label" != "fallback-v6" ]] || budget="$FALLBACK_SECONDS"
  mkdir -p "$base/source" || return 1
  copy_source "$source" "$base/source" || return 1
  echo "round2: building with $label source: $source"
  if (cd "$base/source" && run_timed "$budget" "$PYTHON" -B run_pair_mixer.py \
      --data "$PUBLIC" --out "$out" --cv-val) >"$base/build.log" 2>&1; then
    :
  else
    local rc=$?
    cat "$base/build.log" >&2
    echo "round2: $label build failed or timed out (exit $rc)" >&2
    return 1
  fi
  cat "$base/build.log"
  [[ -s "$out/submission.csv" && -s "$out/entry.zip" ]] || { echo "round2: $label did not produce both required artifacts" >&2; return 1; }
  if [[ -s "$out/size_rule.json" ]]; then
    "$PYTHON" -B -c 'import json,sys; d=json.load(open(sys.argv[1])); print("round2: learned size lags:", d.get("lags")); print("round2: learned atoms:", json.dumps(d.get("atoms",{}), sort_keys=True))' "$out/size_rule.json" || return 1
  fi
  if ! run_timed 45 "$PYTHON" -B "$base/source/check_submission.py" \
      "$out/submission.csv" --data "$PUBLIC" --split test >"$base/check.log" 2>&1; then
    cat "$base/check.log" >&2
    echo "round2: $label test CSV check failed" >&2
    return 1
  fi
  cat "$base/check.log"
  echo "round2: $label build and public-kit checker passed"
  SELECTED="$label"
  SELECTED_OUT="$out"
}

SELECTED=""
SELECTED_OUT=""
if ! run_candidate candidate "$SOURCE"; then
  echo "round2: candidate unavailable; rebuilding from plain v6 fallback" >&2
  run_candidate fallback-v6 "$FALLBACK_SOURCE" || fail "candidate and plain v6 fallback both failed; no submission published"
fi

# Expose only a complete, checker-approved artifact pair. Build in the isolated
# workspace first, so an interrupted run cannot leave stale SUBMIT_R2 files.
PUBLISH_DIR="$WORK/publish"
mkdir -p "$PUBLISH_DIR"
cp "$SELECTED_OUT/submission.csv" "$PUBLISH_DIR/submission.csv"
cp "$SELECTED_OUT/entry.zip" "$PUBLISH_DIR/entry.zip"
[[ -s "$PUBLISH_DIR/submission.csv" && -s "$PUBLISH_DIR/entry.zip" ]] || fail "staged artifacts are missing or empty"
mv "$PUBLISH_DIR" "$ROOT/SUBMIT_R2"
PUBLISHED=1
echo "round2: READY: $ROOT/SUBMIT_R2/{submission.csv,entry.zip} (source: $SELECTED)"
echo "round2: build log: $WORK/$SELECTED/build.log"
