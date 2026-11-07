#!/usr/bin/env bash
# Compute the next semantic version from Conventional Commits since the last vX.Y.Z tag.
# Rules: see AGENTS.md §7. Portable to macOS bash 3.2 + BSD awk.
#
# Usage:
#   scripts/next-version.sh            # human-readable explanation + version
#   scripts/next-version.sh --quiet    # only the version (empty output + exit 3 if no release needed)
#   scripts/next-version.sh --current  # only the current (last tagged) version
set -euo pipefail

QUIET=0; CURRENT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    -q|--quiet) QUIET=1 ;;
    --current) CURRENT_ONLY=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 2; }

LAST_TAG=$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || true)
if [ -n "$LAST_TAG" ]; then
  CURRENT=${LAST_TAG#v}
  CURRENT=${CURRENT%%-*}   # strip any pre-release suffix (v1.0.0-rc.1 -> 1.0.0)
  RANGE="$LAST_TAG..HEAD"
else
  CURRENT="0.0.0"
  RANGE="HEAD"
fi

if [ "$CURRENT_ONLY" = 1 ]; then echo "$CURRENT"; exit 0; fi

IFS=. read -r MAJOR MINOR PATCH <<EOF
$CURRENT
EOF

# No commits yet at all?
if ! git rev-parse --verify -q HEAD >/dev/null; then
  [ "$QUIET" = 1 ] || echo "No commits yet. No release needed."
  exit 3
fi

# Count breaking / feat / fix+perf commits. \036 (RS char) separates commits.
read -r BREAKING FEATS FIXES <<EOF
$(git log --no-merges --format='%x1e%s%n%b' "$RANGE" | awk '
  BEGIN { RS = "\036"; b = 0; f = 0; p = 0 }
  NF == 0 { next }
  {
    n = split($0, lines, "\n"); subj = lines[1]; brk = 0
    if (subj ~ /^[a-zA-Z]+(\([^)]*\))?!:/) brk = 1
    for (i = 2; i <= n; i++) if (lines[i] ~ /^[[:space:]]*BREAKING[ -]CHANGE:/) brk = 1
    if (brk) b++
    else if (subj ~ /^feat(\([^)]*\))?:/) f++
    else if (subj ~ /^(fix|perf)(\([^)]*\))?:/) p++
  }
  END { print b, f, p }')
EOF

if [ "$BREAKING" -gt 0 ]; then
  if [ "$MAJOR" -eq 0 ]; then BUMP="minor (breaking change, pre-1.0)"; MINOR=$((MINOR+1)); PATCH=0
  else BUMP="major (breaking change)"; MAJOR=$((MAJOR+1)); MINOR=0; PATCH=0; fi
elif [ "$FEATS" -gt 0 ]; then
  BUMP="minor (feat)"; MINOR=$((MINOR+1)); PATCH=0
elif [ "$FIXES" -gt 0 ]; then
  BUMP="patch (fix/perf)"; PATCH=$((PATCH+1))
else
  [ "$QUIET" = 1 ] || echo "Current: ${LAST_TAG:-<no tag>}. No feat/fix/perf/breaking commits since. No release needed."
  exit 3
fi

# First ever release is at least 0.1.0
if [ -z "$LAST_TAG" ] && [ "$MAJOR" -eq 0 ] && [ "$MINOR" -eq 0 ]; then
  MINOR=1; PATCH=0; BUMP="$BUMP → first release starts at 0.1.0"
fi

NEXT="$MAJOR.$MINOR.$PATCH"
if [ "$QUIET" = 1 ]; then
  echo "$NEXT"
else
  echo "Current: ${LAST_TAG:-<no tag> (0.0.0)}"
  echo "Since then: $BREAKING breaking, $FEATS feat, $FIXES fix/perf"
  echo "Bump: $BUMP"
  echo "Next version: $NEXT   (tag: v$NEXT)"
fi
