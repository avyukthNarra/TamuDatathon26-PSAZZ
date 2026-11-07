#!/usr/bin/env bash
# Roll CHANGELOG.md's [Unreleased] section into a new version section and commit it.
# Run on a release/vX.Y.Z branch (see AGENTS.md §7). Does NOT push or tag.
#
# Usage:
#   scripts/release.sh            # version from scripts/next-version.sh
#   scripts/release.sh 1.0.0      # explicit version (e.g. to force v1.0.0 at demo freeze)
#   scripts/release.sh --dry-run  # print the new CHANGELOG.md instead of writing/committing
set -euo pipefail

DRY=0; VERSION=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) VERSION="$arg" ;;amke
  esac
done

ROOT=$(git rev-parse --show-toplevel)
CHANGELOG="$ROOT/CHANGELOG.md"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

[ -f "$CHANGELOG" ] || { echo "CHANGELOG.md not found at repo root" >&2; exit 1; }

if [ -z "$VERSION" ]; then
  VERSION=$("$SCRIPT_DIR/next-version.sh" --quiet) || { echo "No release needed (no feat/fix/perf/breaking commits since last tag)." >&2; exit 3; }
fi
VERSION=${VERSION#v}
echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' || { echo "invalid semver: $VERSION" >&2; exit 2; }

if [ "$DRY" = 0 ]; then
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  case "$BRANCH" in
    main|master) echo "Refusing to commit on $BRANCH. Create release/v$VERSION first." >&2; exit 1 ;;
  esac
  git diff --quiet && git diff --cached --quiet || { echo "Working tree not clean." >&2; exit 1; }
  grep -q "^## \[$VERSION\]" "$CHANGELOG" && { echo "CHANGELOG already has [$VERSION]." >&2; exit 1; }
fi

TODAY=$(date +%Y-%m-%d)
TMP=$(mktemp)

awk -v ver="$VERSION" -v today="$TODAY" '
  function flush_section() {
    sub(/\n+$/, "\n", body)
    if (sec != "" && has_items) { out = out "\n" sec "\n" body; any = 1 }
    sec = ""; body = ""; has_items = 0
  }
  function emit_release() {
    flush_section()
    print "## [Unreleased]\n\n### Added\n\n### Changed\n\n### Deprecated\n\n### Removed\n\n### Fixed\n\n### Security\n"
    print "## [" ver "] - " today
    printf "%s", out
    print ""
    done = 1; inrel = 0
  }
  /^## \[Unreleased\]/ { inrel = 1; next }
  inrel && /^## \[/ { emit_release() }
  inrel {
    if ($0 ~ /^### /) { flush_section(); sec = $0; next }
    if ($0 ~ /^[[:space:]]*[-*] /) has_items = 1
    if ($0 ~ /^[[:space:]]*$/ && body == "") next
    if (sec != "") body = body $0 "\n"
    next
  }
  { print }
  END {
    if (inrel) emit_release()
    if (!any) exit 4
  }
' "$CHANGELOG" > "$TMP" || {
  rc=$?; rm -f "$TMP"
  if [ $rc -eq 4 ]; then echo "[Unreleased] has no entries. Add changelog lines before releasing." >&2; else echo "awk failed ($rc)" >&2; fi
  exit 1
}

if [ "$DRY" = 1 ]; then cat "$TMP"; rm -f "$TMP"; exit 0; fi

mv "$TMP" "$CHANGELOG"
git add "$CHANGELOG"
git commit -q -m "chore(release): v$VERSION"
echo "Committed CHANGELOG for v$VERSION on $(git rev-parse --abbrev-ref HEAD)."
echo "Next: git push -u origin HEAD && open a PR titled 'chore(release): v$VERSION'."
echo "After merge: git checkout main && git pull --ff-only && git tag -a v$VERSION -m v$VERSION && git push origin v$VERSION"
