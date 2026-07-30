#!/usr/bin/env bash
#
# check_doc_lifecycle.sh -- pre-commit NUDGE (never blocks) for the
# living-document discipline: CHANGELOG.md gets an entry in the same
# commit as notable work, and report-shaped docs land in an archive
# location instead of the top-level docs/ dir.
#
# Language-agnostic: matches by extension/dir pattern, not by any
# particular stack. Fast, pure git/grep -- no interpreter startup.
# Inspects STAGED files only. Both checks WARN and exit 0 -- notability
# and "is it done" are judgment calls a hook cannot make on its own.

set -uo pipefail

staged_any="$(git diff --cached --name-only || true)"
added_files="$(git diff --cached --name-only --diff-filter=A || true)"

[ -z "$staged_any" ] && exit 0

# ---------------------------------------------------------------------------
# Check A -- CHANGELOG / close-out discipline.
# ---------------------------------------------------------------------------
# Common migration/schema-change directory shapes across stacks.
mig="$(printf '%s\n' "$staged_any" | grep -E \
  '(^|/)(migrations?|alembic/versions|db/migrate|prisma/migrations)/' \
  || true)"

# New source files, common extensions, excluding anything that reads as
# a test/spec/fixture/mock/generated file or lives in a vendored/build dir.
newsrc="$(printf '%s\n' "$added_files" \
  | grep -E '\.(py|js|jsx|ts|tsx|go|rs|java|kt|rb|c|cc|cpp|h|hpp|swift|cs|php|scala|ex|exs)$' \
  | grep -viE '(^|/)(test|tests|spec|specs|__tests__|__mocks__|fixtures?|mocks?)(/|_|\.)' \
  | grep -viE '(^|/)(node_modules|vendor|dist|build|out|target|\.venv|venv)/' \
  || true)"

notable="$(printf '%s\n%s\n' "$mig" "$newsrc" | grep -v '^$' || true)"

if [ -n "$notable" ] && ! printf '%s\n' "$staged_any" | grep -qx 'CHANGELOG.md'; then
  echo "⚠️  Doc-lifecycle WARNING (close-out discipline):"
  printf '%s\n' "$notable" | while IFS= read -r f; do [ -n "$f" ] && echo "   • $f"; done
  echo "   A migration or new source file is staged but CHANGELOG.md is"
  echo "   not. Notable changes get a CHANGELOG entry in the same landing"
  echo "   commit; on close-out also flip GAMEPLAN.md status."
  echo "   (Warning only -- commit proceeds; skip only for the trivial.)"
  echo ""
fi

# ---------------------------------------------------------------------------
# Check B -- report-shaped docs belong in an archive location, not top-level.
# ---------------------------------------------------------------------------
report_docs=""
if [ -n "$added_files" ]; then
  report_docs="$(printf '%s\n' "$added_files" \
    | grep -E '^docs/[^/]+\.md$' \
    | grep -iE '(report|summary|snapshot|handoff|state-of-play|status|post-?mortem)' \
    || true)"
fi

if [ -n "$report_docs" ]; then
  echo "⚠️  Doc-lifecycle WARNING (location signals status):"
  printf '%s\n' "$report_docs" | while IFS= read -r f; do [ -n "$f" ] && echo "   • $f"; done
  echo "   New docs/ file reads as a report/snapshot -- frozen history"
  echo "   belongs in docs/archive/, and current status belongs in"
  echo "   GAMEPLAN.md. (Warning only.)"
  echo ""
fi

exit 0
