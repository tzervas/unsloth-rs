#!/bin/bash
# scripts/close-linked-issues.sh — Close issues linked by Fixes/Closes/Resolves.
#
# Policy (see docs/WORKFLOW.md §0):
#   - Feature work merges into **dev** and does **not** close issues.
#   - Issues close only when work reaches **main** (GitHub native auto-close
#     and/or this script for parity / commit-scan).
#   - Epics stay open until a final ship issue with Closes #<epic> lands on main.
#
# Usage:
#   bash scripts/close-linked-issues.sh --pr 68              # only if base is main
#   bash scripts/close-linked-issues.sh --pr 68 --dry-run
#   bash scripts/close-linked-issues.sh --merged-into main --limit 20
#   bash scripts/close-linked-issues.sh --pr 68 --any-base   # escape hatch (rare)
#
# Keywords: fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved #N
# Also: conventional titles feat(#62): in PR title/commits (task issues only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
PR_NUM=""
MERGED_INTO=""
MERGED_SINCE=""
LIMIT=30
COMMENT=1
# Default: only act when the PR base is main
REQUIRE_MAIN=1
MAIN_BRANCH="${RELAY_MAIN_BRANCH:-main}"

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pr) PR_NUM="${2:-}"; shift 2 ;;
        --merged-into) MERGED_INTO="${2:-}"; shift 2 ;;
        --merged-since) MERGED_SINCE="${2:-}"; shift 2 ;;
        --limit) LIMIT="${2:-30}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-comment) COMMENT=0; shift ;;
        --any-base) REQUIRE_MAIN=0; shift ;;
        --main-branch) MAIN_BRANCH="${2:-main}"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *)
            printf 'close-linked-issues.sh: unknown arg: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if ! command -v gh >/dev/null 2>&1; then
    printf 'close-linked-issues.sh: gh CLI required\n' >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf 'close-linked-issues.sh: python3 required\n' >&2
    exit 1
fi

# Reads text on stdin → one issue number per line.
extract_issue_numbers() {
    # Load program via heredoc into a variable so stdin remains the PR text pipe
    # (avoids bash quoting hazards in python -c '...' single quotes).
    local _prog
    _prog="$(cat <<'PY'
import re, sys
text = sys.stdin.read()
kw = r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)"
found = set()
for m in re.finditer(rf"(?is)\b{kw}\b(?:\s*:)?\s*((?:#?\d+(?:\s*[,&]?\s*(?:and\s+)?#?\d+)*))", text):
    # Ignore negated forms: "No Fixes #22", "not close #N", "without Fixes #N"
    prefix = text[max(0, m.start() - 24) : m.start()].lower()
    if re.search(r"(?:no|not|without|dont)\W*$", prefix):
        continue
    for n in re.findall(r"\d+", m.group(1)):
        found.add(int(n))
# Conventional task titles (not "Epic: #N" prose)
for m in re.finditer(r"(?i)\b(?:feat|fix|docs|test|chore|refactor|perf|ci|build|style)\(#(\d+)\)", text):
    found.add(int(m.group(1)))
for n in sorted(found):
    print(n)
PY
)"
    python3 -c "$_prog"
}

# Returns 0 if issue looks like an epic (label or title); we still close if
# the PR explicitly Fixes/Closes it — caller decides what numbers to pass.
issue_is_epic() {
    local issue="$1" labels title
    labels="$(gh issue view "$issue" --json labels --jq '[.labels[].name]|join(",")' 2>/dev/null || true)"
    title="$(gh issue view "$issue" --json title --jq .title 2>/dev/null || true)"
    [[ "$labels" == *epic* ]] && return 0
    [[ "$title" =~ ^[Ee]pic: ]] && return 0
    return 1
}

close_one() {
    local issue="$1" pr="$2" base="$3" url="$4" state
    state="$(gh issue view "$issue" --json state --jq .state 2>/dev/null || echo MISSING)"
    if [[ "$state" == "MISSING" || -z "$state" ]]; then
        printf '  skip  #%s (not found)\n' "$issue"
        return 0
    fi
    if [[ "$state" == "CLOSED" ]]; then
        printf '  skip  #%s (already closed)\n' "$issue"
        return 0
    fi
    if (( DRY_RUN == 1 )); then
        printf '  dry   would close #%s (from PR #%s → %s)\n' "$issue" "$pr" "$base"
        return 0
    fi
    if (( COMMENT == 1 )); then
        gh issue close "$issue" --comment "$(cat <<EOF
Closed via merged PR #${pr} into \`${base}\`.

Issue close policy: only after work reaches **main** (feature merges to \`dev\` leave issues open). See docs/WORKFLOW.md.

${url}
EOF
)" >/dev/null
    else
        gh issue close "$issue" >/dev/null
    fi
    printf '  closed #%s (PR #%s → %s)\n' "$issue" "$pr" "$base"
}

process_pr() {
    local pr="$1" json title body base state merged url text nums commits
    json="$(gh pr view "$pr" --json number,title,body,baseRefName,state,mergedAt,url 2>/dev/null)" || {
        printf 'close-linked-issues.sh: cannot view PR #%s\n' "$pr" >&2
        return 1
    }
    title="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("title") or "")')"
    body="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("body") or "")')"
    base="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("baseRefName") or "")')"
    state="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state") or "")')"
    merged="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mergedAt") or "")')"
    url="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("url") or "")')"

    if (( REQUIRE_MAIN == 1 )) && [[ "$base" != "$MAIN_BRANCH" ]]; then
        printf 'PR #%s base=%s — skip close (policy: only base=%s; feature work on dev leaves issues open)\n' \
            "$pr" "$base" "$MAIN_BRANCH"
        return 0
    fi

    text="${title}"$'\n'"${body}"
    commits="$(gh pr view "$pr" --json commits --jq '.commits[].messageHeadline' 2>/dev/null || true)"
    text="${text}"$'\n'"${commits}"
    # Full commit messages catch Fixes #N in bodies of commits brought by promote
    commits_full="$(gh pr view "$pr" --json commits --jq '.commits[].messageBody' 2>/dev/null || true)"
    text="${text}"$'\n'"${commits_full}"
    nums="$(printf '%s' "$text" | extract_issue_numbers)"

    is_merged=0
    if [[ "$state" == "MERGED" || -n "$merged" ]]; then
        is_merged=1
    fi

    if (( is_merged == 0 )); then
        if [[ -z "$nums" ]]; then
            printf 'PR #%s (%s → %s): not merged; no Fixes/Closes refs\n' "$pr" "$state" "$base"
            return 0
        fi
        printf 'PR #%s (%s → %s): not merged — would close after merge to %s:\n' "$pr" "$state" "$base" "$base"
        while IFS= read -r n; do
            [[ -z "$n" ]] && continue
            printf '  pending #%s\n' "$n"
        done <<<"$nums"
        return 0
    fi

    if [[ -z "$nums" ]]; then
        printf 'PR #%s → %s: no linked issues found\n' "$pr" "$base"
        return 0
    fi
    printf 'PR #%s → %s\n' "$pr" "$base"
    while IFS= read -r n; do
        [[ -z "$n" ]] && continue
        close_one "$n" "$pr" "$base" "$url"
    done <<<"$nums"
}

if [[ -n "$PR_NUM" ]]; then
    process_pr "$PR_NUM"
    exit 0
fi

if [[ -z "$MERGED_INTO" && -z "$MERGED_SINCE" ]]; then
    printf 'close-linked-issues.sh: pass --pr N, or --merged-into BRANCH, or --merged-since ISO\n' >&2
    exit 2
fi

# Default batch target is main when requiring main-only policy
if (( REQUIRE_MAIN == 1 )) && [[ -z "$MERGED_INTO" ]]; then
    MERGED_INTO="$MAIN_BRANCH"
fi
if (( REQUIRE_MAIN == 1 )) && [[ -n "$MERGED_INTO" && "$MERGED_INTO" != "$MAIN_BRANCH" ]]; then
    printf 'close-linked-issues.sh: refusing --merged-into %s (policy: only %s; use --any-base to override)\n' \
        "$MERGED_INTO" "$MAIN_BRANCH" >&2
    exit 2
fi

QUERY="is:pr is:merged"
if [[ -n "$MERGED_INTO" ]]; then
    QUERY+=" base:${MERGED_INTO}"
fi
if [[ -n "$MERGED_SINCE" ]]; then
    since_date="${MERGED_SINCE:0:10}"
    QUERY+=" merged:>=${since_date}"
fi

printf 'Searching: %s (limit %s)\n' "$QUERY" "$LIMIT"
mapfile -t PRS < <(gh pr list --state merged --search "$QUERY" --limit "$LIMIT" --json number --jq '.[].number' 2>/dev/null)
if [[ ${#PRS[@]} -eq 0 ]]; then
    if [[ -n "$MERGED_INTO" ]]; then
        mapfile -t PRS < <(gh pr list --state merged --base "$MERGED_INTO" --limit "$LIMIT" --json number --jq '.[].number')
    else
        mapfile -t PRS < <(gh pr list --state merged --limit "$LIMIT" --json number --jq '.[].number')
    fi
fi

if [[ ${#PRS[@]} -eq 0 ]]; then
    printf 'No merged PRs matched.\n'
    exit 0
fi

for pr in "${PRS[@]}"; do
    process_pr "$pr" || true
done
