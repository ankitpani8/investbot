#!/usr/bin/env bash
# Move blocked tasks back to the queue, with a fresh branch, plus anything parked only because of them.
#   tools/agents/requeue.sh 160-index-membership-history [more ids...]
# Safe while the runner is running: workers claim it within ~2 minutes.
set -uo pipefail
AG="${INVESTBOT_AGENTS_HOME:-$HOME/investbot_agents}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
[ $# -ge 1 ] || { echo "usage: $0 <task-id> [...]"; ls "$AG/blocked" 2>/dev/null | sed 's/\.md$//; s/^/  /'; exit 2; }
requeue_one() {
  local id=$1 f="$AG/blocked/$1.md"
  [ -f "$f" ] || { echo "not blocked: $id"; return; }
  sed -i '/^## Blocked/,$d' "$f"                       # drop block notes; header + spec stay
  sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$f"          # trim trailing blank lines
  mv "$f" "$AG/queue/"
  git -C "$REPO" worktree remove --force "$AG/worktrees/$id" 2>/dev/null
  git -C "$REPO" branch -D "agent/$id" -q 2>/dev/null
  echo "requeued $id"
  # Dependents parked because this task was blocked
  local d
  for d in $(grep -l "$id(blocked)" "$AG"/blocked/*.md 2>/dev/null); do requeue_one "$(basename "$d" .md)"; done
}
for id in "$@"; do requeue_one "$id"; done
