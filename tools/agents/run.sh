#!/usr/bin/env bash
# InvestBot autonomous build runner.
#
#   tools/agents/run.sh [-n WORKERS] [-m MAX_TASKS]
#
# Each worker loops: claim a task → fresh git worktree on agent/<task> from origin/main →
# headless Claude Code works on it → the RUNNER (not the agent) rebases, runs the test
# suite and fast-forwards main. Failing work is parked in blocked/ with its branch pushed.
#
# Usage limits: when Claude reports a usage limit, the worker sleeps until the reset and
# resumes the same conversation (--continue). The run never needs a human until the queue
# is empty or everything left is blocked.
#
# State lives OUTSIDE the repo in ~/investbot_agents:
#   queue/ active/ done/ blocked/   task files (markdown with a small header)
#   logs/                           one log per task + runner.log
#   worktrees/                      one git worktree per task
#   STOP                            touch this file to stop after current tasks
#
# Safety rails (permission checks are skipped, so these are the guardrails):
#   - agents get NO GitHub credentials (git credential helper and gh config are blanked),
#     so only the runner can push, and it only pushes after tests pass
#   - agents get a scratch copy of the DB via DATABASE_URL; the real DB is never touched
#   - each agent run is capped by wall-clock (AGENT_TIMEOUT, default 3h)
set -uo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
AG="${INVESTBOT_AGENTS_HOME:-$HOME/investbot_agents}"
Q="$AG/queue"; A="$AG/active"; D="$AG/done"; B="$AG/blocked"; L="$AG/logs"; WT="$AG/worktrees"
LOCKS="$AG/locks"
WORKERS=2
MAX_TASKS=100
# Model routing. Workers burn ~95% of tokens, so they default to Sonnet; the planner runs
# rarely and its task specs steer every worker, so it gets Opus. A task file can override
# with `model:` / `effort:` header lines (used for the maths-heavy tasks).
WORKER_MODEL="${WORKER_MODEL:-sonnet}"
WORKER_EFFORT="${WORKER_EFFORT:-high}"
PLANNER_MODEL="${PLANNER_MODEL:-opus}"
PLANNER_EFFORT="${PLANNER_EFFORT:-high}"
HAS_EFFORT=0
claude --help 2>/dev/null | grep -q -- '--effort' && HAS_EFFORT=1
AGENT_TIMEOUT="${AGENT_TIMEOUT:-3h}"
DB_MAIN="$HOME/investbot_data/investbot.db"
PY="$REPO/.venv/bin/python"

while getopts "n:m:" o; do
  case $o in n) WORKERS=$OPTARG ;; m) MAX_TASKS=$OPTARG ;; *) exit 2 ;; esac
done

mkdir -p "$Q" "$A" "$D" "$B" "$L" "$WT" "$LOCKS" "$AG/no-gh"
RLOG="$L/runner.log"
log() { echo "$(date '+%F %T') [${WID:-main}] $*" | tee -a "$RLOG"; }

command -v claude >/dev/null || { echo "claude not found"; exit 1; }
[ -x "$PY" ] || { echo "No venv at $REPO/.venv"; exit 1; }

# ── Seed the queue on first run ─────────────────────────────────────────
if [ -z "$(find "$Q" "$A" "$D" "$B" -name '*.md' 2>/dev/null | head -1)" ]; then
  cp "$REPO"/tools/agents/seed_tasks/*.md "$Q"/ 2>/dev/null && log "Seeded queue with $(ls "$Q" | wc -l) tasks"
fi

# Any task left in active/ by a crashed run goes back to the queue.
for f in "$A"/*.md; do [ -e "$f" ] && mv "$f" "$Q"/ && log "Requeued stale $(basename "$f")"; done

# A crash can leave worktrees registered whose directories are gone or half-written.
git -C "$REPO" worktree prune 2>/dev/null

# Inbox: task files committed to tools/agents/inbox/ on origin/main are imported once.
# This lets tasks be added from anywhere (GitHub, another session) without touching this machine.
import_inbox() {
  git -C "$REPO" fetch -q origin 2>/dev/null || return
  local f id
  for f in $(git -C "$REPO" ls-tree --name-only origin/main tools/agents/inbox/ 2>/dev/null | grep '\.md$' | grep -v '/README\.md$'); do
    id=$(basename "$f" .md)
    [ -e "$Q/$id.md" ] || [ -e "$A/$id.md" ] || [ -e "$D/$id.md" ] || [ -e "$B/$id.md" ] && continue
    git -C "$REPO" show "origin/main:$f" > "$Q/$id.md" && log "Imported inbox task $id"
  done
}

# Why is nothing claimable? Park tasks whose dependencies can never be satisfied.
explain_idle() {
  local f d id reasons waiting=""
  for f in "$Q"/*.md; do
    [ -e "$f" ] || continue
    id=$(basename "$f" .md); reasons=""
    for d in $(deps_of "$f"); do
      if [ -e "$D/$d.md" ]; then continue
      elif [ -e "$B/$d.md" ]; then reasons+=" $d(blocked)"
      elif [ -e "$Q/$d.md" ] || [ -e "$A/$d.md" ]; then waiting+=" $id<-$d"
      else reasons+=" $d(unknown)"
      fi
    done
    if [ -n "$reasons" ]; then
      { echo; echo "## Blocked $(date '+%F %T')"; echo "Unsatisfiable dependencies:$reasons"; } >>"$f"
      mv "$f" "$B"/ 2>/dev/null && log "PARKED $id: unsatisfiable dependencies:$reasons"
    fi
  done
  [ -n "$waiting" ] && log "Idle: queued tasks waiting on running/queued work:$waiting"
}

deps_of() { sed -n 's/^depends:[[:space:]]*//p' "$1" | head -1 | tr -d '[],' ; }

claim() {   # prints claimed task path, or nothing
  (
    flock 9
    for f in $(ls "$Q"/*.md 2>/dev/null | sort); do
      ok=1
      for d in $(deps_of "$f"); do [ -e "$D/$d.md" ] || { ok=0; break; }; done
      if [ $ok = 1 ]; then mv "$f" "$A"/ && echo "$A/$(basename "$f")"; exit 0; fi
    done
  ) 9>"$LOCKS/claim.lock"
}

agent_env() {   # run a command with the agent sandbox environment
  local wt=$1; shift
  env -u GH_TOKEN -u GITHUB_TOKEN \
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0= \
    GIT_TERMINAL_PROMPT=0 GH_CONFIG_DIR="$AG/no-gh" \
    DATABASE_URL="sqlite:///$wt/.agent.db" INVESTBOT_AGENT=1 \
    PATH="$REPO/.venv/bin:$PATH" VIRTUAL_ENV="$REPO/.venv" \
    "$@"
}

# Returns 0 = finished, 2 = fatal (auth etc.). Sleeps through usage limits.
run_claude() {   # wt prompt logf model effort
  local wt=$1 prompt=$2 logf=$3 model=$4 effort=$5 cont=0
  while :; do
    [ -f "$AG/STOP" ] && return 2
    local args=(-p --model "$model" --dangerously-skip-permissions)
    [ $HAS_EFFORT = 1 ] && [ -n "$effort" ] && args+=(--effort "$effort")
    [ $cont = 1 ] && args+=(--continue)
    local msg="$prompt"
    [ $cont = 1 ] && msg="You were interrupted by a usage limit. Continue the task from where you left off. Re-read your previous messages and the git log on this branch first."
    local start=0; [ -f "$logf" ] && start=$(wc -l < "$logf")
    ( cd "$wt" && agent_env "$wt" timeout "$AGENT_TIMEOUT" claude "${args[@]}" "$msg" ) >>"$logf" 2>&1
    local rc=$?
    local out; out=$(tail -n +"$((start+1))" "$logf")
    if [ $rc != 0 ] && [ $rc != 124 ] && echo "$out" | tail -15 | grep -qiE 'usage limit|limit reached|rate.limit|out of (extra )?usage|resets? (at|in)'; then
      local epoch now wait
      epoch=$(echo "$out" | grep -oE 'limit reached\|[0-9]{10}' | grep -oE '[0-9]{10}' | tail -1)
      now=$(date +%s)
      if [ -n "$epoch" ] && [ "$epoch" -gt "$now" ]; then wait=$((epoch - now + 120)); else wait=1800; fi
      log "Usage limit hit. Sleeping $((wait/60)) min, then resuming."
      sleep "$wait"
      cont=1
      continue
    fi
    if echo "$out" | grep -qiE 'invalid api key|please run /login|not logged in|authentication_error'; then
      log "FATAL: Claude Code is not authenticated. Run 'claude' once interactively and log in."
      touch "$AG/STOP"; return 2
    fi
    [ $rc = 124 ] && echo "RESULT: BLOCKED agent timed out after $AGENT_TIMEOUT" >>"$logf"
    return 0
  done
}

block() {   # task_path reason wt branch
  local task=$1 reason=$2 wt=$3 br=$4 id; id=$(basename "$task" .md)
  { echo; echo "## Blocked $(date '+%F %T')"; echo "$reason"; echo "Branch: $br  Worktree: $wt  Log: $L/$id.log"; } >>"$task"
  mv "$task" "$B"/
  ( cd "$wt" && git push -q -f origin "HEAD:refs/heads/$br" ) 2>>"$RLOG" || true
  log "BLOCKED $id: $reason"
}

integrate() {   # task_path wt branch → 0 merged / 1 blocked
  local task=$1 wt=$2 br=$3 id; id=$(basename "$task" .md)
  (
    flock 8
    cd "$wt" || exit 1
    # Commit anything the agent left uncommitted (tests still gate it)
    if [ -n "$(git status --porcelain)" ]; then
      git add -A && git commit -q -m "agent/$id: uncommitted changes at end of run"
    fi
    git fetch -q origin
    if [ "$(git rev-list --count origin/main..HEAD)" = 0 ]; then echo "no commits"; exit 3; fi
    if ! git rebase -q origin/main >>"$L/$id.log" 2>&1; then git rebase --abort; echo "rebase conflict with main"; exit 4; fi
    if ! agent_env "$wt" "$PY" -m pytest -q >>"$L/$id.log" 2>&1; then echo "test suite failed after rebase"; exit 5; fi
    for i in 1 2 3; do
      git push -q origin HEAD:main 2>>"$L/$id.log" && exit 0
      git fetch -q origin && git rebase -q origin/main || { git rebase --abort; echo "rebase conflict on retry"; exit 4; }
    done
    echo "push to main failed"; exit 6
  ) 8>"$LOCKS/merge.lock"
}

run_task() {
  local task=$1 id br wt logf; id=$(basename "$task" .md); br="agent/$id"; wt="$WT/$id"; logf="$L/$id.log"
  log "START $id"
  git -C "$REPO" fetch -q origin
  if [ ! -d "$wt" ]; then
    git -C "$REPO" worktree add -q -B "$br" "$wt" origin/main || { block "$task" "worktree creation failed" "$wt" "$br"; return; }
  fi
  [ -f "$DB_MAIN" ] && [ ! -f "$wt/.agent.db" ] && cp "$DB_MAIN" "$wt/.agent.db"

  local prompt
  prompt="$(sed -e "s|{{TASK_ID}}|$id|g" -e "s|{{BRANCH}}|$br|g" -e "s|{{DB}}|$wt/.agent.db|g" \
            "$REPO/tools/agents/prompts/worker.md")

$(awk 'NR==1 && /^---$/ {h=1; next} h && /^---$/ {h=0; next} !h' "$task")"

  echo "===== $(date '+%F %T') $id =====" >>"$logf"
  local model effort
  model=$(sed -n 's/^model:[[:space:]]*//p' "$task" | head -1); model=${model:-$WORKER_MODEL}
  effort=$(sed -n 's/^effort:[[:space:]]*//p' "$task" | head -1); effort=${effort:-$WORKER_EFFORT}
  echo "model=$model effort=$effort" >>"$logf"
  log "  $id → $model/$effort"
  run_claude "$wt" "$prompt" "$logf" "$model" "$effort" || { mv "$task" "$Q"/; log "Stopped during $id (requeued)"; return; }

  if tail -50 "$logf" | grep -q 'RESULT: BLOCKED'; then
    block "$task" "$(tail -50 "$logf" | grep 'RESULT: BLOCKED' | tail -1)" "$wt" "$br"; return
  fi
  local why
  if why=$(integrate "$task" "$wt" "$br"); then
    mv "$task" "$D"/
    git -C "$REPO" worktree remove --force "$wt" 2>/dev/null
    git -C "$REPO" branch -D "$br" -q 2>/dev/null
    log "DONE $id → merged to main ($(git -C "$REPO" rev-parse --short origin/main 2>/dev/null))"
  else
    block "$task" "${why:-integration failed}" "$wt" "$br"
  fi
}

plan() {   # refill the queue when nothing is claimable
  (
    flock -n 7 || exit 0
    [ -n "$(ls "$Q"/*.md 2>/dev/null)" ] && exit 0
    # Planner already said "nothing left" and no task has finished since: don't re-ask.
    local newest; newest=$(ls -t "$D"/*.md "$B"/*.md 2>/dev/null | head -1)
    if [ -f "$AG/PLANNER_EMPTY" ] && { [ -z "$newest" ] || [ "$AG/PLANNER_EMPTY" -nt "$newest" ]; }; then exit 0; fi
    local pwt="$WT/_planner"
    git -C "$REPO" fetch -q origin
    git -C "$REPO" worktree remove --force "$pwt" 2>/dev/null
    git -C "$REPO" worktree prune 2>/dev/null
    git -C "$REPO" worktree add -q -f --detach "$pwt" origin/main || { WID=planner log "Planner worktree creation failed"; exit 0; }
    local existing; existing=$(ls "$D" "$A" "$B" 2>/dev/null | grep '\.md$' | sed 's/\.md$//' | sort -u | tr '\n' ' ')
    local prompt
    prompt="$(sed -e "s|{{QUEUE}}|$Q|g" -e "s|{{EXISTING}}|$existing|g" "$REPO/tools/agents/prompts/planner.md")"
    WID=planner log "Planning new tasks"
    run_claude "$pwt" "$prompt" "$L/planner.log" "$PLANNER_MODEL" "$PLANNER_EFFORT"
    git -C "$REPO" worktree remove --force "$pwt" 2>/dev/null
    local nq; nq=$(ls "$Q"/*.md 2>/dev/null | wc -l)
    if [ "$nq" = 0 ]; then touch "$AG/PLANNER_EMPTY"; else rm -f "$AG/PLANNER_EMPTY"; fi
    WID=planner log "Planner queued $nq task(s)"
  ) 7>"$LOCKS/plan.lock"
}

worker() {
  WID="w$1"; local idle=0
  log "worker up"
  while :; do
    [ -f "$AG/STOP" ] && { log "STOP file found, exiting"; break; }
    [ "$(ls "$D" | wc -l)" -ge "$MAX_TASKS" ] && { log "MAX_TASKS reached"; break; }
    local task; task=$(claim)
    if [ -z "$task" ]; then
      if [ -z "$(ls "$A"/*.md 2>/dev/null)" ]; then plan; fi
      task=$(claim)
    fi
    if [ -z "$task" ]; then
      idle=$((idle+1))
      # Exit only when the planner has confirmed there's nothing left, no task is running,
      # and the planner isn't mid-run (another worker may be planning right now).
      if [ $idle -ge 3 ] && [ -f "$AG/PLANNER_EMPTY" ] && [ -z "$(ls "$A"/*.md 2>/dev/null)" ] \
         && flock -n "$LOCKS/plan.lock" true; then
        log "Queue empty and planner has nothing left; exiting"; break
      fi
      # Every ~10 min of idling, say why (and park dead-end tasks) instead of going silent.
      if [ $((idle % 5)) = 1 ]; then import_inbox; explain_idle; fi
      sleep 120; continue
    fi
    idle=0
    run_task "$task"
  done
}

import_inbox
log "Runner start: repo=$REPO workers=$WORKERS worker=$WORKER_MODEL/$WORKER_EFFORT planner=$PLANNER_MODEL/$PLANNER_EFFORT effort_flag=$HAS_EFFORT max_tasks=$MAX_TASKS"
if [ -f "$AG/runner.pid" ] && kill -0 "$(cat "$AG/runner.pid")" 2>/dev/null; then
  echo "A runner is already running (pid $(cat "$AG/runner.pid")). Stop it with: touch $AG/STOP"; exit 1
fi
echo $$ > "$AG/runner.pid"
trap 'rm -f "$AG/runner.pid"' EXIT
rm -f "$AG/STOP" "$AG/PLANNER_EMPTY"
for i in $(seq 1 "$WORKERS"); do worker "$i" & sleep 20; done
wait
log "Runner finished. done=$(ls "$D" | wc -l) blocked=$(ls "$B" | wc -l) queued=$(ls "$Q" | wc -l)"
