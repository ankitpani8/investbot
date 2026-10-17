#!/usr/bin/env bash
# Quick view of the autonomous build: tools/agents/status.sh
AG="${INVESTBOT_AGENTS_HOME:-$HOME/investbot_agents}"
c() { ls "$AG/$1"/*.md 2>/dev/null | wc -l; }
echo "queued: $(c queue)  active: $(c active)  done: $(c done)  blocked: $(c blocked)"
echo; echo "Active:";  ls "$AG/active"  2>/dev/null | sed 's/^/  /'
echo "Blocked:"; for f in "$AG"/blocked/*.md; do [ -e "$f" ] && echo "  $(basename "$f" .md): $(grep -A1 '^## Blocked' "$f" | tail -1)"; done
echo; echo "Recent runner log:"; tail -n 15 "$AG/logs/runner.log" 2>/dev/null | sed 's/^/  /'
[ -f "$AG/STOP" ] && echo && echo "STOP file present: workers exit after their current task."
