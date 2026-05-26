#!/bin/bash
# MadScientist container entrypoint.
#
# Two phases:
#   1. ALWAYS: run rl_agent/madscientist/seed.py to ensure the Mongo
#      collections + indexes exist + the placeholder rubric is upserted.
#      Idempotent; cheap; safe to run on every container start.
#
#   2. CONDITIONAL: if MADSCIENTIST_ENABLED=true, exec the main worker
#      loop. Otherwise exec `sleep infinity` so the container stays up
#      (for inspection via `docker compose exec madscientist bash`) but
#      no LLM calls / proposals / file edits actually happen.
#
# The conditional gating exists so the operator can:
#   * Bring up the full stack (`docker compose up -d`) without committing
#     to spinning the agent.
#   * Run ad-hoc commands in the container (`docker compose exec
#     madscientist python ...`) without fighting a running worker.
#   * Flip MADSCIENTIST_ENABLED=true + `docker compose restart
#     madscientist` to actually start the loop.
#
# Defaults to disabled. Phase 1's email + dashboard + budget tracking
# must all be in place before this should ever be flipped on.

set -euo pipefail

echo "[madscientist] container starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Seed Mongo (idempotent) ---------------------------------------------
echo "[madscientist] seeding Mongo collections + indexes..."
python -m rl_agent.madscientist.seed || {
    echo "[madscientist] seed.py failed - continuing anyway; the container"
    echo "                  will still come up so the operator can debug."
}

# --- Conditional worker startup ------------------------------------------
ENABLED="${MADSCIENTIST_ENABLED:-false}"
ENABLED_LC="$(echo "$ENABLED" | tr '[:upper:]' '[:lower:]')"

if [ "$ENABLED_LC" = "true" ] || [ "$ENABLED_LC" = "1" ] || [ "$ENABLED_LC" = "yes" ]; then
    echo "[madscientist] MADSCIENTIST_ENABLED=$ENABLED; launching worker."
    exec python -m rl_agent.madscientist.main
else
    echo "[madscientist] MADSCIENTIST_ENABLED=$ENABLED (worker DISABLED)."
    echo "                  Container will idle so it's available for inspection."
    echo "                  To enable: set MADSCIENTIST_ENABLED=true and"
    echo "                  'docker compose restart madscientist'."
    exec sleep infinity
fi
