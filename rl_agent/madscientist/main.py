"""MadScientist agent main entry point - Phase 0 stub.

Phase 0 (this commit): logs "MadScientist stub running" once per minute
and exits cleanly on SIGTERM. No research, no proposals, no orchestration.
The point is to make `docker compose up madscientist` succeed end-to-end
so we know the container builds, Mongo seeding works, and the dashboard
tab can be wired in parallel.

Phase 1 (next commit): replaces this stub with the real researcher loop
that consumes constants.DEFAULT_RESEARCH_CYCLE_INTERVAL_SECONDS, drafts
proposals, hands off to Judge + email + orchestrator.

To enable / disable the worker independently of the container:
    MADSCIENTIST_ENABLED=true   -> run main() loop (this file)
    MADSCIENTIST_ENABLED=false  -> entrypoint.sh exec's `sleep infinity`
                                   so the container exists but is idle.
The default (no env var set) is FALSE - safer to start; the operator
explicitly opts in once they're ready.
"""
from __future__ import annotations

import datetime
import os
import signal
import sys
import time

from . import constants


_should_exit = False


def _on_sigterm(signum, frame):
    """Graceful shutdown: set a flag the main loop polls."""
    global _should_exit
    print(
        f"madscientist: received signal {signum}; exiting on next tick.",
        flush=True)
    _should_exit = True


def main():
    """Phase 0 stub loop. Replace in Phase 1 with the researcher cycle."""
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    enabled = os.environ.get("MADSCIENTIST_ENABLED", "false").lower() == "true"
    if not enabled:
        # entrypoint.sh should have exec'd `sleep infinity` instead of
        # calling main() in this case, but defensive double-check.
        print(
            "madscientist: MADSCIENTIST_ENABLED is not 'true'; "
            "exiting immediately. Set the env var on the container to "
            "enable the worker loop.",
            flush=True)
        sys.exit(0)

    max_proposals_per_day = int(os.environ.get(
        "MAX_PROPOSALS_PER_DAY",
        constants.DEFAULT_MAX_PROPOSALS_PER_DAY))
    budget_usd_per_month = float(os.environ.get(
        "BUDGET_USD_PER_MONTH",
        constants.DEFAULT_BUDGET_USD_PER_MONTH))

    print(
        f"madscientist: Phase 0 stub running. "
        f"max_proposals_per_day={max_proposals_per_day}, "
        f"budget_usd_per_month={budget_usd_per_month}. "
        f"Phase 1 will replace this with the real researcher loop.",
        flush=True)

    # 60s heartbeat so docker stats / container logs show we're alive.
    tick = 0
    while not _should_exit:
        tick += 1
        if tick % 60 == 0:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(
                f"madscientist: heartbeat tick={tick} at {now} "
                f"(Phase 1 worker not yet implemented).",
                flush=True)
        time.sleep(1)

    print("madscientist: exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
