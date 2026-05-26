"""MadScientist agent main entry point.

Runs the active workers in a single process when MADSCIENTIST_ENABLED=true:

  Phase 1A (live)    : judge_loop       - LLM-scores pending proposals
  Phase 1E (live)    : outcome_loop     - tallies completed TRAIN jobs
                                          into proposal.results
  Phase 1B (pending) : researcher_loop  - auto-generates proposals
  Phase 1C (pending) : orchestrator_loop - spawns Cursor SDK agents
                                           on approved proposals

The two live loops have different cadences (Judge polls every 30s,
Outcome ingester every 5 min) and don't share state, so we run them
in separate threads. SIGTERM/SIGINT set _should_exit; both loops
check it between cycles and drain cleanly within their poll interval.

Boot sequence:
  1. Install signal handlers (graceful exit).
  2. Read env config.
  3. If MADSCIENTIST_ENABLED is not 'true', exit immediately
     (entrypoint.sh's gate should normally short-circuit before we
     get here; defensive double-check).
  4. Open Mongo + Anthropic clients (Anthropic key only here, never
     read elsewhere - audit-friendly).
  5. Spawn judge + outcome threads.
  6. Wait for either thread to exit or _should_exit to be set.

The Anthropic client is constructed in main() (NOT in judge.py) so
the key never needs to be read in any other module - simpler to
audit the secret surface area.
"""
from __future__ import annotations

import datetime
import os
import signal
import sys
import threading
import time

from pymongo import MongoClient

from . import constants
from . import judge
from . import outcome_ingester


_should_exit = False


def _on_signal(signum, frame):
    """Graceful shutdown for SIGTERM + SIGINT.

    The judge_loop polls _should_exit via the should_stop_fn callback
    once per cycle, so the worker drains its current iteration
    cleanly before exiting. Docker's default 10s grace is plenty.
    """
    global _should_exit
    print(
        f"madscientist: received signal {signum}; will exit after "
        f"current judge cycle completes.",
        flush=True)
    _should_exit = True


def _open_mongo():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _open_anthropic():
    """Lazy import + construct Anthropic client.

    Raises if ANTHROPIC_API_KEY is missing - the worker shouldn't be
    enabled without one. The Phase 1B Researcher will share this
    client (single key, two workers).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required when MADSCIENTIST_ENABLED=true. "
            "Set it in your .env (see .env.example) and restart the "
            "madscientist container.")
    # Lazy import so the module loads cleanly in environments without
    # the anthropic package (e.g., AST-checking on the dev host).
    from anthropic import Anthropic  # type: ignore
    return Anthropic(api_key=api_key)


def main():
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    enabled = os.environ.get(
        "MADSCIENTIST_ENABLED", "false").lower() == "true"
    if not enabled:
        print(
            "madscientist: MADSCIENTIST_ENABLED is not 'true'; exiting. "
            "Set the env var + 'docker compose restart madscientist' "
            "to enable the worker.",
            flush=True)
        sys.exit(0)

    max_proposals_per_day = int(os.environ.get(
        "MAX_PROPOSALS_PER_DAY",
        constants.DEFAULT_MAX_PROPOSALS_PER_DAY))
    budget_usd_per_month = float(os.environ.get(
        "BUDGET_USD_PER_MONTH",
        constants.DEFAULT_BUDGET_USD_PER_MONTH))
    judge_poll = int(os.environ.get(
        "JUDGE_POLL_INTERVAL_SECONDS", "30"))
    outcome_poll = int(os.environ.get(
        "OUTCOME_POLL_INTERVAL_SECONDS",
        constants.DEFAULT_OUTCOME_POLL_INTERVAL_SECONDS))

    print(
        f"madscientist: Phase 1A+1E workers starting. "
        f"max_proposals_per_day={max_proposals_per_day}, "
        f"budget_usd_per_month={budget_usd_per_month}, "
        f"judge_poll_seconds={judge_poll}, "
        f"outcome_poll_seconds={outcome_poll}.",
        flush=True)

    try:
        db = _open_mongo()
    except Exception as e:  # noqa: BLE001
        print(f"madscientist: failed to open Mongo: {e}", flush=True)
        sys.exit(2)

    try:
        anthropic_client = _open_anthropic()
    except Exception as e:  # noqa: BLE001
        print(f"madscientist: failed to construct Anthropic client: {e}",
              flush=True)
        sys.exit(2)

    # Spawn the two loops as daemon threads. Each polls
    # _should_exit between cycles for graceful SIGTERM/SIGINT drain.
    # Daemon=True so a sibling-thread crash doesn't pin the process
    # open after the other thread exits.
    judge_thread = threading.Thread(
        target=judge.judge_loop,
        kwargs={
            "db": db,
            "anthropic_client": anthropic_client,
            "poll_interval_seconds": judge_poll,
            "should_stop_fn": lambda: _should_exit,
        },
        name="judge-loop",
        daemon=True,
    )
    outcome_thread = threading.Thread(
        target=outcome_ingester.outcome_loop,
        kwargs={
            "db": db,
            "poll_interval_seconds": outcome_poll,
            "should_stop_fn": lambda: _should_exit,
        },
        name="outcome-loop",
        daemon=True,
    )
    judge_thread.start()
    outcome_thread.start()

    # Block the main thread until either:
    #   (a) a worker thread exits (which usually means it hit an
    #       unrecoverable error - graceful exit goes through
    #       _should_exit + a sub-poll-interval drain),
    #   (b) we get SIGTERM/SIGINT and _should_exit becomes True.
    # We tick at 1s so SIGTERM responsiveness is sub-second even
    # though the worker loops sleep for tens of seconds.
    while not _should_exit:
        if not judge_thread.is_alive():
            print(
                "madscientist: judge thread exited unexpectedly; "
                "signalling other workers to drain.",
                flush=True)
            break
        if not outcome_thread.is_alive():
            print(
                "madscientist: outcome thread exited unexpectedly; "
                "signalling other workers to drain.",
                flush=True)
            break
        time.sleep(1)

    # Worker loops each call should_stop_fn(). We wait up to their
    # max poll interval + a small buffer for graceful drain. Threads
    # are daemons so if drain hangs we still exit on process exit.
    drain_deadline = max(judge_poll, outcome_poll) + 5
    judge_thread.join(timeout=drain_deadline)
    outcome_thread.join(timeout=drain_deadline)
    print("madscientist: exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
