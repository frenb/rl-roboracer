"""MadScientist agent main entry point - Phase 1A.

This module runs the **Judge** worker loop when
MADSCIENTIST_ENABLED=true. Phase 1B will add the Researcher loop
(probably in a sibling thread or separate process) and Phase 1C the
Orchestrator. For now: one worker, one loop, one purpose.

Boot sequence:
  1. Install SIGTERM/SIGINT handlers (graceful exit).
  2. Read env config (MADSCIENTIST_ENABLED, MAX_PROPOSALS_PER_DAY,
     BUDGET_USD_PER_MONTH, etc.).
  3. If disabled, exit immediately (entrypoint.sh's gate should
     normally short-circuit before we get here; this is defensive).
  4. Open Mongo + Anthropic clients.
  5. Enter judge_loop, which polls db.proposals for pending_judge
     and processes one per cycle.

The Anthropic client is constructed here (NOT in judge.py) so the
key never needs to be read in any other module - simpler to audit
the secret surface area.
"""
from __future__ import annotations

import datetime
import os
import signal
import sys
import time

from pymongo import MongoClient

from . import constants
from . import judge


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
    poll_interval = int(os.environ.get(
        "JUDGE_POLL_INTERVAL_SECONDS", "30"))

    print(
        f"madscientist: Phase 1A worker starting. "
        f"max_proposals_per_day={max_proposals_per_day}, "
        f"budget_usd_per_month={budget_usd_per_month}, "
        f"poll_interval_seconds={poll_interval}.",
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

    # Hand off to the judge loop. It polls + processes one proposal
    # per cycle and checks should_stop_fn between cycles for graceful
    # exit on SIGTERM/SIGINT.
    judge.judge_loop(
        db,
        anthropic_client,
        poll_interval_seconds=poll_interval,
        should_stop_fn=lambda: _should_exit,
    )

    print("madscientist: exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
