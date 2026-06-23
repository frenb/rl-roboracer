"""MadScientist agent main entry point.

Runs the active workers in a single process when MADSCIENTIST_ENABLED=true:

  Phase 1A (live)      : judge_loop       - LLM-scores pending proposals
  Phase 1B (live)      : research_loop    - auto-generates proposals from
                                            arxiv + codebase introspection
  Phase 1C-MVP (live)  : orchestrate_loop - seeds derived designs +
                                            queues TRAIN jobs on
                                            approved proposals (no
                                            Cursor SDK yet)
  Phase 1E (live)      : outcome_loop     - tallies completed TRAIN jobs
                                            into proposal.results
  Phase 1C-Full (pending) : Cursor SDK code-writing agent for
                            proposals that need new SCHEMA fields
  Phase 1D (pending)   : email_bridge     - outbound proposals + inbound
                                            natural-language replies

All loops run as daemon threads. Each polls _should_exit between
cycles for graceful drain on SIGTERM/SIGINT. Daemon=True so a
sibling-thread crash never pins the process open.

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
from . import email_bridge
from . import judge
from . import orchestrator
from . import outcome_ingester
from . import researcher


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
    orchestrator_poll = int(os.environ.get(
        "ORCHESTRATOR_POLL_INTERVAL_SECONDS", "30"))
    max_jobs_per_proposal = int(os.environ.get(
        "MAX_JOBS_PER_PROPOSAL",
        orchestrator.DEFAULT_MAX_JOBS_PER_PROPOSAL))
    max_queued_jobs = int(os.environ.get(
        "MAX_QUEUED_JOBS",
        constants.DEFAULT_MAX_QUEUED_JOBS))
    research_cycle_interval = int(os.environ.get(
        "RESEARCH_CYCLE_INTERVAL_SECONDS",
        constants.DEFAULT_RESEARCH_CYCLE_INTERVAL_SECONDS))
    research_query = os.environ.get(
        "RESEARCH_QUERY", researcher.DEFAULT_RESEARCH_QUERY)
    # Email bridge config. base_url is where the magic-link buttons in
    # the proposal email point; default localhost works only when mail
    # is opened on this host - set DASHBOARD_PUBLIC_URL to a LAN /
    # Tailscale / tunnel address to make the buttons clickable from a
    # phone. token_secret signs the buttons; empty => email still sends
    # but without one-click buttons (dashboard link only).
    notify_poll = int(os.environ.get(
        "NOTIFY_POLL_INTERVAL_SECONDS",
        email_bridge.DEFAULT_NOTIFY_POLL_INTERVAL_SECONDS))
    dashboard_public_url = os.environ.get(
        "DASHBOARD_PUBLIC_URL", "http://localhost:8080").strip()
    token_secret = os.environ.get("MADSCIENTIST_TOKEN_SECRET", "").strip()
    token_ttl = int(os.environ.get(
        "DECISION_TOKEN_TTL_SECONDS",
        email_bridge.DEFAULT_DECISION_TOKEN_TTL_SECONDS))

    print(
        f"madscientist: Phase 1A+1B+1C-MVP+1D+1E workers starting. "
        f"max_proposals_per_day={max_proposals_per_day}, "
        f"budget_usd_per_month={budget_usd_per_month}, "
        f"judge_poll_seconds={judge_poll}, "
        f"outcome_poll_seconds={outcome_poll}, "
        f"orchestrator_poll_seconds={orchestrator_poll}, "
        f"notify_poll_seconds={notify_poll}, "
        f"max_jobs_per_proposal={max_jobs_per_proposal}, "
        f"max_queued_jobs={max_queued_jobs}, "
        f"research_cycle_interval_seconds={research_cycle_interval}, "
        f"dashboard_public_url={dashboard_public_url!r}, "
        f"email_buttons={'on' if token_secret else 'off'}, "
        f"research_query={research_query!r}.",
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

    # Spawn the three loops as daemon threads. Each polls
    # _should_exit between cycles for graceful SIGTERM/SIGINT drain.
    # Daemon=True so a sibling-thread crash doesn't pin the process
    # open after the other threads exit.
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
    researcher_thread = threading.Thread(
        target=researcher.research_loop,
        kwargs={
            "db": db,
            "anthropic_client": anthropic_client,
            "poll_interval_seconds": research_cycle_interval,
            "monthly_budget_usd": budget_usd_per_month,
            "max_proposals_per_day": max_proposals_per_day,
            "max_queued_jobs": max_queued_jobs,
            "research_query": research_query,
            "should_stop_fn": lambda: _should_exit,
        },
        name="research-loop",
        daemon=True,
    )
    orchestrator_thread = threading.Thread(
        target=orchestrator.orchestrate_loop,
        kwargs={
            "db": db,
            "poll_interval_seconds": orchestrator_poll,
            "max_jobs": max_jobs_per_proposal,
            "should_stop_fn": lambda: _should_exit,
        },
        name="orchestrator-loop",
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
    email_thread = threading.Thread(
        target=email_bridge.notify_loop,
        kwargs={
            "db": db,
            "base_url": dashboard_public_url,
            "token_secret": token_secret,
            "poll_interval_seconds": notify_poll,
            "token_ttl_seconds": token_ttl,
            "should_stop_fn": lambda: _should_exit,
        },
        name="email-bridge-loop",
        daemon=True,
    )
    judge_thread.start()
    researcher_thread.start()
    orchestrator_thread.start()
    outcome_thread.start()
    email_thread.start()

    # Block the main thread until either:
    #   (a) any worker thread exits (which usually means it hit an
    #       unrecoverable error - graceful exit goes through
    #       _should_exit + a sub-poll-interval drain),
    #   (b) we get SIGTERM/SIGINT and _should_exit becomes True.
    # We tick at 1s so SIGTERM responsiveness is sub-second even
    # though the worker loops sleep for tens of seconds or hours.
    worker_threads = [
        ("judge", judge_thread),
        ("research", researcher_thread),
        ("orchestrator", orchestrator_thread),
        ("outcome", outcome_thread),
        ("email", email_thread),
    ]
    while not _should_exit:
        died = next(
            ((name, t) for name, t in worker_threads if not t.is_alive()),
            None)
        if died is not None:
            print(
                f"madscientist: {died[0]} thread exited unexpectedly; "
                f"signalling other workers to drain.",
                flush=True)
            break
        time.sleep(1)

    # Worker loops each call should_stop_fn(). We wait up to their
    # max poll interval + a small buffer for graceful drain. Threads
    # are daemons so if drain hangs we still exit on process exit.
    drain_deadline = max(
        judge_poll, outcome_poll, orchestrator_poll,
        # Research cycle is much longer (hours); we don't actually wait
        # the full interval - the loop's should_stop_fn check is on a
        # tighter cadence inside time.sleep, so 60s drain is plenty.
        60) + 5
    for _name, t in worker_threads:
        t.join(timeout=drain_deadline)
    print("madscientist: exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
