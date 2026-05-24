"""Backfill model.job_id for historical model documents.

Context
-------
Before commit <add_model job_id plumbing>, ``rl_agent.robotaxi.add_model``
did not stamp ``job_id`` on the model document. Models created prior
to that change therefore have ``job_id == None``, which means the
dashboard's Models -> Job ID column shows an em-dash and the cross-
link to the Jobs tab (and the regex-filtered Tensorboard comparison
flow downstream) doesn't light up for those rows.

This script populates ``job_id`` for those legacy models by matching
each model's ``create_date`` to the TRAIN job whose
[``started_at``, ``ended_at``] window contains it.

Why timestamp matching is unambiguous here
------------------------------------------
The sim-controller's job loop is strictly serial - ``run_jobs_loop()``
in robotaxi.py drains one job at a time via ``do_job(j)``, which only
returns when the job finishes. ``add_model`` is called from inside
``main()`` (which is called from ``do_job``), so every model created
by training inherently falls within the active job's
[started_at, ended_at] interval, and no two training intervals
overlap. If we ever do see overlapping intervals (defensively
guarded below) we pick the one with the latest ``started_at <=
create_date`` - the most recently started job that's still going.

Usage
-----
Run from inside the sim-controller container (which has pymongo +
the right Mongo connection params already configured via env). Dry
run first to inspect proposed changes; rerun without --dry-run to
apply:

    docker compose exec sim-controller python /python_ws/src/backfill_model_job_ids.py --dry-run
    docker compose exec sim-controller python /python_ws/src/backfill_model_job_ids.py

Idempotent: re-running after a successful pass is a no-op because
models that already have ``job_id`` are skipped.

Limitations
-----------
* Models whose ``create_date`` falls outside every TRAIN job's window
  (e.g., the job document was hard-deleted from Mongo, or the model
  predates job-status tracking) get logged as ``unmatched`` and left
  with ``job_id=None``.
* ``RandomPyPolicy`` baseline rows are created by EVAL jobs, not
  TRAIN jobs. They legitimately have no training job; we skip them
  rather than mis-assign an unrelated TRAIN job's id.
"""

import argparse
import datetime
import os
import sys

from pymongo import MongoClient


def _parse_args():
    p = argparse.ArgumentParser(
        description="Backfill model.job_id from job started_at/ended_at windows.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed changes without writing to Mongo.")
    p.add_argument(
        "--mongo-host",
        default=os.environ.get("MONGO_HOST", "mongo"),
        help="Mongo host (default: 'mongo', matches the sim-controller container's DNS alias).")
    p.add_argument(
        "--mongo-user",
        default=os.environ.get("MONGO_USER", "root"))
    p.add_argument(
        "--mongo-password",
        default=os.environ.get("MONGO_PASSWORD", "example"))
    p.add_argument(
        "--db-name",
        default=os.environ.get("DATABASE_NAME", "robotaxi"))
    return p.parse_args()


def main():
    args = _parse_args()

    client = MongoClient(args.mongo_host,
                         username=args.mongo_user,
                         password=args.mongo_password)
    db = client[args.db_name]

    # Build a sorted list of TRAIN jobs with usable windows. We
    # accept jobs that have started_at even without ended_at - if a
    # job is still IN_PROGRESS (or got killed without an ended_at
    # stamp) we treat its upper bound as "now". This means a model
    # whose create_date falls AFTER an active job's started_at gets
    # attributed to that job, which is correct.
    now = datetime.datetime.now(datetime.timezone.utc)
    train_jobs_cursor = db.jobs.find(
        {"job_type": "TRAIN", "started_at": {"$exists": True}},
        {"_id": 1, "started_at": 1, "ended_at": 1, "status": 1},
    )
    jobs = []
    for j in train_jobs_cursor:
        started = j.get("started_at")
        ended = j.get("ended_at") or now
        if started is None:
            continue
        # Defensive: some legacy docs stored started_at as a string.
        # pymongo returns the bson datetime as a tz-naive datetime;
        # we normalise to tz-aware UTC so comparisons against `now`
        # and against model.create_date are unambiguous.
        started = _to_aware(started)
        ended = _to_aware(ended)
        if started is None or ended is None:
            continue
        # Some "DONE" jobs were stamped with ended_at via update_job's
        # path which used datetime.now(timezone.utc), so they should
        # be tz-aware. But the legacy started_at could be either.
        jobs.append({
            "_id": j["_id"],
            "started_at": started,
            "ended_at": ended,
            "status": j.get("status"),
        })
    # Sort by started_at descending so when we walk the list we can
    # short-circuit at the first match (= the most recently started
    # job whose window contains create_date).
    jobs.sort(key=lambda x: x["started_at"], reverse=True)

    print(f"Loaded {len(jobs)} TRAIN jobs with usable windows.", flush=True)

    # Now scan models. We deliberately query all models without a
    # job_id (covers both "field missing" and "field present but
    # None") rather than restricting to model_type so we also pick
    # up Greedy/SAC variants. RandomPyPolicy rows are created by
    # save_results_to_db, not add_model, so they may or may not have
    # job_id - either way the unmatched path handles them gracefully.
    models_cursor = db.models.find(
        {"$or": [{"job_id": None}, {"job_id": {"$exists": False}}]},
        {"_id": 1, "create_date": 1, "model_type": 1, "location": 1, "robot_type": 1},
    )

    matched = 0
    skipped_no_create_date = 0
    unmatched = 0
    unmatched_examples = []
    proposed_updates = []  # (model_id, job_id) tuples

    for m in models_cursor:
        create_date = m.get("create_date")
        if create_date is None:
            skipped_no_create_date += 1
            continue
        cd = _to_aware(create_date)
        if cd is None:
            skipped_no_create_date += 1
            continue
        # Linear scan in sorted order. Lists are short (one entry per
        # training job ever run - usually tens, not thousands), so a
        # full scan is cheap and clearer than maintaining an interval
        # tree.
        chosen = None
        for j in jobs:
            if j["started_at"] <= cd <= j["ended_at"]:
                chosen = j
                break  # most recently started match wins
        if chosen is None:
            unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append({
                    "_id": str(m["_id"]),
                    "create_date": cd.isoformat(),
                    "model_type": m.get("model_type"),
                    "location": m.get("location"),
                })
            continue
        matched += 1
        proposed_updates.append((m["_id"], str(chosen["_id"])))
        print(
            f"  model {str(m['_id'])} ({m.get('model_type','?')}) "
            f"created {cd.isoformat()}  ->  job {str(chosen['_id'])} "
            f"(window {chosen['started_at'].isoformat()} .. "
            f"{chosen['ended_at'].isoformat()})",
            flush=True)

    print("", flush=True)
    print(f"Scanned models without job_id: {matched + unmatched + skipped_no_create_date}", flush=True)
    print(f"  matched (job window contains create_date): {matched}", flush=True)
    print(f"  unmatched (no covering job window):        {unmatched}", flush=True)
    print(f"  skipped (model has no create_date):        {skipped_no_create_date}", flush=True)

    if unmatched_examples:
        print("", flush=True)
        print("Sample unmatched models (left untouched):", flush=True)
        for ex in unmatched_examples:
            print(f"  {ex}", flush=True)

    if not proposed_updates:
        print("", flush=True)
        print("Nothing to update.", flush=True)
        return 0

    if args.dry_run:
        print("", flush=True)
        print(f"--dry-run: NOT writing {len(proposed_updates)} update(s).", flush=True)
        print("Re-run without --dry-run to apply.", flush=True)
        return 0

    print("", flush=True)
    print(f"Writing {len(proposed_updates)} update(s) to db.models...", flush=True)
    write_errors = 0
    for model_oid, job_id_str in proposed_updates:
        try:
            db.models.update_one(
                {"_id": model_oid},
                {"$set": {"job_id": job_id_str}})
        except Exception as e:  # noqa: BLE001
            write_errors += 1
            print(f"  ERROR updating {str(model_oid)}: {e}", flush=True)
    print(f"Done. {len(proposed_updates) - write_errors} succeeded, "
          f"{write_errors} failed.", flush=True)
    return 0 if write_errors == 0 else 1


def _to_aware(dt):
    """Convert a pymongo datetime (often tz-naive) to tz-aware UTC.

    Returns None if dt is not a datetime (defensive against legacy
    rows that stored timestamps as strings or numbers).
    """
    if not isinstance(dt, datetime.datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
