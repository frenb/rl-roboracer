#!/usr/bin/env python3
"""Backfill the `is_global_best` flag on existing model records.

Background
----------
Before 2026-05-25, the trainer's `max_avg_return` was re-initialized to
0.0 at the top of every main() call - including on pause/resume picks
and crash-recovery picks. That meant every resume slice's FIRST eval
"won" the new-max gate and got saved as a model, even if it was below
the historical best for the same job_id.

The fix (see commit "rl_agent: resume-aware max_avg_return seeding +
is_global_best flag") seeds max_avg_return from existing Mongo records
on resume, so going forward every saved model is a true new global
best AND carries is_global_best=True. Historical models predating that
commit don't have the field.

What this script does
---------------------
For each job_id, finds the model with the maximum avg_return and
stamps is_global_best=True on it; every other model for that job_id
gets is_global_best=False. Models with avg_return=None or no job_id
are left untouched (they don't fit the per-job-best semantics).

Idempotent: re-running just rewrites the same values. Safe.

Usage (from the host)
---------------------
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        sim-controller python /python_ws/src/backfill_global_best.py

Add --dry-run to see what would change without writing.
"""
import argparse
import sys

from pymongo import MongoClient


def _open_db():
    # Same connection params the trainer uses (see robotaxi.py top-of-
    # file). When invoked inside the sim-controller container this
    # resolves "mongo" via Docker's service alias.
    client = MongoClient(
        "mongo",
        username="root",
        password="example",
        authSource="admin",
        serverSelectionTimeoutMS=10000)
    # Ping so we fail fast rather than hanging on the first query.
    client.admin.command("ping")
    return client.robotaxi


def backfill(db, dry_run):
    # Index all models by job_id. We skip models with no job_id
    # (legacy / pre-feature) and models with no numeric avg_return
    # (incomplete records).
    by_job = {}
    skipped_no_job = 0
    skipped_no_return = 0
    for m in db.models.find(
            {},
            {"_id": 1, "job_id": 1, "avg_return": 1, "is_global_best": 1}):
        job_id = m.get("job_id")
        if not job_id:
            skipped_no_job += 1
            continue
        v = m.get("avg_return")
        if v is None:
            skipped_no_return += 1
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            skipped_no_return += 1
            continue
        by_job.setdefault(job_id, []).append((m["_id"], v, m.get("is_global_best")))

    print(f"Scanned {sum(len(v) for v in by_job.values())} models across "
          f"{len(by_job)} job_ids "
          f"(skipped: {skipped_no_job} with no job_id, "
          f"{skipped_no_return} with no numeric avg_return).")

    # For each job, find the best record. Tie-break by inserting the
    # LATEST-created tied record as the global-best (assumes _id is
    # monotonically increasing, which Mongo ObjectIds are).
    set_true = 0
    set_false = 0
    already_correct = 0
    for job_id, rows in by_job.items():
        # Sort by avg_return desc, then _id desc (later tie-break).
        rows.sort(key=lambda r: (r[1], str(r[0])), reverse=True)
        best_id = rows[0][0]
        for model_id, _, current in rows:
            desired = (model_id == best_id)
            if current == desired:
                already_correct += 1
                continue
            if dry_run:
                pass
            else:
                db.models.update_one(
                    {"_id": model_id},
                    {"$set": {"is_global_best": bool(desired)}})
            if desired:
                set_true += 1
            else:
                set_false += 1

    label = "would set" if dry_run else "set"
    print(f"{label} is_global_best=True on {set_true} record(s).")
    print(f"{label} is_global_best=False on {set_false} record(s).")
    print(f"Left {already_correct} record(s) already-correct unchanged.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the proposed updates without writing.")
    args = p.parse_args()

    try:
        db = _open_db()
    except Exception as e:  # noqa: BLE001
        print(f"Could not connect to Mongo: {e}", file=sys.stderr)
        sys.exit(2)

    backfill(db, args.dry_run)


if __name__ == "__main__":
    main()
