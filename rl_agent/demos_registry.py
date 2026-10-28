"""Expert-demonstration registry backing the dashboard's Demos tab.

One ``demos`` document per expert-demo TFRecords directory:

    source_job_id  : DEMO job that collected it (absent for hand-registered dirs)
    num_demos      : episodes in the directory
    num_steps      : transitions (TFRecord rows) in the directory
    course         : course the demos were recorded on
    description    : free text
    tfrecords_path : container path the trainer reads, e.g. /tfrecords/job_<id>
    gym_name       : Unity gym the DEMO job ran against, if any
    collected_at   : when collection finished (newest TFRecord file's mtime)
    archived, created_at, updated_at

``num_demos`` / ``course`` / ``description`` / ``tfrecords_path`` are
user-editable on the Demos tab, so registration only ever writes them on
insert (``$setOnInsert``) and never clobbers an edit.

A TRAIN or BC_TRAINING_ONLY job that picks a demo on the New-job form
carries its path in ``demo_tfrecords_paths`` (see do_job).

Backfill the DEMO jobs that finished before this registry existed (run in
the sim-controller container, which has TF + the /tfrecords mount):

    docker compose exec sim-controller python /python_ws/src/demos_registry.py --dry-run
    docker compose exec sim-controller python /python_ws/src/demos_registry.py

Idempotent: jobs that already have a demos document are skipped, apart from
filling in a missing ``collected_at``.
"""

import argparse
import datetime
import os
import sys

# tf_agents StepType.LAST. Hardcoded so this module imports without tf_agents.
_STEP_TYPE_LAST = 2


def tfrecords_dir_for_job(job_id):
    return "/tfrecords/job_" + str(job_id)


def count_demos(path):
    """Return (episodes, steps) for every TFRecord file in ``path``.

    Episodes are the rows whose ``next_step_type`` is LAST. Records written
    before that feature existed only carry ``discount``, which is 0 on a
    crash, so for those directories the count is crash-ended episodes and
    misses any episode that ended on the goal cap.
    """
    import tensorflow as tf

    if not os.path.isdir(path):
        return 0, 0
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)))
    if not files:
        return 0, 0
    spec = {
        "next_step_type": tf.io.VarLenFeature(tf.int64),
        "discount": tf.io.VarLenFeature(tf.float32),
    }
    ds = (tf.data.TFRecordDataset(files)
          .batch(8192)
          .map(lambda s: (tf.shape(s)[0], tf.io.parse_example(s, spec))))
    steps = lasts = has_step_type = zero_discounts = 0
    for n, parsed in ds:
        steps += int(n)
        nst = parsed["next_step_type"].values
        has_step_type += int(tf.size(nst))
        lasts += int(tf.reduce_sum(tf.cast(tf.equal(nst, _STEP_TYPE_LAST), tf.int32)))
        zero_discounts += int(tf.reduce_sum(
            tf.cast(tf.equal(parsed["discount"].values, 0.0), tf.int32)))
    return (lasts if has_step_type else zero_discounts), steps


def collection_completed_at(path):
    """UTC mtime of the newest file in ``path``, or None if it has none.

    collect_expert_demos flushes its last TFRecord file when collection ends,
    so this is the collection's completion time - unlike the job's ended_at,
    which also covers the BC training a DEMO job runs afterwards.
    """
    if not os.path.isdir(path):
        return None
    mtimes = [os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)
              if os.path.isfile(os.path.join(path, f))]
    if not mtimes:
        return None
    return datetime.datetime.fromtimestamp(max(mtimes), datetime.timezone.utc)


def _default_description(job):
    parts = [p for p in (job.get("gym_name"), job.get("experiment_design_name")) if p]
    return " / ".join(parts) or ("DEMO job " + str(job["_id"]))


def register_demo_job(db, job, course=None, path=None, counts=None):
    """Upsert the demos document for a finished DEMO job.

    ``counts`` is (episodes, steps); counted from disk when omitted.
    Returns the document as stored.
    """
    path = path or tfrecords_dir_for_job(job["_id"])
    episodes, steps = counts or count_demos(path)
    now = datetime.datetime.now(datetime.timezone.utc)
    course = course or job.get("course_type") or "donut"
    if course == "trainer default":
        course = os.environ.get("ROBOTAXI_COURSE_TYPE", "donut")
    db.demos.update_one(
        {"source_job_id": str(job["_id"])},
        {"$setOnInsert": {
            "source_job_id": str(job["_id"]),
            "num_demos": episodes,
            "num_steps": steps,
            "course": course,
            "description": _default_description(job),
            "tfrecords_path": path,
            "gym_name": job.get("gym_name") or "",
            "collected_at": collection_completed_at(path),
            "archived": False,
            # Pre-2026 jobs have no ended_at; the ObjectId's creation time
            # is the closest thing they carry.
            "created_at": (job.get("ended_at")
                           or getattr(job["_id"], "generation_time", None)
                           or now),
            "updated_at": now,
        }},
        upsert=True)
    return db.demos.find_one({"source_job_id": str(job["_id"])})


def _parse_args():
    p = argparse.ArgumentParser(
        description="Register finished DEMO jobs in the demos collection.")
    p.add_argument("--dry-run", action="store_true",
                   help="Count and print, but don't write to Mongo.")
    p.add_argument(
        "--mongo-url",
        default=os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))
    p.add_argument("--db-name", default=os.environ.get("DATABASE_NAME", "robotaxi"))
    return p.parse_args()


def main():
    from pymongo import MongoClient

    args = _parse_args()
    db = MongoClient(args.mongo_url)[args.db_name]
    # collected_at is computed, never user-edited, so fill it in on rows
    # registered before the field existed (or by hand via /add_demo, since
    # the dashboard container can't see /tfrecords).
    for d in db.demos.find({"collected_at": None}, {"tfrecords_path": 1}):
        ts = collection_completed_at(d.get("tfrecords_path") or "")
        if ts is None:
            continue
        print(f"  collected_at {d['_id']}: {ts.isoformat()}", flush=True)
        if not args.dry_run:
            db.demos.update_one({"_id": d["_id"]}, {"$set": {"collected_at": ts}})
    registered = {d.get("source_job_id") for d in db.demos.find({}, {"source_job_id": 1})}
    jobs = list(db.jobs.find({"job_type": "DEMO", "status": "DONE"}).sort("_id", 1))
    added = 0
    for job in jobs:
        jid = str(job["_id"])
        path = tfrecords_dir_for_job(jid)
        if jid in registered:
            continue
        if not os.path.isdir(path):
            print(f"  skip {jid}: {path} not found", flush=True)
            continue
        episodes, steps = count_demos(path)
        if not steps:
            print(f"  skip {jid}: {path} has no records", flush=True)
            continue
        if args.dry_run:
            print(f"  would add {jid}: {episodes} demos, {steps} steps, {path}", flush=True)
            continue
        doc = register_demo_job(db, job, path=path, counts=(episodes, steps))
        added += 1
        print(f"  added {jid}: {doc['num_demos']} demos, {doc['num_steps']} steps, "
              f"{path}", flush=True)
    print(f"{len(jobs)} DONE DEMO jobs, {added} newly registered.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
