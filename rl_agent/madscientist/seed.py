"""Idempotent Mongo bootstrap for the MadScientist agent.

Creates the three new collections (proposals, research_notes,
judge_rubric_history), creates their indexes, and adds sparse
proposal_id indexes to the existing jobs/models/experiment_designs
collections so the join from "proposal -> downstream artifact" is
fast.

ALSO upserts a placeholder JudgeRubricVersion (version=0) so the
dashboard tab has something to render before the deep-research pass
authors the real rubric in JUDGE_RUBRIC.md.

Safe to run repeatedly. Each create_index() / update_one(upsert=True)
is itself idempotent. Run from the madscientist container's
entrypoint.sh on every start.

Usage (host):
    docker compose run --rm madscientist python /repo/rl_agent/madscientist/seed.py

The container's entrypoint also calls this on startup so manual
invocation is only needed for ad-hoc maintenance.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import re
import sys

import pymongo
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from . import constants


_RUBRIC_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "JUDGE_RUBRIC.md")


def _get_db():
    """Open a Mongo connection using the same env-var convention the
    dashboard server uses (MONGO_URL fallback to the hardcoded compose
    DSN). Ping fail-fast so misconfigured deployments error here
    instead of later mid-cycle.
    """
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(
        url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _ensure_index(coll, keys, *, name=None, **kwargs):
    """Wrapper around create_index that prints what it did + is loud
    on errors (rather than swallowing them like create_index can).
    """
    full_name = name or "_".join(f"{k}_{d}" for k, d in keys)
    try:
        coll.create_index(keys, name=full_name, **kwargs)
        print(f"  ok: {coll.name}.{full_name}", flush=True)
    except OperationFailure as e:
        # Most common cause: an existing index with the same keys but
        # different options. We don't auto-drop because that's a footgun;
        # the operator can drop manually if needed.
        print(
            f"  WARN: failed to create {coll.name}.{full_name}: {e}. "
            f"Continuing; check db.{coll.name}.getIndexes() for conflict.",
            flush=True)


def ensure_collections_and_indexes(db):
    """Create / ensure each new collection + its indexes. Idempotent."""
    print(f"Ensuring MadScientist collections in db.{db.name}...", flush=True)

    # ---- Proposals ------------------------------------------------------
    # Dashboard pending-list query (status + recency).
    _ensure_index(
        db[constants.COLL_PROPOSALS],
        [("status", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)],
        name="status_1_created_at_-1")
    # Outcome ingester key: find proposals whose linked TRAIN jobs all
    # reached DONE. Sparse because most proposals have empty
    # training_job_ids until implementation completes.
    _ensure_index(
        db[constants.COLL_PROPOSALS],
        [("training_job_ids", pymongo.ASCENDING)],
        name="training_job_ids_1",
        sparse=True)
    # Decision history queries (for the dashboard's "Outcomes" table +
    # future meta-analysis worker).
    _ensure_index(
        db[constants.COLL_PROPOSALS],
        [("decision.action", pymongo.ASCENDING),
         ("decision.at", pymongo.DESCENDING)],
        name="decision_action_1_decision_at_-1",
        sparse=True)

    # ---- Research notes -------------------------------------------------
    _ensure_index(
        db[constants.COLL_RESEARCH_NOTES],
        [("cycle_id", pymongo.ASCENDING)],
        name="cycle_id_1")
    _ensure_index(
        db[constants.COLL_RESEARCH_NOTES],
        [("source_type", pymongo.ASCENDING),
         ("at", pymongo.DESCENDING)],
        name="source_type_1_at_-1")
    # Dedup index: an arxiv abstract or page URL shouldn't be ingested
    # twice in different cycles. Unique-sparse so notes without a
    # source_ref are still allowed (we don't expect any but the schema
    # leaves it Optional[str] for forward compat).
    _ensure_index(
        db[constants.COLL_RESEARCH_NOTES],
        [("source_ref", pymongo.ASCENDING)],
        name="source_ref_1",
        sparse=True)

    # ---- Judge rubric history -------------------------------------------
    _ensure_index(
        db[constants.COLL_JUDGE_RUBRIC_HISTORY],
        [("version", pymongo.DESCENDING)],
        name="version_-1",
        unique=True)

    # ---- Extensions on existing collections -----------------------------
    # Sparse so legacy rows (no proposal_id) don't bloat the index.
    print(f"Ensuring sparse proposal_id indexes on existing collections...", flush=True)
    for coll_name in ("jobs", "models", "experiment_designs"):
        if coll_name not in db.list_collection_names():
            # Existing collection may not exist on a fresh Mongo (the
            # trainer creates them on first write). Skip rather than
            # auto-create; we don't want this seed to pre-empt the
            # trainer's writeConcerns.
            print(
                f"  skip: {coll_name} (doesn't exist yet; index will be "
                f"created the next time this script runs after the trainer "
                f"has written at least one row to that collection).",
                flush=True)
            continue
        _ensure_index(
            db[coll_name],
            [("proposal_id", pymongo.ASCENDING)],
            name="proposal_id_1",
            sparse=True)


def _parse_rubric_version(md):
    """Pull the 'Version' integer from the rubric file's front-matter table.

    The file's first table looks like:
        | **Version** | 1 |
    We grep for that pattern. Fall back to None if the file is malformed -
    we'd rather skip the rubric ingest than insert with a bogus version.
    """
    m = re.search(
        r"^\|\s*\*\*Version\*\*\s*\|\s*(\d+)\s*\|", md, re.MULTILINE)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _parse_rubric_authored_by(md):
    """Pull the 'Authored by' field from the rubric file's front-matter
    table. Used to stamp judge_rubric_history.authored_by.
    """
    m = re.search(
        r"^\|\s*\*\*Authored by\*\*\s*\|\s*`?([^`|]+?)`?\s*(?:\([^)]*\))?\s*\|",
        md, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "unknown"


def ensure_rubric_from_file(db):
    """Read madscientist/JUDGE_RUBRIC.md (if it exists) and upsert it
    into db.judge_rubric_history.

    Behavior:
      * Rubric file missing -> upsert version=0 placeholder so the
        downstream Judge worker has SOMETHING to read (with content
        explaining the rubric isn't drafted yet).
      * Rubric file exists -> parse its Version + content. If that
        (Version, content_hash) tuple isn't yet in the collection,
        insert a new judge_rubric_history record. Idempotent: re-running
        with no file change is a no-op.
      * If two different content hashes claim the same Version, we log
        a warning (the operator should bump the version after edits)
        but still insert under the next-free version number rather than
        clobbering history.
    """
    print(f"Ensuring rubric from {_RUBRIC_FILE}...", flush=True)
    now = datetime.datetime.now(datetime.timezone.utc)

    if not os.path.exists(_RUBRIC_FILE):
        # Placeholder for environments without the rubric yet.
        print(
            f"  rubric file not found - inserting placeholder v0.",
            flush=True)
        placeholder_md = (
            "# Judge Rubric - PLACEHOLDER\n\n"
            "Rubric file rl_agent/madscientist/JUDGE_RUBRIC.md is missing\n"
            "from this container's view of the repo. The Judge worker\n"
            "should refuse to score until the file is present.\n")
        db[constants.COLL_JUDGE_RUBRIC_HISTORY].update_one(
            {"version": 0},
            {"$setOnInsert": {
                "version": 0,
                "effective_from": now,
                "rubric_markdown": placeholder_md,
                "rubric_axes": [],
                "authored_by": "scaffolding_placeholder",
                "git_sha": None,
            }},
            upsert=True)
        return

    with open(_RUBRIC_FILE, "r", encoding="utf-8") as f:
        md = f.read()

    version = _parse_rubric_version(md)
    if version is None:
        print(
            f"  WARNING: could not parse Version from {_RUBRIC_FILE}; "
            f"skipping rubric ingest. Add a '| **Version** | <int> |' row "
            f"to the front-matter table.",
            flush=True)
        return

    content_hash = hashlib.sha256(md.encode("utf-8")).hexdigest()[:16]
    authored_by = _parse_rubric_authored_by(md)

    existing = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
        {"version": version})
    if existing is None:
        # Fresh version; insert.
        db[constants.COLL_JUDGE_RUBRIC_HISTORY].insert_one({
            "version": version,
            "effective_from": now,
            "rubric_markdown": md,
            "rubric_axes": [],
            "authored_by": authored_by,
            "content_hash": content_hash,
            "git_sha": None,
        })
        print(
            f"  inserted new rubric version={version} "
            f"(authored_by='{authored_by}', "
            f"content_hash={content_hash[:8]}...).",
            flush=True)
        return

    existing_hash = existing.get("content_hash")
    if existing_hash == content_hash:
        print(
            f"  rubric version={version} already current "
            f"(content_hash={content_hash[:8]}...; no-op).",
            flush=True)
        return

    # Same version number, different content. This means the operator
    # edited the file but forgot to bump the version. We DON'T overwrite
    # historical records - we insert under the next-free version + warn.
    max_existing = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
        sort=[("version", pymongo.DESCENDING)])
    next_version = (max_existing.get("version", 0) + 1) if max_existing else version + 1
    print(
        f"  WARNING: rubric file version={version} has changed content but "
        f"the version number wasn't bumped. Inserting under version="
        f"{next_version} to preserve history. Please update the file's "
        f"front-matter '| **Version** |' row to {next_version} so the next "
        f"seed run is clean.",
        flush=True)
    db[constants.COLL_JUDGE_RUBRIC_HISTORY].insert_one({
        "version": next_version,
        "effective_from": now,
        "rubric_markdown": md,
        "rubric_axes": [],
        "authored_by": authored_by,
        "content_hash": content_hash,
        "git_sha": None,
        "note": (f"Auto-inserted because file claimed version={version} "
                 f"but content_hash differed from existing record."),
    })


def main():
    try:
        db = _get_db()
    except Exception as e:  # noqa: BLE001
        print(f"Could not connect to Mongo: {e}", file=sys.stderr)
        sys.exit(2)
    ensure_collections_and_indexes(db)
    ensure_rubric_from_file(db)
    print("MadScientist Mongo seed complete.", flush=True)


if __name__ == "__main__":
    main()
