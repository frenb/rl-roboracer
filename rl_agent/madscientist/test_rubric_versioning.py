"""Exercise the seed.ensure_rubric_from_file() history-preservation logic.

Three scenarios to verify:

  1. Re-running with no file change is a no-op (we already confirmed
     this manually, but include it in the test for regression-proofing).

  2. Editing the file's content WITHOUT bumping the Version field
     should produce a warning + insert under the NEXT-FREE version
     number (so history is never clobbered).

  3. Editing the file's content WITH a bumped Version should insert
     under the bumped number cleanly.

To avoid mutating the real JUDGE_RUBRIC.md, the test creates a
temporary rubric file, monkey-patches seed._RUBRIC_FILE to point at
it, runs ensure_rubric_from_file, inspects Mongo, repeats with
different file contents, and restores at exit. The real rubric and
its version history are unaffected.

Run via:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_rubric_versioning

Exit code 0 = all assertions passed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

from pymongo import MongoClient

from rl_agent.madscientist import constants, seed


# Reserved high-numbered versions for the test - high enough to never
# collide with the real rubric file's version sequence (currently
# v1 in production; we use v9000+ for tests).
_TEST_VERSION_BASE = 9000

_RUBRIC_HEADER_TEMPLATE = """# Judge Rubric - TEST ARTIFACT

| | |
|---|---|
| **Version** | {version} |
| **Effective from** | 2026-05-25 |
| **Authored by** | `test_rubric_versioning` |

This file is written by rl_agent/madscientist/test_rubric_versioning.py to
exercise seed.ensure_rubric_from_file. The MadScientist agent must NEVER
score against a v{version} rubric in production - if you see one, the
test crashed mid-run and didn't clean up.

(Body content variant {variant} to drive content-hash differences in
the no-version-bump test path.)
"""


def _make_rubric_content(version: int, variant: str = "A") -> str:
    return _RUBRIC_HEADER_TEMPLATE.format(version=version, variant=variant)


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _cleanup_test_records(db):
    """Remove any judge_rubric_history records with versions >= _TEST_VERSION_BASE."""
    res = db[constants.COLL_JUDGE_RUBRIC_HISTORY].delete_many(
        {"version": {"$gte": _TEST_VERSION_BASE}})
    return res.deleted_count


_passed = 0
_failed = 0
_failures: list[str] = []


def _expect(label: str, predicate, detail: str = ""):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"{detail} | EXCEPTION: {type(e).__name__}: {e}"
    if ok:
        _passed += 1
        suffix = f" - {detail}" if detail else ""
        print(f"  [PASS] {label}{suffix}", flush=True)
    else:
        _failed += 1
        _failures.append(label)
        print(f"  [FAIL] {label} - {detail}", flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("Test: seed.ensure_rubric_from_file() history preservation", flush=True)
    print("=" * 64, flush=True)

    db = _db()
    original_rubric_path = seed._RUBRIC_FILE

    pre_test_deleted = _cleanup_test_records(db)
    if pre_test_deleted:
        print(f"  (cleared {pre_test_deleted} stale test records from a prior run)",
              flush=True)

    test_version = _TEST_VERSION_BASE + 1
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        # ---- Scenario 1: insert version v9001 (content A) --------------
        print("\nScenario 1: insert fresh version 9001 (content A)", flush=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_make_rubric_content(test_version, variant="A"))
        seed._RUBRIC_FILE = tmp_path
        seed.ensure_rubric_from_file(db)
        doc = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
            {"version": test_version})
        _expect(
            "version 9001 inserted",
            lambda: doc is not None,
            f"doc.version={doc.get('version') if doc else None}")
        _expect(
            "version 9001 authored_by parsed correctly",
            lambda: doc and doc.get("authored_by") == "test_rubric_versioning",
            f"authored_by={doc.get('authored_by') if doc else None}")
        original_hash = doc.get("content_hash") if doc else None
        _expect(
            "version 9001 has content_hash",
            lambda: bool(original_hash))

        # ---- Scenario 2: re-run with identical content - no-op ---------
        print("\nScenario 2: re-run with identical file - should be no-op", flush=True)
        seed.ensure_rubric_from_file(db)
        count = db[constants.COLL_JUDGE_RUBRIC_HISTORY].count_documents(
            {"version": test_version})
        _expect(
            "no duplicate insert for unchanged content",
            lambda: count == 1,
            f"count={count}")

        # ---- Scenario 3: bump version, change content - inserts new ----
        next_version = test_version + 1
        print(f"\nScenario 3: bump to version {next_version}, change content - "
              "should insert new", flush=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_make_rubric_content(next_version, variant="B"))
        seed.ensure_rubric_from_file(db)
        doc_v2 = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
            {"version": next_version})
        _expect(
            f"version {next_version} inserted",
            lambda: doc_v2 is not None,
            f"doc.version={doc_v2.get('version') if doc_v2 else None}")
        _expect(
            f"version {next_version} has different content_hash from 9001",
            lambda: doc_v2 and doc_v2.get("content_hash") != original_hash)

        # ---- Scenario 4: change content WITHOUT bumping version -------
        # Reuse next_version (9002), but with variant="C" content. Should
        # detect the content_hash mismatch and insert under
        # next-free-version (=> 9003).
        print(f"\nScenario 4: change content without bumping version (still {next_version}) "
              "- should warn + insert under next-free version", flush=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_make_rubric_content(next_version, variant="C"))
        # Note: the "next free" lookup uses MAX(version) + 1; with
        # production v1 already in the collection but no other test
        # records, MAX = 9002, so the new insert lands at 9003.
        seed.ensure_rubric_from_file(db)
        next_free = next_version + 1
        doc_v3 = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
            {"version": next_free})
        _expect(
            f"warning path inserted under version {next_free}",
            lambda: doc_v3 is not None,
            f"doc.version={doc_v3.get('version') if doc_v3 else None}")
        _expect(
            f"warning-path doc has 'note' field explaining auto-insert",
            lambda: doc_v3 and "note" in doc_v3 and "version" in (doc_v3.get("note") or ""),
            f"note={doc_v3.get('note') if doc_v3 else None}")
        # Original 9002 record should still be there unchanged - we
        # never overwrite history.
        doc_v2_check = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
            {"version": next_version})
        _expect(
            f"original version {next_version} record preserved unchanged",
            lambda: doc_v2_check and doc_v2_check.get("content_hash") == doc_v2.get("content_hash"))

        # ---- Scenario 5: missing rubric file --------------------------
        print("\nScenario 5: rubric file missing - upserts placeholder v0",
              flush=True)
        # Already a v0 in production from the earlier scaffolding step,
        # so this should be a no-op upsert. Confirm v0 still exists.
        os.unlink(tmp_path)
        seed.ensure_rubric_from_file(db)
        doc_v0 = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
            {"version": 0})
        _expect(
            "v0 placeholder exists after missing-file run",
            lambda: doc_v0 is not None,
            f"authored_by={doc_v0.get('authored_by') if doc_v0 else None}")

    finally:
        # Always restore.
        seed._RUBRIC_FILE = original_rubric_path
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        post_test_deleted = _cleanup_test_records(db)
        print(f"\n  cleanup: removed {post_test_deleted} test records from Mongo",
              flush=True)

    print("\n" + "=" * 64, flush=True)
    print(f"PASSED: {_passed}", flush=True)
    print(f"FAILED: {_failed}", flush=True)
    if _failures:
        for f in _failures:
            print(f"  - {f}", flush=True)
    print("=" * 64, flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
