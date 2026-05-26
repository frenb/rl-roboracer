"""End-to-end tests for the Outcome ingester.

Real Mongo (no mocks - the outcome ingester is pure stats + DB I/O,
nothing external to fake). We insert synthetic proposals + jobs +
models with the "[outcome-test]" / "[outcome-test-job]" prefix
convention, run ingest_one across the interesting scenarios, and
clean up everything we created at the end.

Scenarios covered:

  1. Per-arm stat math sanity (bootstrap CI + mean + median + stddev)
  2. Criterion evaluation: parsed + criterion met (verdict=supported)
  3. Criterion evaluation: parsed + criterion missed (verdict=rejected)
  4. Criterion evaluation: no primary_parsed (verdict=inconclusive)
  5. Criterion evaluation: arm names don't match (verdict=inconclusive)
  6. Threshold kind=absolute
  7. Comparator variations (>, >=, <, <=)
  8. All-jobs-failed -> status=failed, verdict=inconclusive
  9. Non-terminal jobs -> ingest_one returns None (proposal stays in flight)
 10. Proposal at status=pending_user -> ingester refuses (not eligible)
 11. Re-running ingest_one on a done proposal is a no-op

Run via:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_outcome_ingester
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback
from typing import Any, Dict, List

from pymongo import MongoClient

from rl_agent.madscientist import constants, outcome_ingester
from rl_agent.madscientist.outcome_ingester import (
    _bootstrap_ci_95,
    _evaluate_criterion,
)


_passed = 0
_failed = 0
_failures: List[str] = []
_inserted_proposals: List[Any] = []
_inserted_jobs: List[Any] = []
_inserted_models: List[Any] = []


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


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


# ---- Synthetic data construction ----------------------------------------


def _insert_synthetic_job(db, arm: str, status: str = "DONE"):
    """Insert a synthetic TRAIN job + its is_global_best model.
    Returns the job's ObjectId."""
    now = _now()
    job_doc = {
        "status": status,
        "job_type": "TRAIN",
        "proposal_arm": arm,
        "started_at": now,
        "ended_at": now,
        "title": f"[outcome-test-job] arm={arm}",
    }
    job_id = db.jobs.insert_one(job_doc).inserted_id
    _inserted_jobs.append(job_id)
    return job_id


def _insert_synthetic_model(db, job_id, avg_return: float):
    """Insert an is_global_best model for the given job."""
    now = _now()
    model_doc = {
        "job_id": str(job_id),
        "create_date": now,
        "training_iterations": 5000,
        "avg_return": float(avg_return),
        "is_global_best": True,
        "robot_type": "robotaxi",
        "model_type": "SacAgent",
        "title": "[outcome-test-model]",
    }
    model_id = db.models.insert_one(model_doc).inserted_id
    _inserted_models.append(model_id)
    return model_id


def _insert_synthetic_proposal(
    db,
    *,
    arms_to_returns: Dict[str, List[float]],
    primary_parsed: Optional[Dict[str, Any]] = None,
    status: str = constants.STATUS_TRAINING,
    failed_arms: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Insert a synthetic proposal + all its TRAIN jobs + models.

    arms_to_returns: {arm_name -> [avg_return per seed]}
    primary_parsed: structured criterion (or None to leave as freetext only)
    status: initial proposal status
    failed_arms: optional {arm_name -> n_failed_jobs_to_add}, jobs
      inserted with status=FAILED + no model. Used to test partial-
      failure scenarios.

    Returns the inserted proposal doc (with _id).
    """
    now = _now()
    training_job_ids = []
    for arm, returns in arms_to_returns.items():
        for r in returns:
            jid = _insert_synthetic_job(db, arm=arm, status="DONE")
            _insert_synthetic_model(db, jid, avg_return=r)
            training_job_ids.append(jid)
    if failed_arms:
        for arm, n_fail in failed_arms.items():
            for _ in range(n_fail):
                jid = _insert_synthetic_job(db, arm=arm, status="FAILED")
                training_job_ids.append(jid)
    proposal_doc = {
        "title": "[outcome-test] synthetic proposal",
        "status": status,
        "created_at": now,
        "updated_at": now,
        "hypothesis": "test hypothesis",
        "experiment_arms": [{"name": a} for a in arms_to_returns.keys()],
        "n_seeds_per_arm": (max(len(v) for v in arms_to_returns.values())
                            if arms_to_returns else 1),
        "num_iterations_per_seed": 5000,
        "success_criteria": {
            "primary": "synthetic test criterion",
            "primary_parsed": primary_parsed,
            "secondary": [],
        },
        "training_job_ids": training_job_ids,
        "audit_events": [],
    }
    pid = db.proposals.insert_one(proposal_doc).inserted_id
    _inserted_proposals.append(pid)
    proposal_doc["_id"] = pid
    return proposal_doc


# Hack: Python 3.9-style Optional import in fixtures (we don't import
# typing.Optional at module top for cleaner test reading).
from typing import Optional  # noqa: E402


# ---- Tests ---------------------------------------------------------------


def test_bootstrap_ci_math():
    print("\nGroup 1: bootstrap CI + stats helpers", flush=True)

    # n=0 / n=1 yield (None, None)
    _expect(
        "n=0 samples -> (None, None)",
        lambda: _bootstrap_ci_95([]) == (None, None))
    _expect(
        "n=1 samples -> (None, None)",
        lambda: _bootstrap_ci_95([42.0]) == (None, None))

    # n>=2 yields a valid interval bracketing the sample mean.
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    lo, hi = _bootstrap_ci_95(samples)
    sample_mean = sum(samples) / len(samples)
    _expect(
        "n=5 samples produces valid CI",
        lambda: lo is not None and hi is not None and lo < hi)
    _expect(
        "CI brackets the sample mean (or is close)",
        lambda: lo <= sample_mean <= hi,
        f"lo={lo:.3f}, mean={sample_mean:.3f}, hi={hi:.3f}")


def test_evaluate_criterion():
    print("\nGroup 2: criterion evaluator", flush=True)

    # No primary_parsed -> parsed=False, criterion_met=None
    ev = _evaluate_criterion(None, {"base": 50.0, "exp1": 55.0})
    _expect(
        "no primary_parsed -> parsed=False",
        lambda: ev.parsed is False and ev.criterion_met is None)

    # Met: 10% improvement, threshold=5% relative
    ev = _evaluate_criterion(
        {"metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
         "comparator": ">=", "threshold": 0.05, "threshold_kind": "relative"},
        {"base": 50.0, "exp1": 55.0})
    _expect(
        "relative threshold met (10% improvement >= 5%)",
        lambda: ev.criterion_met is True and ev.delta == 5.0,
        f"delta={ev.delta}, eff={ev.effective_threshold}")

    # Not met: 2% improvement, threshold=5% relative
    ev = _evaluate_criterion(
        {"metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
         "comparator": ">=", "threshold": 0.05, "threshold_kind": "relative"},
        {"base": 50.0, "exp1": 51.0})
    _expect(
        "relative threshold missed (2% < 5%)",
        lambda: ev.criterion_met is False)

    # Absolute threshold
    ev = _evaluate_criterion(
        {"metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
         "comparator": ">=", "threshold": 5.0, "threshold_kind": "absolute"},
        {"base": 50.0, "exp1": 55.5})
    _expect(
        "absolute threshold met",
        lambda: ev.criterion_met is True and ev.effective_threshold == 5.0)

    # Unknown arm
    ev = _evaluate_criterion(
        {"metric": "avg_return", "arm_a": "ghost", "arm_b": "base",
         "comparator": ">=", "threshold": 0.05, "threshold_kind": "relative"},
        {"base": 50.0, "exp1": 55.0})
    _expect(
        "unknown arm -> parsed=False",
        lambda: ev.parsed is False)

    # Baseline mean = 0 -> falls back to absolute threshold
    ev = _evaluate_criterion(
        {"metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
         "comparator": ">=", "threshold": 1.0, "threshold_kind": "relative"},
        {"base": 0.0, "exp1": 1.5})
    _expect(
        "zero baseline -> absolute fallback, criterion met",
        lambda: ev.criterion_met is True and ev.effective_threshold == 1.0)


def test_ingest_one_supported():
    print("\nGroup 3: ingest_one - criterion supported (verdict=supported)",
          flush=True)
    db = _db()
    proposal = _insert_synthetic_proposal(
        db,
        arms_to_returns={
            "base": [10.0, 12.0, 14.0],
            "exp1": [20.0, 22.0, 24.0],  # +91% over base mean of 12
        },
        primary_parsed={
            "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
            "comparator": ">=", "threshold": 0.10,
            "threshold_kind": "relative"},
        status=constants.STATUS_TRAINING,
    )
    updated = outcome_ingester.ingest_one(db, proposal)
    _expect(
        "supported: ingest_one returns updated doc",
        lambda: updated is not None)
    _expect(
        "supported: status=done",
        lambda: updated["status"] == constants.STATUS_DONE,
        f"status={updated['status']}")
    _expect(
        "supported: results.verdict=supported",
        lambda: updated["results"]["verdict"] == constants.OUTCOME_SUPPORTED,
        f"verdict={updated['results']['verdict']}")
    _expect(
        "supported: per_arm has 2 entries",
        lambda: len(updated["results"]["per_arm"]) == 2)
    _expect(
        "supported: per_arm carries n_trials=3",
        lambda: all(a["n_trials"] == 3
                    for a in updated["results"]["per_arm"]))
    _expect(
        "supported: per_arm means roughly match input",
        lambda: any(abs(a["mean"] - 12.0) < 1e-6
                    for a in updated["results"]["per_arm"] if a["name"] == "base"))


def test_ingest_one_rejected():
    print("\nGroup 4: ingest_one - criterion missed (verdict=rejected)",
          flush=True)
    db = _db()
    proposal = _insert_synthetic_proposal(
        db,
        arms_to_returns={
            "base": [50.0, 52.0, 54.0],   # mean=52
            "exp1": [52.0, 51.0, 53.0],   # mean=52 (no improvement)
        },
        primary_parsed={
            "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
            "comparator": ">=", "threshold": 0.10,
            "threshold_kind": "relative"},
    )
    updated = outcome_ingester.ingest_one(db, proposal)
    _expect(
        "rejected: results.verdict=rejected",
        lambda: updated["results"]["verdict"] == constants.OUTCOME_REJECTED)
    _expect(
        "rejected: primary_criterion_met=False",
        lambda: updated["results"]["primary_criterion_met"] is False)


def test_ingest_one_no_parsed():
    print("\nGroup 5: ingest_one - no primary_parsed (verdict=inconclusive)",
          flush=True)
    db = _db()
    proposal = _insert_synthetic_proposal(
        db,
        arms_to_returns={
            "base": [10.0, 12.0],
            "exp1": [20.0, 22.0],
        },
        primary_parsed=None,
    )
    updated = outcome_ingester.ingest_one(db, proposal)
    _expect(
        "no parsed: verdict=inconclusive",
        lambda: updated["results"]["verdict"] == constants.OUTCOME_INCONCLUSIVE)
    _expect(
        "no parsed: criterion_met is None",
        lambda: updated["results"]["primary_criterion_met"] is None)
    _expect(
        "no parsed: per_arm still populated",
        lambda: len(updated["results"]["per_arm"]) == 2)


def test_ingest_one_all_failed():
    print("\nGroup 6: ingest_one - all jobs FAILED (verdict=inconclusive, status=failed)",
          flush=True)
    db = _db()
    # No successful jobs: only failed_arms entries.
    proposal = _insert_synthetic_proposal(
        db,
        arms_to_returns={},
        failed_arms={"base": 2, "exp1": 2},
    )
    updated = outcome_ingester.ingest_one(db, proposal)
    _expect(
        "all failed: status=failed",
        lambda: updated["status"] == constants.STATUS_FAILED)
    _expect(
        "all failed: verdict=inconclusive",
        lambda: updated["results"]["verdict"] == constants.OUTCOME_INCONCLUSIVE)
    _expect(
        "all failed: n_jobs_failed=4",
        lambda: updated["results"]["n_jobs_failed"] == 4)


def test_ingest_one_not_ready():
    print("\nGroup 7: ingest_one returns None for non-eligible proposals",
          flush=True)
    db = _db()

    # 7a: proposal at status=pending_user -> not in-flight, skip.
    p1 = _insert_synthetic_proposal(
        db,
        arms_to_returns={"base": [10.0], "exp1": [12.0]},
        status=constants.STATUS_PENDING_USER,
    )
    res = outcome_ingester.ingest_one(db, p1)
    _expect(
        "status=pending_user -> ingest_one returns None",
        lambda: res is None)

    # 7b: proposal with no training_job_ids -> skip.
    proposal_doc = {
        "title": "[outcome-test] no jobs",
        "status": constants.STATUS_TRAINING,
        "created_at": _now(), "updated_at": _now(),
        "hypothesis": "x", "experiment_arms": [],
        "n_seeds_per_arm": 1, "num_iterations_per_seed": 1,
        "success_criteria": {
            "primary": "x", "primary_parsed": None, "secondary": []},
        "training_job_ids": [],
        "audit_events": [],
    }
    pid = db.proposals.insert_one(proposal_doc).inserted_id
    _inserted_proposals.append(pid)
    proposal_doc["_id"] = pid
    res = outcome_ingester.ingest_one(db, proposal_doc)
    _expect(
        "empty training_job_ids -> ingest_one returns None",
        lambda: res is None)

    # 7c: proposal with a non-terminal job -> skip.
    p3 = _insert_synthetic_proposal(
        db,
        arms_to_returns={"base": [10.0, 12.0]},
        status=constants.STATUS_TRAINING,
    )
    # Add an IN_PROGRESS job to the mix.
    ip_job = {
        "status": "IN_PROGRESS",
        "job_type": "TRAIN",
        "proposal_arm": "base",
        "started_at": _now(),
        "title": "[outcome-test-job] in progress",
    }
    ip_job_id = db.jobs.insert_one(ip_job).inserted_id
    _inserted_jobs.append(ip_job_id)
    db.proposals.update_one(
        {"_id": p3["_id"]},
        {"$push": {"training_job_ids": ip_job_id}})
    p3 = db.proposals.find_one({"_id": p3["_id"]})
    res = outcome_ingester.ingest_one(db, p3)
    _expect(
        "in-progress job present -> ingest_one returns None",
        lambda: res is None)


def test_ingest_one_idempotent_on_done():
    print("\nGroup 8: ingest_one on already-done proposal -> no-op",
          flush=True)
    db = _db()
    # Insert proposal, ingest it to done, then re-ingest. Should
    # return None on the second call because status=done is not
    # in the eligible set.
    proposal = _insert_synthetic_proposal(
        db,
        arms_to_returns={"base": [10.0, 12.0], "exp1": [20.0, 22.0]},
        primary_parsed={
            "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
            "comparator": ">=", "threshold": 0.10,
            "threshold_kind": "relative"},
    )
    first = outcome_ingester.ingest_one(db, proposal)
    _expect(
        "first ingest -> done",
        lambda: first is not None and first["status"] == constants.STATUS_DONE)
    second = outcome_ingester.ingest_one(db, first)
    _expect(
        "second ingest -> None (no double-update)",
        lambda: second is None)


# ---- Comparator coverage ------------------------------------------------


def test_comparators():
    print("\nGroup 9: comparator coverage (<, <=, >, >=)", flush=True)
    # delta=5, baseline=50 -> effective relative threshold of 10%=5
    # so delta>=5 is True, delta>5 is False, etc.
    base_data = {"base": 50.0, "exp1": 55.0}
    parsed = {
        "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
        "threshold": 0.10, "threshold_kind": "relative",
    }
    for comp, expected in [(">=", True), (">", False),
                           ("<=", True), ("<", False)]:
        ev = _evaluate_criterion({**parsed, "comparator": comp}, base_data)
        _expect(
            f"comparator={comp} with delta == threshold",
            lambda e=expected, ev=ev: ev.criterion_met is e,
            f"got {ev.criterion_met}")


# ---- Cleanup + main -----------------------------------------------------


def _cleanup(db):
    if _inserted_proposals:
        db.proposals.delete_many({"_id": {"$in": _inserted_proposals}})
    if _inserted_jobs:
        db.jobs.delete_many({"_id": {"$in": _inserted_jobs}})
    if _inserted_models:
        db.models.delete_many({"_id": {"$in": _inserted_models}})
    print(
        f"\n  cleanup: removed {len(_inserted_proposals)} proposals, "
        f"{len(_inserted_jobs)} jobs, {len(_inserted_models)} models",
        flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/outcome_ingester.py", flush=True)
    print("=" * 64, flush=True)

    db = _db()
    try:
        test_bootstrap_ci_math()
        test_evaluate_criterion()
        test_ingest_one_supported()
        test_ingest_one_rejected()
        test_ingest_one_no_parsed()
        test_ingest_one_all_failed()
        test_ingest_one_not_ready()
        test_ingest_one_idempotent_on_done()
        test_comparators()
    finally:
        _cleanup(db)

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
