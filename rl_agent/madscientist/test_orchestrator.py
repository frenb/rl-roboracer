"""End-to-end tests for the orchestrator (Phase 1C-MVP).

Real Mongo. The orchestrator is pure DB-I/O - no external services,
no LLM - so tests run fast + don't need mocks.

Scenarios covered:

  1. Plain approved proposal -> jobs queued, status=training,
     training_job_ids populated, audit event stamped.
  2. Arm with experiment_design_fields -> derived design seeded in
     db.experiment_designs with provenance fields, job references
     its ObjectId.
  3. Arm with only experiment_design_id (no overlay) -> job uses
     that id directly, no new design created.
  4. Multiple arms x seeds -> N jobs queued, all stamped with
     proposal_id + proposal_arm.
  5. Reject inline reward_design_fields -> status=failed with clear
     reason.
  6. Reject too-many-jobs (exceeds max_jobs cap) -> status=failed.
  7. Reject empty experiment_arms -> not eligible (returns None
     without flipping status).
  8. Reject status != approved -> orchestrator returns None
     immediately.
  9. Idempotent: re-running on a proposal that already transitioned
     out of approved returns None.
 10. Outcome ingester picks up the queued jobs once we mark them
     DONE + insert synthetic models (verifies the end-to-end
     1C -> 1E linkage).

Run via:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_orchestrator
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from rl_agent.madscientist import (
    constants, orchestrator, outcome_ingester)


_passed = 0
_failed = 0
_failures: List[str] = []
_inserted_proposals: List[Any] = []
_inserted_jobs: List[Any] = []
_inserted_models: List[Any] = []
_inserted_designs: List[Any] = []


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


def _insert_proposal(
    db, *,
    arms: List[Dict[str, Any]],
    status: str = constants.STATUS_APPROVED,
    n_seeds_per_arm: int = 2,
    num_iterations_per_seed: int = 5000,
    title_suffix: str = "",
) -> Dict[str, Any]:
    """Insert a synthetic proposal at the requested status. Returns
    the inserted doc with _id populated."""
    now = _now()
    doc = {
        "title": f"[orchestrator-test] {title_suffix}".strip(),
        "status": status,
        "created_at": now,
        "updated_at": now,
        "hypothesis": "test hypothesis",
        "experiment_arms": arms,
        "n_seeds_per_arm": n_seeds_per_arm,
        "num_iterations_per_seed": num_iterations_per_seed,
        "success_criteria": {
            "primary": "x", "primary_parsed": None, "secondary": []},
        "audit_events": [],
        "training_job_ids": [],
    }
    pid = db.proposals.insert_one(doc).inserted_id
    _inserted_proposals.append(pid)
    doc["_id"] = pid
    return doc


# ---- Tests ---------------------------------------------------------------


def test_plain_approved_proposal():
    print("\nGroup 1: plain approved proposal queues jobs + advances to training",
          flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1"},
    ], n_seeds_per_arm=2, title_suffix="plain")
    res = orchestrator.orchestrate_one(db, p)
    _expect(
        "orchestrate_one returned updated doc",
        lambda: res is not None)
    _expect(
        "status=training",
        lambda: res["status"] == constants.STATUS_TRAINING,
        f"status={res['status']}")
    _expect(
        "4 jobs queued (2 arms x 2 seeds)",
        lambda: len(res["training_job_ids"]) == 4)
    # Verify each job has proposal_id + proposal_arm stamped.
    for jid in res["training_job_ids"]:
        _inserted_jobs.append(jid)
    jobs = list(db.jobs.find({"_id": {"$in": res["training_job_ids"]}}))
    _expect(
        "all jobs have proposal_id stamped",
        lambda: all(j["proposal_id"] == str(p["_id"]) for j in jobs))
    _expect(
        "jobs have proposal_arm in {base, exp1}",
        lambda: {j["proposal_arm"] for j in jobs} == {"base", "exp1"})
    _expect(
        "all jobs have experiment_design_name stamped (so the Jobs tab "
        "Experiment design column renders a label)",
        lambda: all(j.get("experiment_design_name") for j in jobs),
        f"got names={[j.get('experiment_design_name') for j in jobs]!r}")
    _expect(
        "jobs have status=NOT_STARTED",
        lambda: all(j["status"] == "NOT_STARTED" for j in jobs))
    _expect(
        "audit_event 'queued_training_jobs' stamped",
        lambda: any(e.get("event") == "queued_training_jobs"
                    for e in res["audit_events"]))


def test_arm_with_overlay():
    print("\nGroup 2: arm with experiment_design_fields seeds a derived "
          "design with provenance", flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1", "experiment_design_fields": {"gamma": 0.95, "batch_size": 256}},
    ], n_seeds_per_arm=1, title_suffix="overlay")
    res = orchestrator.orchestrate_one(db, p)
    _expect(
        "orchestrate_one succeeded",
        lambda: res is not None and res["status"] == constants.STATUS_TRAINING)
    for jid in res["training_job_ids"]:
        _inserted_jobs.append(jid)
    jobs = list(db.jobs.find({"_id": {"$in": res["training_job_ids"]}}))
    exp1_jobs = [j for j in jobs if j["proposal_arm"] == "exp1"]
    base_jobs = [j for j in jobs if j["proposal_arm"] == "base"]
    # Every arm now gets its own derived design (so num_iterations
    # from the proposal can be stamped onto it; otherwise the
    # canonical base's num_iterations would silently override the
    # job's stamped value via apply_to_main_kwargs at trainer time).
    base_derived_id = base_jobs[0]["experiment_design_id"]
    derived_id = exp1_jobs[0]["experiment_design_id"]
    _expect(
        "base arm gets a derived ObjectId (NOT canonical string)",
        lambda: base_derived_id != orchestrator.CANONICAL_EXPERIMENT_DESIGN_ID
                and not isinstance(base_derived_id, str))
    _expect(
        "exp1 arm gets a derived ObjectId",
        lambda: derived_id != orchestrator.CANONICAL_EXPERIMENT_DESIGN_ID
                and not isinstance(derived_id, str))
    base_derived_doc = db.experiment_designs.find_one({"_id": base_derived_id})
    derived_doc = db.experiment_designs.find_one({"_id": derived_id})
    _inserted_designs.append(base_derived_id)
    _inserted_designs.append(derived_id)
    _expect(
        "exp1 derived design exists",
        lambda: derived_doc is not None)
    _expect(
        "exp1 derived design carries gamma + batch_size at top level",
        lambda: derived_doc.get("gamma") == 0.95 and derived_doc.get("batch_size") == 256)
    _expect(
        "exp1 derived design carries proposal_id provenance",
        lambda: derived_doc.get("proposal_id") == str(p["_id"]))
    _expect(
        "exp1 derived design carries base_design_id pointing at canonical",
        lambda: derived_doc.get("base_design_id") == orchestrator.CANONICAL_EXPERIMENT_DESIGN_ID)
    # num_iterations stamping (the fix for the dashboard-vs-job
    # divergence the operator hit).
    expected_iters = int(p.get("num_iterations_per_seed") or 5000)
    _expect(
        "base derived design has num_iterations stamped from proposal",
        lambda: base_derived_doc.get("num_iterations") == expected_iters,
        f"got {base_derived_doc.get('num_iterations')!r}, expected {expected_iters}")
    _expect(
        "exp1 derived design has num_iterations stamped from proposal",
        lambda: derived_doc.get("num_iterations") == expected_iters,
        f"got {derived_doc.get('num_iterations')!r}, expected {expected_iters}")


def test_arm_with_design_reference_only():
    print("\nGroup 3: arm with experiment_design_id (no fields) uses it directly",
          flush=True)
    db = _db()
    # Pre-create a reference design.
    ref_doc = {
        "name": "[orchestrator-test] ref design",
        "version": 1,
        "archived": False,
        "create_date": _now(),
        "gamma": 0.97,
    }
    ref_id = db.experiment_designs.insert_one(ref_doc).inserted_id
    _inserted_designs.append(ref_id)

    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1", "experiment_design_id": str(ref_id)},
    ], n_seeds_per_arm=1, title_suffix="ref-only")
    res = orchestrator.orchestrate_one(db, p)
    _expect(
        "orchestrate_one succeeded",
        lambda: res["status"] == constants.STATUS_TRAINING)
    for jid in res["training_job_ids"]:
        _inserted_jobs.append(jid)
    exp1_job = db.jobs.find_one({"_id": res["training_job_ids"][1]})
    # Even when arm references an existing design without overlay
    # fields, the orchestrator now creates a derived design that
    # inherits from ref_id and stamps num_iterations. The derived
    # design carries base_design_id=ref_id for provenance.
    exp1_design_id = exp1_job["experiment_design_id"]
    _expect(
        "exp1 design id is a derived ObjectId (not the bare ref_id)",
        lambda: exp1_design_id != ref_id and not isinstance(exp1_design_id, str),
        f"got {exp1_design_id!r}")
    exp1_derived = db.experiment_designs.find_one({"_id": exp1_design_id})
    _inserted_designs.append(exp1_design_id)
    _expect(
        "exp1 derived design has base_design_id pointing at the referenced design",
        lambda: exp1_derived.get("base_design_id") == str(ref_id),
        f"got base_design_id={exp1_derived.get('base_design_id')!r}, ref_id={ref_id}")
    _expect(
        "exp1 derived design has num_iterations stamped from proposal",
        lambda: exp1_derived.get("num_iterations") == int(
            p.get("num_iterations_per_seed") or 5000))

    # ref-only arms still produce a derived design per arm (2 arms
    # here -> 2 derived). The ref_id itself stays around as the
    # base; we count derived designs by proposal_id provenance.
    derived_count = db.experiment_designs.count_documents(
        {"proposal_id": str(p["_id"])})
    _expect(
        "2 derived designs total (one per arm) for ref-only proposal",
        lambda: derived_count == 2,
        f"found {derived_count} derived design(s)")


def test_inline_reward_rejected():
    print("\nGroup 4: arm with inline reward_design_fields is rejected (Phase 1C-MVP scope)",
          flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1", "reward_design_fields": {"reward_standard": "def reward_standard(...): return 0"}},
    ], n_seeds_per_arm=1, title_suffix="reward-inline")
    res = orchestrator.orchestrate_one(db, p)
    _expect(
        "status=failed",
        lambda: res["status"] == constants.STATUS_FAILED)
    _expect(
        "implementation_failure_reason mentions reward",
        lambda: "reward_design_fields" in (res.get("implementation_failure_reason") or "")
                or "reward" in (res.get("implementation_failure_reason") or "").lower())
    # And no jobs should have been queued.
    leaked = db.jobs.find_one({"proposal_id": str(p["_id"])})
    _expect(
        "no TRAIN jobs leaked",
        lambda: leaked is None)


def test_job_cap_enforced():
    print("\nGroup 5: max_jobs cap is enforced", flush=True)
    db = _db()
    # 5 arms x 4 seeds = 20 jobs; cap=10 -> rejection.
    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1"}, {"name": "exp2"}, {"name": "exp3"}, {"name": "exp4"},
    ], n_seeds_per_arm=4, title_suffix="too-many-jobs")
    res = orchestrator.orchestrate_one(db, p, max_jobs=10)
    _expect(
        "status=failed (job cap)",
        lambda: res["status"] == constants.STATUS_FAILED)
    _expect(
        "failure reason mentions MAX_JOBS_PER_PROPOSAL",
        lambda: "MAX_JOBS_PER_PROPOSAL" in (res.get("implementation_failure_reason") or "")
                or "20" in (res.get("implementation_failure_reason") or ""))


def test_unknown_schema_key_rejected():
    print("\nGroup 6: experiment_design_fields with unknown schema key rejected",
          flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[
        {"name": "base"},
        {"name": "exp1", "experiment_design_fields": {"nonexistent_knob": 42}},
    ], n_seeds_per_arm=1, title_suffix="unknown-key")
    res = orchestrator.orchestrate_one(db, p)
    # If SCHEMA was importable (default in container), we expect rejection.
    # If SCHEMA was empty (host without bind mount), we'd skip - but
    # we're inside the container so this should fail.
    _expect(
        "status=failed (unknown schema key)",
        lambda: res["status"] == constants.STATUS_FAILED)
    _expect(
        "failure reason mentions unknown key",
        lambda: "nonexistent_knob" in (res.get("implementation_failure_reason") or ""))


def test_non_eligible_proposals():
    print("\nGroup 7: non-eligible proposals are silently skipped", flush=True)
    db = _db()

    # 7a: status=pending_user
    p1 = _insert_proposal(db, arms=[
        {"name": "base"}, {"name": "exp1"}
    ], status=constants.STATUS_PENDING_USER, title_suffix="pending")
    res = orchestrator.orchestrate_one(db, p1)
    _expect("status=pending_user returns None", lambda: res is None)
    # And status should be unchanged.
    after = db.proposals.find_one({"_id": p1["_id"]})
    _expect(
        "status stays pending_user",
        lambda: after["status"] == constants.STATUS_PENDING_USER)

    # 7b: empty experiment_arms
    p2 = _insert_proposal(db, arms=[], status=constants.STATUS_APPROVED,
                          title_suffix="empty-arms")
    res = orchestrator.orchestrate_one(db, p2)
    _expect("empty arms returns None", lambda: res is None)
    after = db.proposals.find_one({"_id": p2["_id"]})
    _expect(
        "status stays approved (eligibility miss is non-mutating)",
        lambda: after["status"] == constants.STATUS_APPROVED)

    # 7c: n_seeds_per_arm = 0
    p3 = _insert_proposal(db, arms=[{"name": "base"}, {"name": "exp1"}],
                          n_seeds_per_arm=0, title_suffix="zero-seeds")
    res = orchestrator.orchestrate_one(db, p3)
    _expect("n_seeds_per_arm=0 returns None", lambda: res is None)


def test_idempotent():
    print("\nGroup 8: re-running orchestrate_one on a transitioned proposal "
          "returns None (no double-queue)", flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[{"name": "base"}, {"name": "exp1"}],
                        n_seeds_per_arm=1, title_suffix="idempotent")
    first = orchestrator.orchestrate_one(db, p)
    _expect(
        "first orchestrate: status=training",
        lambda: first["status"] == constants.STATUS_TRAINING)
    for jid in first["training_job_ids"]:
        _inserted_jobs.append(jid)
    n_jobs_after_first = db.jobs.count_documents({"proposal_id": str(p["_id"])})

    second = orchestrator.orchestrate_one(db, first)
    _expect(
        "second orchestrate: returns None (no eligibility)",
        lambda: second is None)
    n_jobs_after_second = db.jobs.count_documents({"proposal_id": str(p["_id"])})
    _expect(
        "no extra jobs were queued",
        lambda: n_jobs_after_first == n_jobs_after_second,
        f"before={n_jobs_after_first}, after={n_jobs_after_second}")


def test_1C_to_1E_handoff():
    print("\nGroup 9: end-to-end 1C->1E handoff (orchestrator queues jobs, "
          "then synthetic DONE + model -> outcome ingester picks up)",
          flush=True)
    db = _db()
    p = _insert_proposal(db, arms=[{"name": "base"}, {"name": "exp1"}],
                        n_seeds_per_arm=2, title_suffix="1C-to-1E handoff")
    # Add a structured criterion so the ingester emits a non-inconclusive
    # verdict.
    db.proposals.update_one(
        {"_id": p["_id"]},
        {"$set": {"success_criteria.primary_parsed": {
            "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
            "comparator": ">=", "threshold": 0.10,
            "threshold_kind": "relative",
        }}})
    p = db.proposals.find_one({"_id": p["_id"]})

    # 1C: orchestrate -> queues 4 jobs.
    res = orchestrator.orchestrate_one(db, p)
    _expect(
        "1C: orchestrate succeeded, status=training",
        lambda: res["status"] == constants.STATUS_TRAINING)
    for jid in res["training_job_ids"]:
        _inserted_jobs.append(jid)

    # Simulate the trainer running all jobs to DONE with synthetic
    # is_global_best models. arm_b=base gets avg_return=50; arm_a=exp1
    # gets avg_return=60 (20% improvement, exceeds 10% threshold).
    for jid in res["training_job_ids"]:
        job = db.jobs.find_one({"_id": jid})
        db.jobs.update_one({"_id": jid}, {"$set": {"status": "DONE"}})
        avg = 60.0 if job["proposal_arm"] == "exp1" else 50.0
        model_doc = {
            "job_id": str(jid),
            "create_date": _now(),
            "avg_return": avg,
            "is_global_best": True,
            "robot_type": "robotaxi",
            "model_type": "SacAgent",
            "training_iterations": 5000,
            "title": "[orchestrator-test-model]",
        }
        mid = db.models.insert_one(model_doc).inserted_id
        _inserted_models.append(mid)

    # 1E: outcome ingester picks it up.
    p2 = db.proposals.find_one({"_id": p["_id"]})
    out_res = outcome_ingester.ingest_one(db, p2)
    _expect(
        "1E: ingest_one returns updated doc",
        lambda: out_res is not None)
    _expect(
        "1E: status=done",
        lambda: out_res["status"] == constants.STATUS_DONE)
    _expect(
        "1E: verdict=supported (20% delta >= 10% threshold)",
        lambda: out_res["results"]["verdict"] == constants.OUTCOME_SUPPORTED,
        f"verdict={out_res['results']['verdict']}")
    _expect(
        "1E: per_arm has 2 entries with n_trials=2 each",
        lambda: len(out_res["results"]["per_arm"]) == 2
                and all(a["n_trials"] == 2 for a in out_res["results"]["per_arm"]))


# ---- Cleanup + main -----------------------------------------------------


def _cleanup(db):
    if _inserted_proposals:
        db.proposals.delete_many({"_id": {"$in": _inserted_proposals}})
    if _inserted_jobs:
        db.jobs.delete_many({"_id": {"$in": _inserted_jobs}})
    if _inserted_models:
        db.models.delete_many({"_id": {"$in": _inserted_models}})
    if _inserted_designs:
        db.experiment_designs.delete_many({"_id": {"$in": _inserted_designs}})
    # Also clean any derived designs we created during the orchestration
    # that we don't have explicit ids for (those have a proposal_id field).
    if _inserted_proposals:
        derived = db.experiment_designs.delete_many(
            {"proposal_id": {"$in": [str(p) for p in _inserted_proposals]}})
        if derived.deleted_count:
            print(
                f"  cleanup: also removed {derived.deleted_count} derived "
                f"designs created during orchestration",
                flush=True)
    print(
        f"  cleanup: removed {len(_inserted_proposals)} proposals, "
        f"{len(_inserted_jobs)} jobs, {len(_inserted_models)} models, "
        f"{len(_inserted_designs)} designs",
        flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/orchestrator.py "
          "(Phase 1C-MVP)", flush=True)
    print("=" * 64, flush=True)

    db = _db()
    try:
        test_plain_approved_proposal()
        test_arm_with_overlay()
        test_arm_with_design_reference_only()
        test_inline_reward_rejected()
        test_job_cap_enforced()
        test_unknown_schema_key_rejected()
        test_non_eligible_proposals()
        test_idempotent()
        test_1C_to_1E_handoff()
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
