"""Orchestrator worker - implements approved proposals (Phase 1C-MVP).

This is the MVP variant that does NOT invoke a Cursor SDK code-writing
agent. Instead, when a proposal is approved, the worker:

  1. Validates the proposal (experiment_arms shape, max-jobs cap).
  2. Seeds derived experiment_designs in Mongo, one per arm that has
     experiment_design_fields overrides (cloning the canonical Default
     + applying the overlay as top-level fields the trainer's
     apply_to_main_kwargs already knows how to read).
  3. Queues TRAIN jobs in Mongo, one per (arm x seed), with
     proposal_id + proposal_arm fields stamped. The existing trainer
     poll loop picks these up automatically; no trainer changes
     required.
  4. Advances proposal.status approved -> implementing -> training,
     stores training_job_ids, stamps an audit_event.

What this MVP does NOT do (deferred to Phase 1C-Full):
  * Spawn a Cursor cloud agent to edit experiment_designs.py SCHEMA
    or reward_designs.py source. So proposals that require NEW schema
    fields can't ship through this MVP - they'd need a human to
    implement the code first, then the operator can re-approve.
  * Open a GitHub PR.
  * Stream live activity into proposal.implementation_log (for now
    the implementation is instantaneous - status flips happen in one
    transaction, no progress streaming needed).

Pre-rubric check E already enforces that proposals only reference
KNOWN schema keys; combined with the manual approval gate, this MVP
is safe for typical "tune existing hyperparameters" proposals
authored by the Researcher (Phase 1B).

Note on reward designs: arms that REFERENCE an existing reward_design
via reward_design_id are supported. Arms that try to seed a NEW
reward design via inline reward_design_fields are rejected with a
clear error - reward authoring through the Researcher will land
together with Phase 1C-Full's Cursor SDK integration.

Worker shape mirrors judge.py / outcome_ingester.py: orchestrate_one
is the unit-tested entry point; orchestrate_loop polls.
"""
from __future__ import annotations

import datetime
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    from rl_agent.experiment_designs import SCHEMA as _EXPERIMENT_DESIGNS_SCHEMA
except ImportError:
    _EXPERIMENT_DESIGNS_SCHEMA = {}

from . import constants


# ---- Configuration -------------------------------------------------------

# Cap the total jobs a single proposal can queue. A buggy / malicious
# proposal with 100 arms x 100 seeds would otherwise spam the trainer
# queue; this halts well-formed proposals only if they're genuinely
# huge, in which case the operator can raise the cap or re-architect
# the experiment.
DEFAULT_MAX_JOBS_PER_PROPOSAL = 50

# Reference to the canonical "Default" experiment design. Phase 0's
# seed_canonical_experiment_design() upserts a doc with this string
# _id; the trainer's resolution code in robotaxi.py handles both
# string and ObjectId lookups.
CANONICAL_EXPERIMENT_DESIGN_ID = "experiment-default"


# ---- Eligibility ---------------------------------------------------------


def _is_eligible(proposal_doc: Dict[str, Any]) -> Tuple[bool, str]:
    """Decide whether this proposal can be orchestrated right now.

    Returns (eligible, reason). Eligible only when status=approved.
    Other statuses are silently no-op (the loop just skips them).
    """
    status = proposal_doc.get("status")
    if status != constants.STATUS_APPROVED:
        return (False, f"status={status!r} not approved; skipping")
    arms = proposal_doc.get("experiment_arms") or []
    if not arms:
        return (False, "no experiment_arms; cannot queue any jobs")
    n_seeds = int(proposal_doc.get("n_seeds_per_arm") or 0)
    if n_seeds < 1:
        return (False, f"n_seeds_per_arm={n_seeds} < 1")
    return (True, "ok")


def _validate_or_fail(proposal_doc: Dict[str, Any], *, max_jobs: int) -> Optional[str]:
    """Run pre-queue validations on the proposal. Returns None on success
    or a reason string on failure.

    Catches three classes of problem the Judge's pre-rubric checks may
    not have caught (because they're orchestrator-specific, not
    rubric-axis-relevant):

      * Proposals that would exceed max_jobs.
      * Arms with inline reward_design_fields (deferred to Phase 1C-Full).
      * experiment_design_fields keys that aren't in the schema (defense
        in depth against an operator who edited the proposal between
        the Judge's approval and the orchestrator's pickup).
    """
    arms = proposal_doc.get("experiment_arms") or []
    n_seeds = int(proposal_doc.get("n_seeds_per_arm") or 1)
    total_jobs = len(arms) * n_seeds
    if total_jobs > max_jobs:
        return (
            f"would queue {total_jobs} jobs ({len(arms)} arms x {n_seeds} "
            f"seeds), exceeds MAX_JOBS_PER_PROPOSAL={max_jobs}. Either "
            f"reduce n_seeds_per_arm or raise the cap.")

    valid_design_keys = {
        k for k in _EXPERIMENT_DESIGNS_SCHEMA.keys()
        if not k.startswith("_section_")}

    for arm in arms:
        arm_name = arm.get("name") or "?"
        # Reject inline reward_design_fields - Phase 1C-MVP only supports
        # reference-by-id for rewards.
        if arm.get("reward_design_fields"):
            return (
                f"arm '{arm_name}' has inline reward_design_fields; "
                f"Phase 1C-MVP only supports reward_design_id references. "
                f"Author the reward design on the Reward Design tab "
                f"and reference it by id, or wait for Phase 1C-Full.")

        # Validate experiment_design_fields keys (defense in depth).
        # Only checked when SCHEMA was importable (otherwise the set is
        # empty and we'd reject everything).
        if valid_design_keys:
            for k in (arm.get("experiment_design_fields") or {}).keys():
                if k not in valid_design_keys:
                    return (
                        f"arm '{arm_name}': experiment_design_fields key "
                        f"{k!r} is not in experiment_designs.SCHEMA. "
                        f"Pre-rubric check E should have caught this; "
                        f"orchestrator refuses to queue jobs against an "
                        f"unknown schema key.")

    return None


# ---- Design seeding ------------------------------------------------------


def _ensure_experiment_design(
    db,
    arm: Dict[str, Any],
    proposal_id_str: str,
    arm_name: str,
    now: datetime.datetime,
) -> Any:
    """Return an experiment_design_id appropriate for the arm.

    Three cases:

      (a) arm has experiment_design_id + no experiment_design_fields:
          plain reference. Return the existing id as-is.

      (b) arm has only experiment_design_id (no fields) OR no design
          reference at all: return the canonical default id. The
          trainer's lookup resolves "experiment-default" via string
          _id match.

      (c) arm has experiment_design_fields (with or without
          experiment_design_id base): create a NEW experiment_designs
          document that clones the base and overlays the fields as
          top-level keys. The new doc's _id is an ObjectId; we stamp
          proposal_id on it for downstream provenance.
    """
    overlay = arm.get("experiment_design_fields") or {}
    base_id = arm.get("experiment_design_id") or CANONICAL_EXPERIMENT_DESIGN_ID

    if not overlay:
        # (a) or (b) - plain reference, no overlay needed.
        return base_id

    # (c) - need to seed a derived design.
    # Try to pull the base for any extra metadata (name, description);
    # missing-base is non-fatal because the trainer can resolve
    # individual fields without needing a base doc.
    base_doc = None
    try:
        from bson import ObjectId
        try:
            base_doc = db.experiment_designs.find_one({"_id": ObjectId(str(base_id))})
        except Exception:
            base_doc = None
    except ImportError:
        base_doc = None
    if base_doc is None:
        base_doc = db.experiment_designs.find_one({"_id": str(base_id)})

    base_name = (base_doc or {}).get("name", "default")
    derived_name = f"auto:{proposal_id_str[-8:]}:{arm_name}"
    derived_description = (
        f"Auto-generated by orchestrator from proposal {proposal_id_str} "
        f"arm '{arm_name}'. Base: {base_name}. Overlay: "
        f"{', '.join(overlay.keys())}.")

    derived = {
        "name": derived_name,
        "description": derived_description,
        "version": 1,
        "archived": False,
        "create_date": now,
        # Provenance fields - let the Models / Jobs / Outcomes tabs trace
        # back to the originating proposal.
        "proposal_id": proposal_id_str,
        "base_design_id": str(base_id),
        # Overlay fields are stored at the top level so the trainer's
        # apply_to_main_kwargs() (which iterates over SCHEMA keys looking
        # at doc[field_name]) reads them automatically without a new
        # code path.
        **overlay,
    }
    res = db.experiment_designs.insert_one(derived)
    return res.inserted_id  # ObjectId; trainer resolves via ObjectId match


def _resolve_reward_design_id(arm: Dict[str, Any]) -> Optional[str]:
    """Return the reward_design_id for this arm, or None.

    Phase 1C-MVP doesn't seed new reward designs (rejected upstream
    by _validate_or_fail). We just pass through the arm's id.
    None means "use the course's default reward formulas".
    """
    rid = arm.get("reward_design_id")
    if not rid:
        return None
    return str(rid)


# ---- Job queueing --------------------------------------------------------


def _build_job_doc(
    *,
    proposal_id_str: str,
    arm_name: str,
    experiment_design_id: Any,
    reward_design_id: Optional[str],
    seed: int,
    num_iterations: int,
    now: datetime.datetime,
) -> Dict[str, Any]:
    """One TRAIN job document, shaped to match what the existing
    dashboard's New-job form produces.

    Keeps all the legacy-but-required fields (pass_through_actions,
    nn_size_x/y, demo_job_id) populated with their defaults so the
    trainer's job-document accessors don't KeyError on them.
    """
    return {
        # ---- Core job fields the trainer reads -------------------------
        "status": "NOT_STARTED",
        "job_type": "TRAIN",
        "model_type": "SacAgent",
        "robot_type": "robotaxi",
        "num_iterations": num_iterations,
        "training_steps": 0,
        "demo_job_id": "",
        "pass_through_actions": False,
        # Network architecture: empty string lets the trainer fall back
        # to its built-in 512x512 defaults (see robotaxi.py's
        # actor_fc_layer_params_* resolution).
        "nn_size_x": "",
        "nn_size_y": "",
        # ---- Design references -----------------------------------------
        "experiment_design_id": experiment_design_id,
        "reward_design_id": reward_design_id,
        "seed": seed,
        # ---- Proposal linkage (new in Phase 1C) ------------------------
        # outcome_ingester.py groups results per arm using
        # job.proposal_arm; this is the field that closes the loop.
        "proposal_id": proposal_id_str,
        "proposal_arm": arm_name,
        # ---- Bookkeeping -----------------------------------------------
        "percent_complete": 0,
        "create_date": now,
    }


def _queue_jobs(
    db,
    proposal_doc: Dict[str, Any],
    now: datetime.datetime,
) -> List[Any]:
    """Insert one TRAIN job per (arm x seed). Returns the list of
    inserted job _ids in insertion order."""
    arms = proposal_doc.get("experiment_arms") or []
    n_seeds = max(1, int(proposal_doc.get("n_seeds_per_arm") or 1))
    num_iterations = int(proposal_doc.get("num_iterations_per_seed") or 5000)
    proposal_id_str = str(proposal_doc["_id"])

    inserted: List[Any] = []
    for arm in arms:
        arm_name = arm.get("name") or "?"
        design_id = _ensure_experiment_design(
            db, arm, proposal_id_str, arm_name, now)
        reward_id = _resolve_reward_design_id(arm)
        for seed in range(n_seeds):
            job = _build_job_doc(
                proposal_id_str=proposal_id_str,
                arm_name=arm_name,
                experiment_design_id=design_id,
                reward_design_id=reward_id,
                seed=seed,
                num_iterations=num_iterations,
                now=now)
            jid = db.jobs.insert_one(job).inserted_id
            inserted.append(jid)
    return inserted


# ---- Single-proposal orchestrator ----------------------------------------


def orchestrate_one(
    db,
    proposal_doc: Dict[str, Any],
    *,
    max_jobs: int = DEFAULT_MAX_JOBS_PER_PROPOSAL,
) -> Optional[Dict[str, Any]]:
    """Process a single proposal end-to-end. Returns the updated doc
    or None if the proposal wasn't eligible (status != approved /
    bad shape / etc.).

    On success: status approved -> implementing -> training, training
    job records inserted into db.jobs, audit event stamped.

    On validation failure: status -> failed with implementation_failure_reason.

    On exception during job insertion: status -> failed with the
    exception message in implementation_failure_reason. The orchestrator
    does NOT try to roll back partially-inserted jobs because those
    are still valid TRAIN jobs - the trainer just won't pick them up
    against the failed proposal's stats since the proposal's
    training_job_ids list will only contain successfully-inserted ids.
    """
    eligible, reason = _is_eligible(proposal_doc)
    if not eligible:
        return None  # silent skip

    proposal_id = proposal_doc["_id"]
    now = datetime.datetime.now(datetime.timezone.utc)

    # ---- Step 1: validate before any state change -----------------------
    validation_error = _validate_or_fail(proposal_doc, max_jobs=max_jobs)
    if validation_error:
        return _persist_failure(
            db, proposal_id, validation_error, now,
            audit_detail={"phase": "validation"})

    # ---- Step 2: flip status to implementing (so concurrent
    # orchestrators don't double-pick the same proposal) ---------------
    db[constants.COLL_PROPOSALS].update_one(
        {"_id": proposal_id, "status": constants.STATUS_APPROVED},
        {"$set": {
            "status": constants.STATUS_IMPLEMENTING,
            "implementation_started_at": now,
            "updated_at": now,
        }})

    # ---- Step 3: seed designs + queue jobs ----------------------------
    try:
        training_job_ids = _queue_jobs(db, proposal_doc, now)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(
            f"orchestrator: job queue failed for {proposal_id}: "
            f"{type(e).__name__}: {e}\n{tb}", flush=True)
        return _persist_failure(
            db, proposal_id,
            f"{type(e).__name__}: {e}",
            now, audit_detail={"phase": "job_queue", "traceback_tail": tb[-500:]})

    # ---- Step 4: advance to training + record results ----------------
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    update = {
        "$set": {
            "status": constants.STATUS_TRAINING,
            "training_job_ids": training_job_ids,
            "implementation_finished_at": finished_at,
            "updated_at": finished_at,
        },
        "$push": {"audit_events": {
            "at": finished_at,
            "by_agent": constants.AGENT_ORCHESTRATOR,
            "event": "queued_training_jobs",
            "detail": {
                "n_jobs": len(training_job_ids),
                "n_arms": len(proposal_doc.get("experiment_arms") or []),
                "n_seeds_per_arm": int(proposal_doc.get("n_seeds_per_arm") or 1),
            },
        }},
    }
    db[constants.COLL_PROPOSALS].update_one({"_id": proposal_id}, update)
    return db[constants.COLL_PROPOSALS].find_one({"_id": proposal_id})


def _persist_failure(
    db,
    proposal_id,
    reason: str,
    now: datetime.datetime,
    *,
    audit_detail: Dict[str, Any],
) -> Dict[str, Any]:
    """Mark a proposal failed with an explanation."""
    db[constants.COLL_PROPOSALS].update_one(
        {"_id": proposal_id},
        {
            "$set": {
                "status": constants.STATUS_FAILED,
                "implementation_failure_reason": reason,
                "updated_at": now,
            },
            "$push": {"audit_events": {
                "at": now,
                "by_agent": constants.AGENT_ORCHESTRATOR,
                "event": "implementation_failed",
                "detail": {"reason": reason, **audit_detail},
            }},
        },
    )
    return db[constants.COLL_PROPOSALS].find_one({"_id": proposal_id})


# ---- Worker loop ---------------------------------------------------------


def orchestrate_loop(
    db,
    *,
    poll_interval_seconds: int = 30,
    should_stop_fn=None,
    max_jobs: int = DEFAULT_MAX_JOBS_PER_PROPOSAL,
):
    """Poll for approved proposals and process them.

    One per cycle - implementation is fast but we want each
    proposal's job queue + status transition to land atomically
    before the next cycle starts. With poll_interval=30s and
    typical proposals (~10 jobs), throughput is plenty.
    """
    print(
        f"orchestrator: starting loop, poll interval = "
        f"{poll_interval_seconds}s, max_jobs_per_proposal = {max_jobs}",
        flush=True)
    while True:
        if should_stop_fn is not None and should_stop_fn():
            print("orchestrator: stop signal received; exiting.", flush=True)
            return
        try:
            pending = db[constants.COLL_PROPOSALS].find_one(
                {"status": constants.STATUS_APPROVED},
                sort=[("decision.at", 1)])
        except Exception as e:  # noqa: BLE001
            print(f"orchestrator: Mongo lookup failed: {e}", flush=True)
            time.sleep(poll_interval_seconds)
            continue
        if pending is None:
            time.sleep(poll_interval_seconds)
            continue
        proposal_id = pending.get("_id")
        title = pending.get("title", "?")
        print(
            f"orchestrator: processing approved proposal {proposal_id} "
            f"({title!r})",
            flush=True)
        try:
            result = orchestrate_one(db, pending, max_jobs=max_jobs)
            if result is not None:
                new_status = result.get("status")
                n_jobs = len(result.get("training_job_ids") or [])
                print(
                    f"orchestrator: finished {proposal_id}: "
                    f"status={new_status}, queued {n_jobs} TRAIN job(s).",
                    flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(
                f"orchestrator: uncaught error on {proposal_id}: "
                f"{type(e).__name__}: {e}\n{tb}",
                flush=True)
            try:
                _persist_failure(
                    db, proposal_id, f"{type(e).__name__}: {e}",
                    datetime.datetime.now(datetime.timezone.utc),
                    audit_detail={"phase": "outer_loop"})
            except Exception:  # noqa: BLE001
                pass
        time.sleep(poll_interval_seconds)
