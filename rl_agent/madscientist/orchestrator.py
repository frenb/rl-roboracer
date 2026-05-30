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
from . import cursor_orchestrator


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

    # Phase 1C-Full: keys declared in proposed_schema_extensions are
    # also accepted - the Cursor SDK path will add them to SCHEMA
    # before training jobs run. Mirrors the relaxed pre_rubric_check_e.
    extensions = proposal_doc.get("proposed_schema_extensions") or []
    proposed_keys = set()
    for ext in extensions:
        name = ext.get("name") if isinstance(ext, dict) else getattr(ext, "name", None)
        if isinstance(name, str) and name:
            proposed_keys.add(name)

    for arm in arms:
        arm_name = arm.get("name") or "?"
        # Reject inline reward_design_fields - Phase 1C-MVP only supports
        # reference-by-id for rewards.
        if arm.get("reward_design_fields"):
            return (
                f"arm '{arm_name}' has inline reward_design_fields; "
                f"Phase 1C-MVP only supports reward_design_id references. "
                f"Author the reward design on the Reward Design tab "
                f"and reference it by id, or wait for a future "
                f"reward-authoring orchestrator extension.")

        # Validate experiment_design_fields keys (defense in depth).
        # Only checked when SCHEMA was importable (otherwise the set is
        # empty and we'd reject everything). Keys declared in
        # proposed_schema_extensions are accepted.
        if valid_design_keys:
            for k in (arm.get("experiment_design_fields") or {}).keys():
                if k not in valid_design_keys and k not in proposed_keys:
                    return (
                        f"arm '{arm_name}': experiment_design_fields key "
                        f"{k!r} is not in experiment_designs.SCHEMA and "
                        f"not declared in proposed_schema_extensions. "
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
    num_iterations: int,
    now: datetime.datetime,
) -> Any:
    """Return an experiment_design_id appropriate for the arm.

    Always creates a NEW derived experiment_designs document, even when
    the arm has no experiment_design_fields overrides. This guarantees
    every orchestrator-queued job has a derived design that:
      * stamps num_iterations from proposal.num_iterations_per_seed
        (so the dashboard's design view matches what the job actually
        runs; otherwise the base's default num_iterations - e.g. 50000
        on `experiment-default` - would silently override the job's
        stamped value via apply_to_main_kwargs)
      * carries proposal_id provenance
      * overlays any arm.experiment_design_fields on top of the base

    Arms with no overlay still get a derived design - same fields as
    the base but with num_iterations stamped and provenance set. The
    storage overhead is tiny (one small doc per arm per proposal).

    The base is `arm.experiment_design_id` if set, else canonical.
    """
    overlay = dict(arm.get("experiment_design_fields") or {})
    base_id = arm.get("experiment_design_id") or CANONICAL_EXPERIMENT_DESIGN_ID

    # Resolve base for metadata (name, description). Missing-base is
    # non-fatal: the derived design still gets created with the
    # overlay fields the proposal explicitly requested, and the
    # trainer's apply_to_main_kwargs falls back to main()'s defaults
    # for any field neither in the derived doc nor on the job.
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
    overlay_keys = sorted(overlay.keys())
    derived_description = (
        f"Auto-generated by orchestrator from proposal {proposal_id_str} "
        f"arm '{arm_name}'. Base: {base_name}. "
        f"num_iterations={num_iterations}"
        + (f". Overlay: {', '.join(overlay_keys)}." if overlay_keys
           else ". No field overrides."))

    # Force the proposal's num_iterations onto the derived design.
    # If the arm explicitly tried to override num_iterations via
    # experiment_design_fields, the arm's value wins (researcher's
    # explicit per-arm intent beats the proposal-level default).
    if "num_iterations" not in overlay:
        overlay["num_iterations"] = num_iterations

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
        # Overlay fields stored at the top level so the trainer's
        # apply_to_main_kwargs() (which iterates over SCHEMA keys
        # looking at doc[field_name]) reads them automatically.
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
    experiment_design_name: Optional[str],
    reward_design_id: Optional[str],
    reward_design_name: Optional[str],
    seed: int,
    num_iterations: int,
    now: datetime.datetime,
) -> Dict[str, Any]:
    """One TRAIN job document, shaped to match what the existing
    dashboard's New-job form produces.

    Keeps all the legacy-but-required fields (pass_through_actions,
    nn_size_x/y, demo_job_id) populated with their defaults so the
    trainer's job-document accessors don't KeyError on them. Stamps
    the {experiment,reward}_design_name fields so the Jobs tab's
    badge columns render proper labels (matching what
    dashboard-submitted jobs carry).
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
        "experiment_design_name": experiment_design_name,
        "reward_design_id": reward_design_id,
        "reward_design_name": reward_design_name,
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


def _lookup_design_name(db, design_id) -> Optional[str]:
    """Return the `name` field of a design document by its id, or None
    if the lookup fails (missing doc, bad id type, etc.).

    Tolerates id stored as ObjectId, str(ObjectId), or the canonical
    string '_id' like 'experiment-default'.
    """
    if not design_id:
        return None
    doc = None
    try:
        from bson import ObjectId
        try:
            doc = db.experiment_designs.find_one({"_id": ObjectId(str(design_id))})
        except Exception:  # noqa: BLE001
            doc = None
    except ImportError:
        doc = None
    if doc is None:
        doc = db.experiment_designs.find_one({"_id": str(design_id)})
    if doc is None:
        return None
    return doc.get("name")


def _lookup_reward_design_name(db, reward_id) -> Optional[str]:
    """Same as _lookup_design_name but for the reward_designs
    collection."""
    if not reward_id:
        return None
    doc = None
    try:
        from bson import ObjectId
        try:
            doc = db.reward_designs.find_one({"_id": ObjectId(str(reward_id))})
        except Exception:  # noqa: BLE001
            doc = None
    except ImportError:
        doc = None
    if doc is None:
        doc = db.reward_designs.find_one({"_id": str(reward_id)})
    if doc is None:
        return None
    return doc.get("name")


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
            db, arm, proposal_id_str, arm_name,
            num_iterations=num_iterations, now=now)
        reward_id = _resolve_reward_design_id(arm)
        # Resolve human-readable names so the Jobs tab's badge
        # columns render proper labels (matching dashboard-submitted
        # jobs which stamp these at submit time).
        design_name = _lookup_design_name(db, design_id)
        reward_name = _lookup_reward_design_name(db, reward_id)
        for seed in range(n_seeds):
            job = _build_job_doc(
                proposal_id_str=proposal_id_str,
                arm_name=arm_name,
                experiment_design_id=design_id,
                experiment_design_name=design_name,
                reward_design_id=reward_id,
                reward_design_name=reward_name,
                seed=seed,
                num_iterations=num_iterations,
                now=now)
            jid = db.jobs.insert_one(job).inserted_id
            inserted.append(jid)
    return inserted


# ---- Single-proposal orchestrator ----------------------------------------


def _needs_cursor_path(proposal_doc: Dict[str, Any]) -> bool:
    """True iff the proposal declares non-empty proposed_schema_extensions.

    Phase 1C-Full: such proposals can't be implemented via auto-queue
    because they reference SCHEMA fields that don't exist yet. We
    spawn a Cursor SDK code-writing agent to add the fields + open
    a PR. After human PR review + merge, a follow-up command (Phase
    1C-Full v2) queues the actual training jobs.
    """
    exts = proposal_doc.get("proposed_schema_extensions") or []
    return bool(exts)


def orchestrate_one(
    db,
    proposal_doc: Dict[str, Any],
    *,
    max_jobs: int = DEFAULT_MAX_JOBS_PER_PROPOSAL,
    cursor_spawn_fn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single proposal end-to-end. Returns the updated doc
    or None if the proposal wasn't eligible (status != approved /
    bad shape / etc.).

    Two paths:

      * AUTO-QUEUE (Phase 1C-MVP): proposal references only existing
        SCHEMA fields. Orchestrator seeds derived experiment_designs
        + queues TRAIN jobs immediately. status: approved -> training.

      * CURSOR (Phase 1C-Full): proposal carries
        proposed_schema_extensions. Orchestrator spawns a Cursor cloud
        agent that adds the SCHEMA fields + opens a PR. After PR
        opens: status -> pr_open. Training jobs are NOT queued until
        a follow-up operation (Phase 1C-Full v2) confirms the PR has
        been merged - we don't want to run training against
        unmerged code.

    cursor_spawn_fn: test hook overriding the real
        cursor_orchestrator.spawn_cursor_agent_for_proposal.

    On validation failure: status -> failed with implementation_failure_reason.
    On exception during the chosen path: status -> failed with details.
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

    # ---- Step 3: pick the right path ---------------------------------
    if _needs_cursor_path(proposal_doc):
        return _run_cursor_path(
            db, proposal_doc, now,
            cursor_spawn_fn=cursor_spawn_fn)
    return _run_auto_queue_path(db, proposal_doc, now)


def _run_auto_queue_path(
    db, proposal_doc: Dict[str, Any], now: datetime.datetime,
) -> Optional[Dict[str, Any]]:
    """Phase 1C-MVP: seed derived designs + queue TRAIN jobs."""
    proposal_id = proposal_doc["_id"]
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


def _run_cursor_path(
    db,
    proposal_doc: Dict[str, Any],
    now: datetime.datetime,
    *,
    cursor_spawn_fn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Phase 1C-Full: spawn Cursor SDK code-writing agent.

    Status flow:
      approved -> implementing (already set by orchestrate_one)
                -> pr_open  (if agent succeeded + PR URL extracted)
                -> failed   (if agent didn't start / errored / no PR URL)
    """
    proposal_id = proposal_doc["_id"]
    spawn = cursor_spawn_fn or cursor_orchestrator.spawn_cursor_agent_for_proposal

    try:
        result = spawn(db, proposal_doc)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(
            f"orchestrator: Cursor spawn raised for {proposal_id}: "
            f"{type(e).__name__}: {e}\n{tb}", flush=True)
        return _persist_failure(
            db, proposal_id,
            f"Cursor spawn raised: {type(e).__name__}: {e}",
            now, audit_detail={
                "phase": "cursor_spawn",
                "traceback_tail": tb[-500:]})

    status = result.get("status")
    pr_url = result.get("pr_url")
    branch_name = result.get("branch_name")
    agent_id = result.get("agent_id")
    run_id = result.get("run_id")
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    if status == "finished" and pr_url:
        update = {
            "$set": {
                "status": constants.STATUS_PR_OPEN,
                "implementation_pr_url": pr_url,
                "implementation_branch": branch_name,
                "implementation_finished_at": finished_at,
                "updated_at": finished_at,
            },
            "$push": {"audit_events": {
                "at": finished_at,
                "by_agent": constants.AGENT_ORCHESTRATOR,
                "event": "cursor_pr_opened",
                "detail": {
                    "pr_url": pr_url,
                    "branch_name": branch_name,
                    "agent_id": agent_id,
                    "run_id": run_id,
                },
            }},
        }
        db[constants.COLL_PROPOSALS].update_one(
            {"_id": proposal_id}, update)
        print(
            f"orchestrator (Cursor): proposal {proposal_id} -> pr_open, "
            f"PR={pr_url}",
            flush=True)
        return db[constants.COLL_PROPOSALS].find_one({"_id": proposal_id})

    # Cursor finished but no PR URL OR errored OR didn't start ->
    # treat as failure. Branch name + agent metadata still stamped
    # for forensics.
    reason = result.get("error") or f"Cursor agent ended with status={status!r}"
    db[constants.COLL_PROPOSALS].update_one(
        {"_id": proposal_id},
        {
            "$set": {
                "status": constants.STATUS_FAILED,
                "implementation_failure_reason": reason,
                "implementation_branch": branch_name,
                "implementation_finished_at": finished_at,
                "updated_at": finished_at,
            },
            "$push": {"audit_events": {
                "at": finished_at,
                "by_agent": constants.AGENT_ORCHESTRATOR,
                "event": "cursor_implementation_failed",
                "detail": {
                    "cursor_status": status,
                    "reason": reason,
                    "branch_name": branch_name,
                    "agent_id": agent_id,
                    "run_id": run_id,
                },
            }},
        })
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
