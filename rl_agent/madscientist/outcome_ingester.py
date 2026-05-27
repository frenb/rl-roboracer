"""Outcome ingester - computes proposal verdicts from completed TRAIN jobs.

When all TRAIN jobs linked to a proposal reach a terminal status
(DONE, FAILED, or CANCELLED), this worker:

  1. Gathers each arm's seed samples - the avg_return of the
     is_global_best model produced by each job, grouped by the job's
     `proposal_arm` field (set by the Phase 1C orchestrator when it
     queues jobs).

  2. Computes per-arm summary statistics: mean, sample stddev,
     median, and a bootstrap 95% confidence interval (1000 resamples).
     With typical 3-5 seeds per arm this is what the literature
     supports (Henderson 2018, Agarwal 2021); pure-stdlib so no
     scipy/numpy dependency.

  3. Evaluates the proposal's structured primary criterion (if
     present in success_criteria.primary_parsed) against the
     computed deltas. Produces a verdict of "supported", "rejected",
     or "inconclusive".

  4. Writes proposal.results, advances proposal.status to "done"
     (or "failed" if every linked job failed), stamps an audit_event.

Pure-Python: no scipy, no numpy, no LLM. The worker is cheap to
run, idempotent (re-running a finished proposal is a no-op), and
fully testable with synthetic Mongo data.

Worker shape mirrors judge.py: outcome_loop polls; the unit-tested
entry point is ingest_one(db, proposal_doc).
"""
from __future__ import annotations

import datetime
import math
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import constants


# ---- Statistics helpers --------------------------------------------------


_BOOTSTRAP_ITERATIONS = 1000  # plenty for 3-5 samples + sub-100ms
_BOOTSTRAP_SEED = 42          # deterministic results for the same inputs


def _bootstrap_ci_95(samples: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Percentile-bootstrap 95% CI on the mean.

    Returns (ci_low, ci_high) or (None, None) if too few samples.
    n=1 has no defined CI; n=2 is borderline but we still attempt
    (operator can decide what to do with a wide interval).
    """
    n = len(samples)
    if n < 2:
        return (None, None)
    rng = random.Random(_BOOTSTRAP_SEED)
    means: List[float] = []
    for _ in range(_BOOTSTRAP_ITERATIONS):
        # Resample with replacement.
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()
    lo_idx = int(0.025 * len(means))
    hi_idx = int(0.975 * len(means)) - 1
    hi_idx = max(hi_idx, lo_idx)
    return (means[lo_idx], means[hi_idx])


def _safe_stddev(samples: List[float]) -> Optional[float]:
    if len(samples) < 2:
        return None
    try:
        return statistics.stdev(samples)
    except statistics.StatisticsError:
        return None


def _safe_median(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    return statistics.median(samples)


def _safe_mean(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    return statistics.fmean(samples)


# ---- Data gathering ------------------------------------------------------


# Terminal statuses for a TRAIN job - any of these means the
# ingester can stop waiting and process the proposal.
_TERMINAL_JOB_STATUSES = ("DONE", "FAILED", "CANCELLED")

# Non-terminal job statuses. If a stale-failed proposal has ANY job
# back in one of these, the proposal is eligible for recovery.
_NON_TERMINAL_JOB_STATUSES = (
    "NOT_STARTED", "IN_PROGRESS", "PAUSED", "PAUSE_REQUESTED")


def _all_training_jobs_terminal(db, training_job_ids: List[Any]) -> Tuple[bool, int, int]:
    """Check whether every linked TRAIN job has reached a terminal
    status. Returns (all_terminal, n_succeeded, n_failed).

    n_succeeded counts jobs at status=DONE.
    n_failed counts jobs at status in {FAILED, CANCELLED}.
    Jobs in non-terminal states return all_terminal=False.
    """
    if not training_job_ids:
        return (False, 0, 0)
    # Query just the status field - cheap.
    cursor = db.jobs.find(
        {"_id": {"$in": list(training_job_ids)}},
        {"status": 1})
    succeeded = 0
    failed = 0
    found_ids = set()
    for j in cursor:
        found_ids.add(j["_id"])
        status = j.get("status")
        if status == "DONE":
            succeeded += 1
        elif status in ("FAILED", "CANCELLED"):
            failed += 1
        else:
            # IN_PROGRESS / NOT_STARTED / PAUSED / etc. - not terminal.
            return (False, succeeded, failed)
    # If some training_job_ids didn't resolve to a job document, treat
    # them as failures (job was deleted out from under the proposal).
    missing = len(training_job_ids) - len(found_ids)
    if missing:
        failed += missing
    return (True, succeeded, failed)


def _gather_arm_samples(db, training_job_ids: List[Any]) -> Dict[str, List[float]]:
    """For each TRAIN job's best model, group avg_return by arm name.

    Reads `job.proposal_arm` (set by the orchestrator at queue time)
    and `model.avg_return` (set by the trainer's add_model when a new
    is_global_best is saved).

    Returns dict[arm_name] -> list of avg_return floats. Jobs whose
    best model is missing avg_return or whose arm tag is missing are
    silently skipped (logged at the caller).
    """
    by_arm: Dict[str, List[float]] = {}
    if not training_job_ids:
        return by_arm
    jobs_cursor = db.jobs.find(
        {"_id": {"$in": list(training_job_ids)}},
        {"_id": 1, "status": 1, "proposal_arm": 1})
    for j in jobs_cursor:
        arm = j.get("proposal_arm")
        if not arm:
            continue
        job_id_str = str(j["_id"])
        # The is_global_best model per job is the canonical "this run's
        # best policy". With the resume-aware seeding fix landed in
        # 4b549fe, exactly one model per job carries is_global_best=True.
        best = db.models.find_one(
            {"job_id": job_id_str, "is_global_best": True},
            {"avg_return": 1})
        if not best:
            # Fall back to highest-avg_return model for this job if no
            # explicit is_global_best flag (covers proposals whose jobs
            # predate the flag).
            best = db.models.find_one(
                {"job_id": job_id_str},
                {"avg_return": 1},
                sort=[("avg_return", -1)])
        if not best:
            continue
        avg = best.get("avg_return")
        if avg is None:
            continue
        try:
            avg = float(avg)
        except (TypeError, ValueError):
            continue
        by_arm.setdefault(arm, []).append(avg)
    return by_arm


# ---- Criterion evaluation ------------------------------------------------


@dataclass
class CriterionEvaluation:
    """Outcome of a primary-criterion check against per-arm means."""
    parsed: bool                  # Did we have a primary_parsed to evaluate?
    criterion_met: Optional[bool] # None when not parseable.
    delta: Optional[float]        # Raw delta in metric units, arm_a - arm_b.
    effective_threshold: Optional[float]
    reason: str                   # Human-readable explanation for the dashboard.


def _evaluate_criterion(
    primary_parsed: Optional[Dict[str, Any]],
    arm_means: Dict[str, float],
) -> CriterionEvaluation:
    """Apply the structured primary criterion to the computed per-arm
    means.

    Returns a CriterionEvaluation. parsed=False when no primary_parsed
    was provided OR it referenced arms / metrics we couldn't find.

    Threshold semantics:
      threshold_kind="relative" -> effective = abs(threshold * mean(arm_b))
      threshold_kind="absolute" -> effective = threshold
    """
    if not primary_parsed:
        return CriterionEvaluation(
            parsed=False, criterion_met=None, delta=None,
            effective_threshold=None,
            reason="No primary_parsed; manual verdict from per-arm stats.")

    arm_a = primary_parsed.get("arm_a")
    arm_b = primary_parsed.get("arm_b")
    comparator = primary_parsed.get("comparator", ">=")
    threshold = primary_parsed.get("threshold")
    kind = primary_parsed.get("threshold_kind", "relative")

    if arm_a not in arm_means or arm_b not in arm_means:
        return CriterionEvaluation(
            parsed=False, criterion_met=None, delta=None,
            effective_threshold=None,
            reason=(
                f"Criterion references arms {arm_a!r}/{arm_b!r} but the "
                f"completed jobs only produced arms {sorted(arm_means.keys())}."))

    if threshold is None or not isinstance(threshold, (int, float)):
        return CriterionEvaluation(
            parsed=False, criterion_met=None, delta=None,
            effective_threshold=None,
            reason=f"Criterion threshold {threshold!r} is not numeric.")

    mean_a = arm_means[arm_a]
    mean_b = arm_means[arm_b]
    delta = mean_a - mean_b

    if kind == "relative":
        # Relative threshold is a fraction of arm_b's mean magnitude.
        # If arm_b's mean is 0 the relative comparison is degenerate;
        # we still compute it but tag the reason for the operator.
        baseline_mag = abs(mean_b)
        if baseline_mag == 0:
            effective_threshold = float(threshold)
            reason_prefix = (
                "Baseline arm has mean=0; using threshold as absolute "
                "since relative comparison is degenerate. ")
        else:
            effective_threshold = float(threshold) * baseline_mag
            reason_prefix = ""
    elif kind == "absolute":
        effective_threshold = float(threshold)
        reason_prefix = ""
    else:
        return CriterionEvaluation(
            parsed=False, criterion_met=None, delta=None,
            effective_threshold=None,
            reason=f"Unknown threshold_kind={kind!r}.")

    # Apply comparator.
    if comparator == ">=":
        criterion_met = delta >= effective_threshold
    elif comparator == ">":
        criterion_met = delta > effective_threshold
    elif comparator == "<=":
        criterion_met = delta <= effective_threshold
    elif comparator == "<":
        criterion_met = delta < effective_threshold
    else:
        return CriterionEvaluation(
            parsed=False, criterion_met=None, delta=delta,
            effective_threshold=effective_threshold,
            reason=f"Unknown comparator {comparator!r}.")

    reason = (
        f"{reason_prefix}delta={delta:.4f}, effective threshold "
        f"({comparator} {effective_threshold:.4f}): "
        f"{'PASS' if criterion_met else 'FAIL'}")
    return CriterionEvaluation(
        parsed=True,
        criterion_met=criterion_met,
        delta=delta,
        effective_threshold=effective_threshold,
        reason=reason,
    )


# ---- Single-proposal ingester --------------------------------------------


@dataclass
class IngestDecision:
    """Whether ingest_one will actually run or skip."""
    eligible: bool
    reason: str


def _can_ingest(proposal_doc: Dict[str, Any]) -> IngestDecision:
    """Return (eligible, reason). eligible=True only when:
      * proposal has training_job_ids (orchestrator has queued them), AND
      * proposal.status is one of the in-flight states (approved /
        implementing / pr_open / training) - we don't re-ingest done/
        failed proposals.
    """
    status = proposal_doc.get("status")
    valid_pre_ingest = (
        constants.STATUS_APPROVED,
        constants.STATUS_IMPLEMENTING,
        constants.STATUS_PR_OPEN,
        constants.STATUS_TRAINING,
    )
    if status not in valid_pre_ingest:
        return IngestDecision(
            eligible=False,
            reason=f"status={status!r} is not in-flight; ingester skips.")
    if not proposal_doc.get("training_job_ids"):
        return IngestDecision(
            eligible=False,
            reason="no training_job_ids stamped yet; orchestrator hasn't queued.")
    return IngestDecision(eligible=True, reason="ready")


def ingest_one(db, proposal_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single proposal end-to-end.

    Returns the updated proposal_doc if work was done, None if the
    proposal wasn't yet eligible (still has non-terminal jobs / no
    jobs queued / already done). All Mongo writes happen as a side
    effect; the return is for tests + the worker's logging.
    """
    decision = _can_ingest(proposal_doc)
    if not decision.eligible:
        return None

    training_job_ids = proposal_doc.get("training_job_ids") or []
    all_terminal, n_ok, n_fail = _all_training_jobs_terminal(
        db, training_job_ids)
    if not all_terminal:
        return None  # not ready yet

    proposal_id = proposal_doc["_id"]
    now = datetime.datetime.now(datetime.timezone.utc)

    # If NOTHING succeeded, flip to failed with a brief note. No stats
    # to compute.
    if n_ok == 0:
        results = {
            "primary_criterion_met": None,
            "primary_delta": None,
            "primary_p_value": None,
            "per_arm": [],
            "verdict": constants.OUTCOME_INCONCLUSIVE,
            "notes": (
                f"All {n_fail} linked TRAIN job(s) failed or were "
                f"cancelled. No data to summarize."),
            "computed_at": now,
            "n_jobs_succeeded": 0,
            "n_jobs_failed": n_fail,
        }
        return _persist_results(
            db, proposal_id, results, constants.STATUS_FAILED, now,
            audit_detail={
                "outcome": constants.OUTCOME_INCONCLUSIVE,
                "n_jobs_failed": n_fail,
                "reason": "no_successful_jobs",
            })

    # Gather per-arm samples + compute stats.
    by_arm = _gather_arm_samples(db, training_job_ids)
    per_arm: List[Dict[str, Any]] = []
    arm_means: Dict[str, float] = {}
    for arm in sorted(by_arm.keys()):
        samples = by_arm[arm]
        mean = _safe_mean(samples)
        med = _safe_median(samples)
        sd = _safe_stddev(samples)
        ci_low, ci_high = _bootstrap_ci_95(samples)
        per_arm.append({
            "name": arm,
            "n_trials": len(samples),
            "mean": mean,
            "median": med,
            "stddev": sd,
            "ci_low_95": ci_low,
            "ci_high_95": ci_high,
        })
        if mean is not None:
            arm_means[arm] = mean

    # Evaluate primary criterion.
    primary_parsed = (
        (proposal_doc.get("success_criteria") or {}).get("primary_parsed"))
    criterion = _evaluate_criterion(primary_parsed, arm_means)

    # Verdict mapping:
    #   primary_criterion_met=True  -> supported
    #   primary_criterion_met=False -> rejected
    #   criterion_met=None  -> inconclusive (parsing failed or no
    #                          structured criterion)
    if criterion.criterion_met is True:
        verdict = constants.OUTCOME_SUPPORTED
    elif criterion.criterion_met is False:
        verdict = constants.OUTCOME_REJECTED
    else:
        verdict = constants.OUTCOME_INCONCLUSIVE

    notes = criterion.reason
    if n_fail:
        notes = (
            f"{notes} "
            f"({n_fail} job(s) failed; {n_ok} succeeded.)")

    results = {
        "primary_criterion_met": criterion.criterion_met,
        "primary_delta": criterion.delta,
        "primary_p_value": None,  # placeholder; Phase 1E doesn't compute
        "per_arm": per_arm,
        "verdict": verdict,
        "notes": notes,
        "computed_at": now,
        "n_jobs_succeeded": n_ok,
        "n_jobs_failed": n_fail,
    }
    return _persist_results(
        db, proposal_id, results, constants.STATUS_DONE, now,
        audit_detail={
            "outcome": verdict,
            "n_jobs_succeeded": n_ok,
            "n_jobs_failed": n_fail,
            "primary_delta": criterion.delta,
            "criterion_met": criterion.criterion_met,
        })


def _persist_results(
    db,
    proposal_id,
    results: Dict[str, Any],
    next_status: str,
    now: datetime.datetime,
    *,
    audit_detail: Dict[str, Any],
) -> Dict[str, Any]:
    """Atomic-ish proposal update: results + status + audit event."""
    audit_event = {
        "at": now,
        "by_agent": constants.AGENT_OUTCOME_INGESTER,
        "event": "ingested",
        "detail": audit_detail,
    }
    update = {
        "$set": {
            "results": results,
            "status": next_status,
            "updated_at": now,
        },
        "$push": {"audit_events": audit_event},
    }
    db[constants.COLL_PROPOSALS].update_one({"_id": proposal_id}, update)
    return db[constants.COLL_PROPOSALS].find_one({"_id": proposal_id})


# ---- Worker loop ---------------------------------------------------------


def outcome_loop(
    db,
    *,
    poll_interval_seconds: int = 300,
    should_stop_fn=None,
):
    """Poll for in-flight proposals whose TRAIN jobs have all terminated.

    Cheaper than judge_loop: just Mongo queries + pure-Python stats,
    no LLM. We poll less aggressively (default 5 min vs 30s for the
    judge) because the bottleneck is "TRAIN jobs finishing", which
    takes hours.

    Processes all eligible proposals per cycle (unlike judge_loop
    which processes one). There's no budget cost so no reason to
    rate-limit.
    """
    print(
        f"outcome_ingester: starting loop, poll interval = "
        f"{poll_interval_seconds}s",
        flush=True)
    while True:
        if should_stop_fn is not None and should_stop_fn():
            print("outcome_ingester: stop signal received; exiting.",
                  flush=True)
            return
        try:
            # Recovery sweep BEFORE the normal ingestion pass. Catches
            # proposals that were prematurely flipped to `failed` by an
            # earlier ingest cycle (e.g. trainer crashed before any
            # job could finish, marking them all FAILED) but whose
            # underlying jobs have since been re-queued (NOT_STARTED /
            # IN_PROGRESS). Moves them back to `training` so the
            # normal loop below can re-evaluate when the jobs
            # ultimately terminate.
            recovered = recover_stale_failures(db)
            for rid in recovered:
                print(
                    f"outcome_ingester: recovered stale-failed proposal "
                    f"{rid} -> status=training (jobs back in non-terminal "
                    f"state)",
                    flush=True)

            ingestible_statuses = [
                constants.STATUS_APPROVED,
                constants.STATUS_IMPLEMENTING,
                constants.STATUS_PR_OPEN,
                constants.STATUS_TRAINING,
            ]
            candidates = db[constants.COLL_PROPOSALS].find(
                {"status": {"$in": ingestible_statuses},
                 "training_job_ids": {"$exists": True, "$ne": []}})
            for prop in candidates:
                try:
                    result = ingest_one(db, prop)
                    if result is not None:
                        verdict = (result.get("results") or {}).get("verdict")
                        new_status = result.get("status")
                        print(
                            f"outcome_ingester: ingested {prop.get('_id')} "
                            f"-> status={new_status} verdict={verdict}",
                            flush=True)
                except Exception as e:  # noqa: BLE001
                    tb = traceback.format_exc()
                    print(
                        f"outcome_ingester: error on {prop.get('_id')}: "
                        f"{type(e).__name__}: {e}\n{tb}",
                        flush=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"outcome_ingester: Mongo lookup failed: {e}",
                flush=True)
        time.sleep(poll_interval_seconds)


def recover_stale_failures(db) -> List[Any]:
    """Scan for stale-failed proposals and reset them to `training`.

    A proposal is "stale-failed" when:
      * proposal.status == 'failed'
      * its last audit_event is an `ingested` event whose
        detail.reason == 'no_successful_jobs' (i.e., we marked it
        failed BECAUSE all linked TRAIN jobs were in FAILED state at
        the time, not because of any LLM / orchestrator decision)
      * AT LEAST ONE of its linked TRAIN jobs is currently in a non-
        terminal status (NOT_STARTED / IN_PROGRESS / PAUSED), meaning
        the operator has since re-queued at least one job and we
        should re-evaluate when training completes.

    For each match we:
      * flip status back to `training`
      * clear proposal.results (so the next ingest produces fresh
        per-arm stats from the recovered jobs)
      * append an audit_event recording the recovery

    Returns the list of recovered proposal _ids.

    Idempotent: a proposal that's already been recovered won't match
    again (status != 'failed' after the first recovery).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    candidates = db[constants.COLL_PROPOSALS].find(
        {"status": constants.STATUS_FAILED,
         "training_job_ids": {"$exists": True, "$ne": []},
         # Conservative gate: only recover proposals that were marked
         # failed BY the outcome_ingester with the specific "no
         # successful jobs" signature. Don't touch proposals failed
         # by the orchestrator (those have a different audit event)
         # or by the cursor agent.
         "results.notes": {"$regex": "TRAIN job.* failed or were cancelled"}})
    recovered_ids: List[Any] = []
    for prop in candidates:
        training_job_ids = prop.get("training_job_ids") or []
        if not training_job_ids:
            continue
        # Any job back in a non-terminal state?
        has_non_terminal = db.jobs.count_documents({
            "_id": {"$in": list(training_job_ids)},
            "status": {"$in": list(_NON_TERMINAL_JOB_STATUSES)},
        }) > 0
        if not has_non_terminal:
            continue

        audit_event = {
            "at": now,
            "by_agent": constants.AGENT_OUTCOME_INGESTER,
            "event": "recovered_from_stale_failure",
            "detail": {
                "previous_status": constants.STATUS_FAILED,
                "next_status": constants.STATUS_TRAINING,
                "reason": "jobs_re_queued_after_stale_failure",
            },
        }
        db[constants.COLL_PROPOSALS].update_one(
            {"_id": prop["_id"], "status": constants.STATUS_FAILED},
            {
                "$set": {
                    "status": constants.STATUS_TRAINING,
                    "updated_at": now,
                },
                "$unset": {"results": ""},
                "$push": {"audit_events": audit_event},
            },
        )
        recovered_ids.append(prop["_id"])
    return recovered_ids
