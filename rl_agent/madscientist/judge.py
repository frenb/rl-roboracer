"""Judge worker - applies JUDGE_RUBRIC.md to proposals.

Pipeline for each proposal with status="pending_judge":

  1. Run pre_rubric_checks.run_all (the 7 deterministic checks A-G).
     If any fail, write a rejection judge_review and advance status to
     "rejected" WITHOUT calling the LLM. Saves ~$0.50 per rejection.

  2. Otherwise, call Anthropic Claude with:
       * system prompt = role description + the FULL rubric markdown
         (loaded from db.judge_rubric_history at its latest version)
       * user prompt = the proposal dict, formatted as compact JSON

     The model is instructed to return ONE JSON object exactly matching
     the schema in JUDGE_RUBRIC.md section 5.

  3. Parse + validate the JSON response. If parse fails, retry once
     with a stricter "JSON only, no markdown fences" reminder. If
     parse still fails, write a "judge_failure" rejection so the
     operator sees it on the dashboard.

  4. Mechanically compute the overall verdict + normalized_score from
     the parsed scores per JUDGE_RUBRIC.md section 4. This is NOT
     trusted to the LLM - prevents verdict inflation.

  5. Construct a JudgeReview, persist to proposal.judge_review,
     advance status to "pending_user" (or "rejected" if verdict is
     "reject"), and stamp an audit_event.

Module is import-side-effect-free: no Mongo or Anthropic clients
are opened at import time. The main loop opens them on demand.

Phase 1A: judge_loop is callable but not yet wired into main.py;
the next commit does that. For now, the module is unit-testable
in isolation via test_judge.py (uses a mocked Anthropic client).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import constants
from . import pre_rubric_checks


# ---- Configuration -------------------------------------------------------

# Anthropic model id for the Judge. Per the user's earlier decision
# (b: same model for MadScientist and Judge), this matches the
# planned Researcher model. Override via env if needed.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

# Approximate Anthropic pricing per million tokens (input / output).
# Used for cost attribution on proposal.cost.judge_usd. Updated
# from https://www.anthropic.com/pricing as needed. Pricing changes
# rarely; an annual review is fine.
_JUDGE_INPUT_USD_PER_MTOK = 15.0
_JUDGE_OUTPUT_USD_PER_MTOK = 75.0


# ---- Rubric loader -------------------------------------------------------


def get_latest_rubric(db) -> Dict[str, Any]:
    """Return the highest-version document from db.judge_rubric_history.

    Falls back to version=0 placeholder if no real rubric exists. The
    judge_loop refuses to score against version=0 (the placeholder
    just says "rubric not yet authored").
    """
    doc = db[constants.COLL_JUDGE_RUBRIC_HISTORY].find_one(
        sort=[("version", -1)])
    if not doc:
        raise RuntimeError(
            "No rubric in judge_rubric_history. Run seed.py first.")
    return doc


# ---- Prompt builder + response parser ------------------------------------


_JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are the Judge agent for the MadScientist research system. Your job is to evaluate a single research proposal against the rubric below, then emit a structured JSON review.

You are independent (not adversarial). Surface concerns to the operator without replacing their judgment. Even on accepts, list at least one concern unless the verdict is strong_accept.

Score each of the 8 axes 0-5 per the level anchors in the rubric. Mechanically follow the anchors - do not improvise scoring categories.

If an axis is genuinely N/A (e.g., a pure-codebase proposal has no source papers, so paper_faithfulness=N/A), add its key to axes_skipped instead of giving it a score.

Be terse in strengths/concerns/suggested_revisions - 1 sentence per item.

OUTPUT FORMAT: Return EXACTLY one JSON object (no markdown fences, no commentary, no leading text). It must match this schema:

{{
  "scores": {{
    "hypothesis_specificity": 0-5,
    "novelty": 0-5,
    "significance": 0-5,
    "statistical_power_and_baseline_rigor": 0-5,
    "goodhart_resistance": 0-5,
    "paper_faithfulness": 0-5,
    "implementation_feasibility": 0-5,
    "cost_and_reproducibility": 0-5
  }},
  "axes_skipped": ["..."],
  "strengths": ["...", "..."],
  "concerns": ["...", "..."],
  "suggested_revisions": ["...", "..."]
}}

You do NOT emit the "overall" verdict, "normalized_score", or any of the meta fields (judge_model, judged_at, etc.) - the judge worker computes those mechanically from your scores.

Do NOT include any axes_skipped entries unless that axis genuinely cannot be scored. Do NOT score axes you put in axes_skipped.

===== BEGIN RUBRIC =====
{rubric_markdown}
===== END RUBRIC =====
"""


_JUDGE_USER_PROMPT_TEMPLATE = """Review this proposal. Return the JSON object specified in the system prompt - nothing else.

```json
{proposal_json}
```
"""


def build_judge_prompts(
    rubric_markdown: str, proposal_doc: Dict[str, Any],
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Anthropic call."""
    # Compact the proposal so we don't waste tokens on indentation but
    # keep it human-readable enough that the LLM doesn't choke.
    proposal_for_prompt = _clean_proposal_for_prompt(proposal_doc)
    proposal_json = json.dumps(
        proposal_for_prompt, indent=2, default=str, sort_keys=False)
    return (
        _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
            rubric_markdown=rubric_markdown),
        _JUDGE_USER_PROMPT_TEMPLATE.format(
            proposal_json=proposal_json),
    )


def _clean_proposal_for_prompt(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip fields the Judge doesn't need to see.

    Removes:
      * _id, audit_events, implementation_log, training_job_ids,
        results, cost - judge doesn't decide based on these
      * Any field still null/empty (compresses prompt size)

    Keeps:
      * everything scientifically relevant: hypothesis, motivation,
        source_papers, experiment_arms, success_criteria,
        code_changes_summary, n_seeds_per_arm,
        num_iterations_per_seed, expected_wall_time_hours
    """
    drop_keys = {
        "_id", "audit_events", "implementation_log", "training_job_ids",
        "results", "cost", "implementation_started_at",
        "implementation_branch", "implementation_pr_url",
        "implementation_finished_at", "implementation_failure_reason",
        "decision", "judge_review", "updated_at",
    }
    cleaned: Dict[str, Any] = {}
    for k, v in doc.items():
        if k in drop_keys:
            continue
        if v is None or v == [] or v == {} or v == "":
            continue
        cleaned[k] = v
    return cleaned


# Matches a JSON object that starts with { and (greedily, but
# balanced) finds the matching closing }. We use this to extract the
# JSON object from the LLM's response even when it wraps it in
# explanatory text or markdown fences.
_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Extract the JSON object from the LLM's response.

    Handles three cases:
      1. Response is pure JSON - parse it directly.
      2. Response wraps the JSON in markdown fences ```json ... ```
         - strip the fences first.
      3. Response has prose before/after the JSON - find the first
         well-formed top-level object via regex.

    Raises ValueError if no parseable JSON is found.
    """
    text = text.strip()

    # Try pure JSON first - fastest path.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown-fenced JSON.
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: scan for any well-formed top-level object.
    # Iterates over candidates so a stray `{` inside prose doesn't
    # short-circuit us on the wrong substring.
    for m in _JSON_BLOCK_RE.finditer(text):
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"No parseable JSON object found in Judge response (first 200 "
        f"chars: {text[:200]!r})")


# ---- Verdict computation -------------------------------------------------


# Per JUDGE_RUBRIC.md section 4.
_AXIS_KEYS = (
    "hypothesis_specificity",
    "novelty",
    "significance",
    "statistical_power_and_baseline_rigor",
    "goodhart_resistance",
    "paper_faithfulness",
    "implementation_feasibility",
    "cost_and_reproducibility",
)
_MAX_PER_AXIS = 5

# Normalized-score thresholds per JUDGE_RUBRIC.md section 4.
_VERDICT_THRESHOLDS = (
    (0.875, "strong_accept"),
    (0.700, "accept"),
    (0.525, "weak_accept"),
    (0.350, "weak_reject"),
    (0.000, "reject"),  # catch-all floor
)


@dataclass
class VerdictComputation:
    """Mechanical-verdict output, kept separate from the JudgeReview
    so the test suite can assert on the math independently of the
    Pydantic layer."""
    overall: str
    normalized_score: float
    raw_sum: int
    max_possible: int
    bumped: bool  # whether the §4 robust-proposal one-notch bump was applied


def compute_verdict(
    scores: Dict[str, int],
    axes_skipped: List[str],
    *,
    apply_robustness_bump: bool = False,
) -> VerdictComputation:
    """Compute the overall verdict per JUDGE_RUBRIC.md section 4.

    Rules:
      * Sum scores for axes NOT in axes_skipped.
      * Normalize by (5 * number of scored axes).
      * Map normalized -> verdict via thresholds.
      * OVERRIDE: any axis scored 0 -> force overall=reject.
      * OPTIONAL BUMP: if apply_robustness_bump and the proposal
        scored 5 on both goodhart_resistance and paper_faithfulness,
        bump the verdict one notch up (e.g., weak_accept -> accept).

    Returns VerdictComputation with the full diagnostic context so
    the dashboard / Judge can show the math.
    """
    # Validate inputs.
    valid_axes = set(_AXIS_KEYS)
    skipped = set(axes_skipped or [])
    bad = skipped - valid_axes
    if bad:
        raise ValueError(f"Unknown axes in axes_skipped: {sorted(bad)}")

    scored_keys = [k for k in _AXIS_KEYS if k not in skipped]
    raw_sum = 0
    for k in scored_keys:
        v = scores.get(k)
        if not isinstance(v, int):
            raise ValueError(
                f"Score for {k!r} must be int 0-5, got {v!r}")
        if v < 0 or v > _MAX_PER_AXIS:
            raise ValueError(
                f"Score for {k!r} must be 0-5, got {v}")
        raw_sum += v

    max_possible = len(scored_keys) * _MAX_PER_AXIS
    normalized = raw_sum / max_possible if max_possible else 0.0

    # Map to verdict via thresholds.
    verdict = "reject"
    for threshold, label in _VERDICT_THRESHOLDS:
        if normalized >= threshold:
            verdict = label
            break

    # Override: any axis scored 0 = forced reject.
    has_zero = any(
        scores.get(k) == 0 for k in scored_keys)
    if has_zero:
        verdict = "reject"

    # Optional one-notch bump on robust-proposal markers.
    bumped = False
    if (apply_robustness_bump
            and not has_zero
            and scores.get("goodhart_resistance") == 5
            and scores.get("paper_faithfulness") == 5):
        order = ["reject", "weak_reject", "weak_accept", "accept", "strong_accept"]
        idx = order.index(verdict)
        if idx < len(order) - 1:
            verdict = order[idx + 1]
            bumped = True

    return VerdictComputation(
        overall=verdict,
        normalized_score=normalized,
        raw_sum=raw_sum,
        max_possible=max_possible,
        bumped=bumped,
    )


# ---- Cost attribution ----------------------------------------------------


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Anthropic-Opus-ish pricing. Tunable via env if pricing changes."""
    return (
        (input_tokens / 1_000_000.0) * _JUDGE_INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000.0) * _JUDGE_OUTPUT_USD_PER_MTOK)


# ---- Anthropic call -------------------------------------------------------


def call_anthropic_judge(
    anthropic_client,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 2048,
):
    """Wrapper around anthropic.messages.create that returns a
    structured result. Decoupled from the rest of the file so tests
    can pass a mock client.

    Returns dict with: {text, input_tokens, output_tokens, stop_reason}.
    Raises on transport/RPC error - the caller is expected to wrap in
    try/except.
    """
    resp = anthropic_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Anthropic v0.39 returns resp.content as a list of content blocks
    # (TextBlock, ToolUseBlock, etc.). We expect a single TextBlock for
    # the Judge's JSON-only response. Concatenate just in case.
    text_parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            text_parts.append(block["text"])
    text = "".join(text_parts)
    return {
        "text": text,
        "input_tokens": getattr(resp.usage, "input_tokens", 0),
        "output_tokens": getattr(resp.usage, "output_tokens", 0),
        "stop_reason": getattr(resp, "stop_reason", None),
    }


# ---- Single-proposal handler ---------------------------------------------


def judge_one(
    db,
    proposal_doc: Dict[str, Any],
    anthropic_client,
    *,
    monthly_budget_usd: float = 250.0,
    spent_so_far_usd: float = 0.0,
    probe_arxiv_urls: bool = False,
) -> Dict[str, Any]:
    """Score a single proposal end-to-end.

    Args:
        db: pymongo database object.
        proposal_doc: raw proposal dict (as from db.proposals.find_one).
        anthropic_client: instance of anthropic.Anthropic (or a mock
            with the same .messages.create interface).
        monthly_budget_usd: cap for check D.
        spent_so_far_usd: month-to-date spend for check D.
        probe_arxiv_urls: when True, check A hits arxiv.org/HEAD to
            verify cited papers actually exist. Set False in tests.

    Returns the updated proposal_doc with judge_review + new status
    populated. The DB is updated as a side effect.
    """
    proposal_id = proposal_doc.get("_id")
    now = datetime.datetime.now(datetime.timezone.utc)

    # ---- Step 1: pre-rubric checks --------------------------------------
    check_result = pre_rubric_checks.run_all(
        proposal_doc,
        monthly_budget_usd=monthly_budget_usd,
        spent_so_far_usd=spent_so_far_usd,
        probe_urls=probe_arxiv_urls,
    )

    rubric = get_latest_rubric(db)
    rubric_version = int(rubric.get("version", 0))

    if not check_result.all_passed:
        # Short-circuit: reject without calling the LLM.
        failed_letters = [r.check_id for r in check_result.failed]
        concerns = [
            f"Pre-rubric check {r.check_id} failed: {r.reason}"
            for r in check_result.failed]
        review = {
            "overall": "reject",
            "scores": {k: 0 for k in _AXIS_KEYS},
            "axes_skipped": [],
            "normalized_score": 0.0,
            "strengths": [],
            "concerns": concerns,
            "suggested_revisions": [],
            "pre_rubric_check_failures": failed_letters,
            "judged_at": now,
            "rubric_version": rubric_version,
            "judge_model": JUDGE_MODEL,
            "judge_token_count": 0,
            "judge_cost_usd": 0.0,
        }
        return _persist_review_and_status(
            db, proposal_id, review, constants.STATUS_REJECTED, now,
            audit_detail={
                "reason": "pre_rubric_check_failure",
                "failed": failed_letters,
            })

    # ---- Step 2: call the LLM -------------------------------------------
    if rubric_version == 0:
        # Refuse to score against the placeholder rubric.
        review = {
            "overall": "reject",
            "scores": {k: 0 for k in _AXIS_KEYS},
            "axes_skipped": [],
            "normalized_score": 0.0,
            "strengths": [],
            "concerns": [
                "Judge cannot score against placeholder rubric version=0. "
                "Author rl_agent/madscientist/JUDGE_RUBRIC.md and re-seed."],
            "suggested_revisions": [],
            "pre_rubric_check_failures": [],
            "judged_at": now,
            "rubric_version": 0,
            "judge_model": JUDGE_MODEL,
            "judge_token_count": 0,
            "judge_cost_usd": 0.0,
        }
        return _persist_review_and_status(
            db, proposal_id, review, constants.STATUS_REJECTED, now,
            audit_detail={"reason": "placeholder_rubric"})

    rubric_md = rubric["rubric_markdown"]
    system_prompt, user_prompt = build_judge_prompts(rubric_md, proposal_doc)

    parsed: Optional[Dict[str, Any]] = None
    llm_response_meta = {"input_tokens": 0, "output_tokens": 0}
    last_error: Optional[str] = None
    # One retry on parse failure with a stricter reminder. After two
    # failures, give up.
    for attempt in (0, 1):
        try:
            llm_response_meta = call_anthropic_judge(
                anthropic_client, system_prompt, user_prompt)
            parsed = parse_judge_response(llm_response_meta["text"])
            break
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            if attempt == 0:
                user_prompt = (
                    user_prompt
                    + "\n\nREMINDER: your previous response could not be "
                    "parsed as JSON. Return ONLY the JSON object - no "
                    "markdown fences, no commentary.")
                continue

    if parsed is None:
        # LLM failed twice. Stamp a failure on the proposal so it
        # shows up in the dashboard's outcomes list.
        review = {
            "overall": "reject",
            "scores": {k: 0 for k in _AXIS_KEYS},
            "axes_skipped": [],
            "normalized_score": 0.0,
            "strengths": [],
            "concerns": [
                f"Judge LLM call failed: {last_error}. The proposal is "
                f"automatically rejected but may be re-submittable after "
                f"investigating the LLM issue."],
            "suggested_revisions": [],
            "pre_rubric_check_failures": [],
            "judged_at": now,
            "rubric_version": rubric_version,
            "judge_model": JUDGE_MODEL,
            "judge_token_count": (
                llm_response_meta["input_tokens"]
                + llm_response_meta["output_tokens"]),
            "judge_cost_usd": _estimate_cost_usd(
                llm_response_meta["input_tokens"],
                llm_response_meta["output_tokens"]),
        }
        return _persist_review_and_status(
            db, proposal_id, review, constants.STATUS_REJECTED, now,
            audit_detail={
                "reason": "llm_parse_failure",
                "last_error": last_error,
            })

    # ---- Step 3: validate the parsed response ---------------------------
    scores_in = parsed.get("scores") or {}
    axes_skipped = parsed.get("axes_skipped") or []
    # Coerce skipped axes to a list of strings.
    if not isinstance(axes_skipped, list):
        axes_skipped = []
    # Coerce score values to int (LLMs occasionally emit floats).
    scores: Dict[str, int] = {}
    for k in _AXIS_KEYS:
        if k in axes_skipped:
            continue
        v = scores_in.get(k)
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            iv = 0
        scores[k] = max(0, min(_MAX_PER_AXIS, iv))

    # ---- Step 4: compute verdict mechanically ---------------------------
    verdict = compute_verdict(scores, axes_skipped, apply_robustness_bump=True)

    # ---- Step 5: build review + persist ---------------------------------
    input_tokens = llm_response_meta["input_tokens"]
    output_tokens = llm_response_meta["output_tokens"]
    cost = _estimate_cost_usd(input_tokens, output_tokens)

    # Ensure non-empty concerns if not strong_accept (per rubric §5).
    concerns = parsed.get("concerns") or []
    if verdict.overall != "strong_accept" and not concerns:
        concerns = ["(judge produced no explicit concerns - review the proposal manually)"]

    review = {
        "overall": verdict.overall,
        "scores": scores,
        "axes_skipped": list(axes_skipped),
        "normalized_score": round(verdict.normalized_score, 4),
        "strengths": parsed.get("strengths") or [],
        "concerns": concerns,
        "suggested_revisions": parsed.get("suggested_revisions") or [],
        "pre_rubric_check_failures": [],
        "judged_at": now,
        "rubric_version": rubric_version,
        "judge_model": JUDGE_MODEL,
        "judge_token_count": input_tokens + output_tokens,
        "judge_cost_usd": round(cost, 6),
        "robustness_bump_applied": verdict.bumped,
    }

    # Move to pending_user unless the verdict is "reject" (in which
    # case skip the operator's queue entirely - rejection is final
    # without user action).
    next_status = (
        constants.STATUS_REJECTED if verdict.overall == "reject"
        else constants.STATUS_PENDING_USER)

    return _persist_review_and_status(
        db, proposal_id, review, next_status, now,
        audit_detail={
            "verdict": verdict.overall,
            "normalized_score": review["normalized_score"],
            "rubric_version": rubric_version,
            "tokens": review["judge_token_count"],
            "cost_usd": review["judge_cost_usd"],
        })


def _persist_review_and_status(
    db,
    proposal_id,
    review: Dict[str, Any],
    next_status: str,
    now: datetime.datetime,
    *,
    audit_detail: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper: $set judge_review + status + updated_at, $push audit_event.

    Also bumps proposal.cost.judge_usd by the new judge_cost_usd
    so the budget gauge stays accurate.
    """
    audit_event = {
        "at": now,
        "by_agent": constants.AGENT_JUDGE,
        "event": "judged",
        "detail": audit_detail,
    }
    update = {
        "$set": {
            "judge_review": review,
            "status": next_status,
            "updated_at": now,
        },
        "$inc": {
            "cost.judge_usd": float(review.get("judge_cost_usd") or 0.0),
        },
        "$push": {"audit_events": audit_event},
    }
    db[constants.COLL_PROPOSALS].update_one({"_id": proposal_id}, update)
    return db[constants.COLL_PROPOSALS].find_one({"_id": proposal_id})


# ---- Worker loop ---------------------------------------------------------


def judge_loop(
    db,
    anthropic_client,
    *,
    poll_interval_seconds: int = 30,
    should_stop_fn=None,
):
    """Watch db.proposals for pending_judge entries and process them.

    Polls every `poll_interval_seconds`. One proposal per cycle (FIFO
    by created_at) so the budget cap stays predictable.

    should_stop_fn: callable returning True when the loop should exit
        gracefully. Used by tests + by the SIGTERM handler in main.py.
    """
    print(
        f"judge: starting loop, poll interval = {poll_interval_seconds}s, "
        f"model = {JUDGE_MODEL}",
        flush=True)
    while True:
        if should_stop_fn is not None and should_stop_fn():
            print("judge: stop signal received; exiting loop.", flush=True)
            return
        try:
            pending = db[constants.COLL_PROPOSALS].find_one(
                {"status": constants.STATUS_PENDING_JUDGE},
                sort=[("created_at", 1)])
        except Exception as e:  # noqa: BLE001
            print(f"judge: Mongo lookup failed: {e}", flush=True)
            time.sleep(poll_interval_seconds)
            continue
        if pending is None:
            time.sleep(poll_interval_seconds)
            continue
        proposal_id = pending.get("_id")
        print(
            f"judge: processing proposal {proposal_id} "
            f"({pending.get('title', '?')!r})",
            flush=True)
        try:
            month_spent_usd = _running_month_spend(db)
            judge_one(
                db, pending, anthropic_client,
                spent_so_far_usd=month_spent_usd,
                probe_arxiv_urls=True)
            print(
                f"judge: finished proposal {proposal_id}",
                flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(
                f"judge: uncaught error processing {proposal_id}: "
                f"{type(e).__name__}: {e}\n{tb}",
                flush=True)
            # Don't crash the worker. Mark the proposal as failed so
            # we don't re-pick it up forever.
            try:
                db[constants.COLL_PROPOSALS].update_one(
                    {"_id": proposal_id},
                    {"$set": {
                        "status": constants.STATUS_FAILED,
                        "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        "judge_review.judge_error": f"{type(e).__name__}: {e}",
                    }})
            except Exception:
                pass
        # Sleep even after work so we don't loop tightly if Anthropic
        # is flaky (every pending proposal would otherwise re-try
        # immediately).
        time.sleep(poll_interval_seconds)


def _running_month_spend(db) -> float:
    """Sum cost.total of all proposals created in the current month.

    Used by judge_loop's check D budget input. Calendar-month boundaries
    (UTC); a rolling 30d window would also work but calendar is simpler
    for an operator to mentally track.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    month_start = datetime.datetime(now.year, now.month, 1, tzinfo=datetime.timezone.utc)
    pipeline = [
        {"$match": {"created_at": {"$gte": month_start}}},
        {"$group": {
            "_id": None,
            "total": {"$sum": {
                "$add": [
                    {"$ifNull": ["$cost.madscientist_usd", 0.0]},
                    {"$ifNull": ["$cost.judge_usd", 0.0]},
                    {"$ifNull": ["$cost.cursor_usd", 0.0]},
                    {"$ifNull": ["$cost.other_usd", 0.0]},
                ]
            }},
        }},
    ]
    try:
        rows = list(db[constants.COLL_PROPOSALS].aggregate(pipeline))
    except Exception:  # noqa: BLE001
        return 0.0
    if not rows:
        return 0.0
    return float(rows[0].get("total") or 0.0)
