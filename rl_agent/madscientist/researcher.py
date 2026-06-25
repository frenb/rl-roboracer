"""Researcher worker - autonomous proposal generation (Phase 1B).

One cycle ~ every RESEARCH_CYCLE_INTERVAL_SECONDS (default 6h):

  1. Budget + rate-limit gate: refuse to start if MAX_PROPOSALS_PER_DAY
     would be exceeded OR BUDGET_USD_PER_MONTH already spent.

  2. Survey arxiv: fetch recent abstracts matching the research
     keywords (default: "reinforcement learning behavior cloning
     imitation learning autonomous driving"). Filter to the last
     N days via SubmittedDate sort.

  3. Inspect codebase state:
       * experiment_designs SCHEMA (which knobs exist + their types
         + defaults).
       * Existing named experiment_designs in db (so the proposal can
         cite + diff from them for novelty).
       * Top-K past models by avg_return (so the LLM knows what's
         worked + what's been tried).
       * Current budget + cap status.

  4. Build prompts: condensed rubric (axes + anchors only, NOT the
     full markdown) + abstracts + codebase context. Ask Claude to
     produce ONE Proposal JSON matching schemas.Proposal.

  5. Parse + self-critique: run the proposed proposal through
     pre_rubric_checks.run_all. If any fail, send the failure list
     back to Claude with a "revise to address these" prompt. Max 2
     retries; abandon if still failing.

  6. Submit: insert into db.proposals with status=pending_judge,
     stamp proposal.cost.madscientist_usd, audit_event. The Judge
     worker (Phase 1A) picks it up within ~30s.

Cost per cycle is ~$0.20-0.30 with Claude Opus (~10k input tokens +
~2k output). At 4 cycles/day max (gated by MAX_PROPOSALS_PER_DAY=1
default), monthly spend ~$10-15 of the $250 budget.

What this MVP does NOT do:
  * Read full PDFs (token cost prohibitive at this cadence).
  * Generate multiple candidate proposals per cycle.
  * Maintain a vector store of past papers for similarity retrieval.
  * Self-revise the topic / query based on results so far.

Those are Phase 3 ("multi-step research plans"). Phase 1B ships a
useful single-cycle proposal generator that any human-in-the-loop
operator can rely on for daily research suggestions.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId

from . import constants
from . import pre_rubric_checks


# ---- Configuration -------------------------------------------------------

RESEARCHER_MODEL = os.environ.get("RESEARCHER_MODEL", "claude-opus-4-7")

# Same pricing constants as Judge. Tunable via env if pricing changes.
_RESEARCHER_INPUT_USD_PER_MTOK = 15.0
_RESEARCHER_OUTPUT_USD_PER_MTOK = 75.0

# arxiv search defaults. Override via env vars in docker-compose.yml.
DEFAULT_RESEARCH_QUERY = (
    "reinforcement learning behavior cloning imitation learning "
    "autonomous driving")
DEFAULT_RESEARCH_DAYS_BACK = 30
DEFAULT_RESEARCH_MAX_PAPERS = 25

# Max retries on self-critique loop (LLM produces a proposal -> we
# run pre-rubric checks -> if any fail, ask LLM to revise -> repeat).
# After this many attempts, abandon the cycle.
DEFAULT_MAX_REVISIONS = 2

# How many past best models to include in the codebase context.
_TOP_K_MODELS = 5


# ---- arxiv fetching ------------------------------------------------------


def fetch_recent_arxiv_papers(
    query: str = DEFAULT_RESEARCH_QUERY,
    *,
    days_back: int = DEFAULT_RESEARCH_DAYS_BACK,
    max_papers: int = DEFAULT_RESEARCH_MAX_PAPERS,
    arxiv_client=None,
) -> List[Dict[str, Any]]:
    """Fetch recent arxiv papers matching the research query.

    Returns list of dicts with {arxiv_id, title, summary, authors,
    published, primary_category}. Sorted newest first; papers older
    than `days_back` are dropped.

    arxiv_client allows tests to inject a fake client. None uses the
    real arxiv package.
    """
    # Lazy import so unit tests without network access can monkey-
    # patch this function entirely.
    import arxiv

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days_back)

    if arxiv_client is None:
        arxiv_client = arxiv.Client(
            page_size=max_papers,
            delay_seconds=3.0,  # be polite to arxiv
            num_retries=2,
        )

    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers: List[Dict[str, Any]] = []
    for result in arxiv_client.results(search):
        pub = result.published
        # Coerce to UTC-aware for the cutoff comparison.
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=datetime.timezone.utc)
        if pub < cutoff:
            continue
        # Extract just the arxiv id (e.g., "2104.06129v2") from the URL.
        m = re.search(r"abs/([\w./-]+)$", result.entry_id)
        arxiv_id = m.group(1) if m else result.entry_id
        papers.append({
            "arxiv_id": arxiv_id,
            "title": result.title.strip(),
            "summary": result.summary.strip(),
            "authors": [a.name for a in result.authors],
            "published": pub.isoformat(),
            "primary_category": getattr(result, "primary_category", "?"),
        })
    return papers


# ---- Codebase + budget context -------------------------------------------


def fetch_codebase_context(db) -> Dict[str, Any]:
    """Gather what the LLM needs to know about the project state.

    Returns a dict the prompt builder JSON-serializes:
      * experiment_designs_schema: compact view of SCHEMA (field name
        -> {type, default, doc}). Skips section headers.
      * existing_experiment_designs: list of named designs in db
        (name, version, key field overrides).
      * existing_reward_designs: list of named reward designs (name,
        version, brief description).
      * top_models: top_K models by avg_return, with arm + design
        provenance.
      * recent_proposals: last 5 proposals + their outcomes.
    """
    # Schema dump - import lazily so tests don't require the trainer
    # package on import.
    try:
        from rl_agent.experiment_designs import SCHEMA as ED_SCHEMA
    except ImportError:
        ED_SCHEMA = {}
    schema_compact = {}
    for k, v in ED_SCHEMA.items():
        if k.startswith("_section_") or not isinstance(v, dict):
            continue
        schema_compact[k] = {
            "type": v.get("type"),
            "default": v.get("default"),
            "doc": (v.get("doc") or "")[:200],  # truncate verbose docs
        }

    # Named experiment_designs (limit 20).
    existing_designs = []
    try:
        for d in db.experiment_designs.find(
                {}, {"_id": 1, "name": 1, "version": 1}).limit(20):
            existing_designs.append({
                "_id": str(d.get("_id")),
                "name": d.get("name"),
                "version": d.get("version"),
            })
    except Exception:  # noqa: BLE001
        pass

    # Named reward_designs (limit 10).
    existing_rewards = []
    try:
        for d in db.reward_designs.find(
                {"archived": {"$ne": True}},
                {"_id": 1, "name": 1, "version": 1, "description": 1}).limit(10):
            existing_rewards.append({
                "_id": str(d.get("_id")),
                "name": d.get("name"),
                "version": d.get("version"),
                "description": (d.get("description") or "")[:200],
            })
    except Exception:  # noqa: BLE001
        pass

    # Top-K models by avg_return.
    top_models = []
    try:
        for m in db.models.find(
                {"avg_return": {"$ne": None}, "is_global_best": True},
                {"avg_return": 1, "experiment_design_name": 1,
                 "reward_design_name": 1, "create_date": 1, "job_id": 1}
                ).sort("avg_return", -1).limit(_TOP_K_MODELS):
            top_models.append({
                "avg_return": m.get("avg_return"),
                "experiment_design": m.get("experiment_design_name"),
                "reward_design": m.get("reward_design_name"),
                "job_id": str(m.get("job_id")) if m.get("job_id") else None,
            })
    except Exception:  # noqa: BLE001
        pass

    # Recent proposals + outcomes (any status).
    recent_proposals = []
    try:
        for p in db.proposals.find(
                {}, {"title": 1, "status": 1, "results.verdict": 1,
                     "results.primary_delta": 1, "judge_review.overall": 1}
                ).sort("created_at", -1).limit(5):
            recent_proposals.append({
                "title": p.get("title"),
                "status": p.get("status"),
                "judge_verdict": (p.get("judge_review") or {}).get("overall"),
                "outcome_verdict": (p.get("results") or {}).get("verdict"),
                "outcome_delta": (p.get("results") or {}).get("primary_delta"),
            })
    except Exception:  # noqa: BLE001
        pass

    return {
        "experiment_designs_schema": schema_compact,
        "existing_experiment_designs": existing_designs,
        "existing_reward_designs": existing_rewards,
        "top_models_by_avg_return": top_models,
        "recent_proposals": recent_proposals,
    }


def fetch_budget_status(
    db, *, monthly_budget_usd: float, max_proposals_per_day: int,
) -> Dict[str, Any]:
    """Compute current spend + proposal counts vs caps."""
    now = datetime.datetime.now(datetime.timezone.utc)
    month_start = datetime.datetime(
        now.year, now.month, 1, tzinfo=datetime.timezone.utc)
    day_start = datetime.datetime(
        now.year, now.month, now.day, tzinfo=datetime.timezone.utc)

    spent_this_month = 0.0
    try:
        agg = list(db.proposals.aggregate([
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
        ]))
        if agg:
            spent_this_month = float(agg[0].get("total") or 0.0)
    except Exception:  # noqa: BLE001
        spent_this_month = 0.0

    proposals_today = 0
    try:
        proposals_today = db.proposals.count_documents(
            {"created_at": {"$gte": day_start}})
    except Exception:  # noqa: BLE001
        proposals_today = 0

    return {
        "monthly_budget_usd": monthly_budget_usd,
        "spent_this_month_usd": spent_this_month,
        "remaining_this_month_usd": max(0.0, monthly_budget_usd - spent_this_month),
        "max_proposals_per_day": max_proposals_per_day,
        "proposals_today": proposals_today,
        "can_propose_today": proposals_today < max_proposals_per_day,
    }


# ---- Prompt builders -----------------------------------------------------


_RESEARCHER_SYSTEM_PROMPT = """You are the MadScientist Researcher agent. Your job is to draft ONE experiment proposal for the rl-roboracer codebase based on:
  * recent papers from arxiv
  * what's already been tried (existing experiment_designs, reward_designs, top past models)
  * the project's research goals: improve a Unity-rendered autonomous-driving RL agent that mixes BC pretraining with online SAC

CURRENT DATE: {today_iso}

Note on arxiv IDs: the format is YYMM.NNNNN. So 2605.NNNNN means submitted in May 2026 - these are LEGITIMATE recent papers, not "future-dated" hallucinations. Don't dismiss recent papers solely on YYMM. Every arxiv_id you cite will be HTTP-probed against https://arxiv.org/abs/<arxiv_id> before your proposal is accepted - a 404 means the paper doesn't exist and you'll be asked to revise.

The proposal will be reviewed by a Judge agent against a rubric. Your output must:

  1. Pass the Judge's 7 pre-rubric checks:
     A. source_papers entries have syntactically-valid arxiv_ids
     B. hypothesis + success_criteria.primary are non-empty
     C. >=2 experiment_arms; exactly one named "base"
     D. wall-time estimate fits remaining budget (provided below)
     E. experiment_design_fields keys are in the SCHEMA (provided below)
     F. code_changes_summary does NOT mention safety-critical paths
        (_emergency_pause_handler, _get_job_lifecycle_state, RpcClient.__init__,
        dashboard/, etc.)
     G. if any arm overrides reward_design_fields, success_criteria.secondary
        MUST include at least one of: avg_goals_per_episode, avg_speed,
        avg_episode_length, avg_steering_angle_ratio

  2. Score reasonably across these axes (anchors abbreviated):
     - hypothesis_specificity: single inequality, measurable, falsifiable
     - novelty: differs from existing designs + 2-year-old literature
     - significance: outcome changes what we try next
     - statistical_power_and_baseline_rigor: n_seeds>=3, paired comparisons
     - goodhart_resistance: track reward-invariant secondaries
     - paper_faithfulness: implementation matches cited paper's method
     - implementation_feasibility: fits cleanly in SCHEMA, no SAC refactor
     - cost_and_reproducibility: <=10% of remaining budget; explicit seeds

  3. Produce a STRUCTURED primary_parsed criterion so the outcome ingester
     can evaluate mechanically (without parsing natural language).

CITING PAPERS (required for paper-backed proposals):
  Every entry in `source_papers` must include BOTH:
    1. `section_refs`: concrete locators inside the paper. A human
       reviewer must be able to flip to the right page from the locator.
       GOOD: ["Section 4.2", "Eq. 12", "Theorem 3", "Algorithm 1",
              "Fig. 5", "§3.1", "Page 7", "Appendix B"]
       BAD:  ["Langevin-type diffusion variance analysis",
              "Bellman recursion dependence",
              "the variance bound"]
       (BAD entries describe content but don't tell the reader WHERE.)
    2. `supporting_evidence`: 1-3 sentence excerpt or paraphrase of the
       cited passage explaining WHY this paper supports your specific
       hypothesis. Aim for >=40 chars.
       GOOD: "Eq. 12 bounds the policy-gradient variance as O(1/sqrt(B))
              in the batch size B, predicting >=2x noise reduction for
              our 256->1024 batch increase."
       BAD:  "supports our hypothesis"
       BAD:  "discusses batch sizes"
  Pre-rubric check H rejects any paper missing either field.

PROJECT SCOPE:
  * Prefer tuning EXISTING experiment_designs SCHEMA knobs - that's the
    fast path (orchestrator queues training immediately after approval).
  * If your hypothesis genuinely needs a NEW knob not in SCHEMA,
    declare it in `proposed_schema_extensions` (see schema below). The
    orchestrator will spawn a Cursor SDK code-writing agent that opens
    a PR adding the SCHEMA entry + trainer-kwarg plumbing. This is the
    slow path - PR review adds latency before training runs.
  * If you reference a reward_design, do it via reward_design_id (an
    existing design) - inline reward_design_fields are rejected by
    the orchestrator.
  * Curriculum / track-difficulty knobs (corner_radius,
    curvature_difficulty) are LIVE: the trainer forwards them to Unity's
    procedural TrackGenerator on every episode reset, so they genuinely
    change the simulated track. corner_radius is the primary lever
    (smaller = tighter turns); curvature_difficulty (0..1) adds chicanes.
    A curriculum is expressed as a sequence of arms/jobs with DECREASING
    corner_radius (and/or increasing curvature_difficulty) - one fixed
    difficulty per arm. NOTE: difficulty is constant WITHIN a single run
    (no in-run annealing yet), so design curricula as multi-arm or
    multi-job ladders, not as a ramp inside one job.
  * Cost model: ~0.5s/iter at training time; ~$1/hr compute-proxy.
    A 5-seed x 2-arm x 5000-iter run costs ~$1.50.

OUTPUT FORMAT:
Return EXACTLY one JSON object (no markdown fences, no commentary,
no leading text). It must match this schema:

{
  "title": "concise title",
  "hypothesis": "X causes Y measured by Z, with directional inequality + effect size",
  "motivation": "1-2 paragraphs: why this paper, why this codebase, why now",
  "code_changes_summary": "what changes go into experiment_design_fields (or proposed_schema_extensions if you need a new knob)",
  "source_papers": [
    {
      "arxiv_id": "...",
      "title": "...",
      "section_refs": ["Section 4.2", "Eq. 12"],
      "supporting_evidence": "1-3 sentence quote or paraphrase of the cited passage explaining WHY this paper supports your hypothesis."
    }
  ],
  "experiment_arms": [
    {"name": "base", "description": "...", "experiment_design_id": "experiment-default"},
    {"name": "exp1", "description": "...", "experiment_design_fields": {"<key>": <value>}}
  ],
  "n_seeds_per_arm": 3,
  "num_iterations_per_seed": 5000,
  "expected_wall_time_hours": 6.0,
  "success_criteria": {
    "primary": "free-form sentence with metric, comparison, effect size",
    "primary_parsed": {
      "metric": "avg_return",
      "arm_a": "exp1",
      "arm_b": "base",
      "comparator": ">=",
      "threshold": 0.10,
      "threshold_kind": "relative"
    },
    "secondary": ["...", "..."]
  },
  "proposed_schema_extensions": [
    {
      "name": "aux_bc_loss_weight",
      "type": "float",
      "default": 0.0,
      "min_value": 0.0,
      "max_value": 1.0,
      "doc": "DAPG-style aux BC loss weight in the SAC actor update.",
      "paper_ref": "1709.10089",
      "section": "_section_bc"
    }
  ]
}

Use the existing experiment_design_fields keys + types shown in the
SCHEMA wherever possible. If your hypothesis needs a NEW knob, declare
it in proposed_schema_extensions with a complete schema entry (name,
type, default, bounds, doc, paper_ref, section). Then reference that
new key in experiment_design_fields just like any existing key - the
orchestrator will accept it as long as it's listed in
proposed_schema_extensions.

If proposed_schema_extensions is non-empty, your code_changes_summary
SHOULD briefly mention the SCHEMA addition (e.g., "adds
aux_bc_loss_weight to experiment_designs.SCHEMA + plumbs through
main()'s kwargs"). Otherwise leave it empty or short.

Leave proposed_schema_extensions = [] if you can express the
hypothesis with existing knobs only - that's the strongly-preferred
path because it avoids the PR-review latency."""


_RESEARCHER_USER_TEMPLATE = """Draft one proposal based on the following context.

RECENT ARXIV PAPERS (subset by relevance, last {days_back}d):
{papers_block}

CODEBASE STATE:
```json
{codebase_json}
```

BUDGET CONTEXT:
{budget_block}

Return the JSON proposal object specified in the system prompt - nothing else."""


_RESEARCHER_REVISION_TEMPLATE = """Your previous proposal failed these pre-rubric checks:

{failures}

Please revise the proposal to address each failure and return the corrected JSON object. Same schema as before. No markdown fences, no commentary."""


def build_researcher_prompts(
    papers: List[Dict[str, Any]],
    codebase: Dict[str, Any],
    budget: Dict[str, Any],
    *,
    days_back: int = DEFAULT_RESEARCH_DAYS_BACK,
    today: Optional[datetime.date] = None,
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the first researcher call."""
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    today_iso = today.isoformat()
    papers_block = "\n\n".join(
        f"[{p['arxiv_id']}] {p['title']}\n"
        f"  authors: {', '.join((p.get('authors') or [])[:3])}"
        f"{', ...' if len(p.get('authors') or []) > 3 else ''}\n"
        f"  category: {p.get('primary_category', '?')}\n"
        f"  abstract: {(p.get('summary') or '')[:600]}"
        for p in papers
    ) if papers else "(no papers in this window)"

    budget_block = (
        f"  monthly_budget_usd: ${budget['monthly_budget_usd']:.2f}\n"
        f"  spent_this_month_usd: ${budget['spent_this_month_usd']:.2f}\n"
        f"  remaining_this_month_usd: ${budget['remaining_this_month_usd']:.2f}\n"
        f"  proposals_today: {budget['proposals_today']} / {budget['max_proposals_per_day']}"
    )

    user_prompt = _RESEARCHER_USER_TEMPLATE.format(
        days_back=days_back,
        papers_block=papers_block,
        codebase_json=json.dumps(codebase, indent=2, default=str),
        budget_block=budget_block,
    )
    # Plain .replace() rather than .format() because the system prompt
    # contains literal JSON examples with { / } characters that
    # str.format() would mis-parse as placeholders.
    system_prompt = _RESEARCHER_SYSTEM_PROMPT.replace(
        "{today_iso}", today_iso)
    return (system_prompt, user_prompt)


def build_revision_prompt(failures: List[str]) -> str:
    """Build the user-message that asks the LLM to revise after a failed
    pre-rubric check pass."""
    failure_lines = "\n".join(f"  * {f}" for f in failures)
    return _RESEARCHER_REVISION_TEMPLATE.format(failures=failure_lines)


# ---- Response parsing (shared with judge.py shape) -----------------------


# Canonical URL prefix for arxiv papers. The Researcher overwrites
# any LLM-emitted source_papers[*].url with this canonical form
# derived from arxiv_id.
_ARXIV_ABS_URL = "https://arxiv.org/abs/"


def _canonicalize_paper_urls(candidate: Dict[str, Any]) -> None:
    """In-place rewrite of every source_papers[*].url to the canonical
    https://arxiv.org/abs/<arxiv_id> form.

    Idempotent. No-op if source_papers is missing / not a list / the
    entry is missing arxiv_id. Tolerates string entries (rare LLM
    failure mode) by skipping them.
    """
    papers = candidate.get("source_papers")
    if not isinstance(papers, list):
        return
    for p in papers:
        if not isinstance(p, dict):
            continue
        aid = p.get("arxiv_id")
        if isinstance(aid, str) and aid.strip():
            p["url"] = _ARXIV_ABS_URL + aid.strip()


_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def parse_proposal_response(text: str) -> Dict[str, Any]:
    """Extract the JSON object from the LLM's response. Same logic as
    judge.parse_judge_response but kept separate so the two workers
    can evolve independently."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    for m in _JSON_BLOCK_RE.finditer(text):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(
        f"No parseable JSON in researcher response (first 200 chars: {text[:200]!r})")


# ---- Cost helpers --------------------------------------------------------


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000.0) * _RESEARCHER_INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000.0) * _RESEARCHER_OUTPUT_USD_PER_MTOK)


# ---- Anthropic call wrapper ----------------------------------------------


def call_anthropic_researcher(
    anthropic_client,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 3000,
):
    """Wrapper around anthropic.messages.create for the researcher.

    `messages` is the full multi-turn message list (initial + revision
    rounds). We keep system in `system` (not in messages) per Anthropic's
    API.
    """
    resp = anthropic_client.messages.create(
        model=RESEARCHER_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
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


# ---- Single-cycle orchestration ------------------------------------------


def _build_proposal_doc(
    parsed: Dict[str, Any],
    *,
    cycle_id,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    research_notes_used: List[Any],
    git_sha: Optional[str] = None,
    git_branch: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert the LLM's parsed JSON into a Mongo-ready proposal doc.

    Fills in the required `created_at`/`updated_at`/`status`/`cost`/
    `audit_events` fields the LLM doesn't produce.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        # ---- Metadata ---------------------------------------------------
        "title": str(parsed.get("title") or "")[:200],
        "status": constants.STATUS_PENDING_JUDGE,
        "created_at": now,
        "updated_at": now,
        "git_sha_at_proposal": git_sha,
        "git_branch_at_proposal": git_branch,

        # ---- Scientific content ----------------------------------------
        "hypothesis": str(parsed.get("hypothesis") or ""),
        "motivation": str(parsed.get("motivation") or ""),
        "code_changes_summary": str(parsed.get("code_changes_summary") or ""),
        "source_papers": list(parsed.get("source_papers") or []),
        "experiment_arms": list(parsed.get("experiment_arms") or []),
        "n_seeds_per_arm": int(parsed.get("n_seeds_per_arm") or 1),
        "num_iterations_per_seed": int(parsed.get("num_iterations_per_seed") or 5000),
        "expected_wall_time_hours": (
            float(parsed["expected_wall_time_hours"])
            if parsed.get("expected_wall_time_hours") is not None
            else None),
        "success_criteria": parsed.get("success_criteria") or {
            "primary": "", "primary_parsed": None, "secondary": []},
        # Phase 1C-Full: pass through the new-schema-fields proposal
        # for the orchestrator's dispatch + the Cursor SDK agent's
        # implementation prompt.
        "proposed_schema_extensions": list(
            parsed.get("proposed_schema_extensions") or []),

        # ---- Cycle provenance ------------------------------------------
        "research_note_ids": research_notes_used,
        # Cycle id helps trace which cycle produced which proposal -
        # useful for cost-per-cycle aggregations later.
        "research_cycle_id": cycle_id,

        # ---- Bookkeeping -----------------------------------------------
        "training_job_ids": [],
        "judge_review": None,
        "decision": None,
        "results": None,
        "implementation_log": [],
        "cost": {
            "madscientist_usd": cost_usd,
            "judge_usd": 0.0,
            "cursor_usd": 0.0,
            "other_usd": 0.0,
        },
        "audit_events": [
            {
                "at": now,
                "by_agent": constants.AGENT_RESEARCHER,
                "event": "drafted",
                "detail": {
                    "cycle_id": cycle_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                },
            },
        ],
    }


def _record_research_note(
    db, cycle_id, source_ref: str, text: str,
    *, source_type: str = "codebase_observation",
    tokens_used: int = 0, cost_usd: float = 0.0,
) -> Any:
    """Append a research_note. Returns the inserted _id for backref."""
    now = datetime.datetime.now(datetime.timezone.utc)
    doc = {
        "at": now,
        "cycle_id": cycle_id,
        "source_type": source_type,
        "source_ref": source_ref,
        "text": text[:8000],
        "embedding": None,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
    }
    return db[constants.COLL_RESEARCH_NOTES].insert_one(doc).inserted_id


def research_one_cycle(
    db,
    anthropic_client,
    *,
    monthly_budget_usd: float = 250.0,
    max_proposals_per_day: int = 1,
    max_queued_jobs: int = constants.DEFAULT_MAX_QUEUED_JOBS,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    research_query: str = DEFAULT_RESEARCH_QUERY,
    days_back: int = DEFAULT_RESEARCH_DAYS_BACK,
    max_papers: int = DEFAULT_RESEARCH_MAX_PAPERS,
    arxiv_fetcher=None,
) -> Optional[Dict[str, Any]]:
    """Run one full research cycle. Returns the inserted proposal doc
    or None if the cycle aborted (budget gate, rate gate, parse
    failure, validation failure).

    arxiv_fetcher: optional callable(query, days_back, max_papers) ->
    list[paper_dict]. None uses fetch_recent_arxiv_papers. Tests pass
    a mock here.
    """
    cycle_id = ObjectId()
    now = datetime.datetime.now(datetime.timezone.utc)

    # ---- Step 1: budget + rate-limit gates ----------------------------
    budget = fetch_budget_status(
        db,
        monthly_budget_usd=monthly_budget_usd,
        max_proposals_per_day=max_proposals_per_day,
    )
    if not budget["can_propose_today"]:
        print(
            f"researcher: cycle skipped - proposals_today="
            f"{budget['proposals_today']} >= max_proposals_per_day="
            f"{max_proposals_per_day}.",
            flush=True)
        return None
    if budget["remaining_this_month_usd"] <= 0.50:
        print(
            f"researcher: cycle skipped - remaining_this_month_usd="
            f"${budget['remaining_this_month_usd']:.2f} too low to safely "
            f"start a cycle. Raise BUDGET_USD_PER_MONTH or wait for the "
            f"next month rollover.",
            flush=True)
        return None

    # ---- Step 1b: unapproved-proposal depth gate --------------------------
    # Don't generate another proposal while there are already unapproved
    # ones waiting. "Unapproved" means pending_judge, pending_user, or
    # deferred - proposals that have not yet been approved (or rejected)
    # by the judge + operator. This prevents the researcher from piling
    # up more experiments than the operator can review.
    if max_queued_jobs > 0:
        _unapproved_statuses = [
            constants.STATUS_PENDING_JUDGE,
            constants.STATUS_PENDING_USER,
            constants.STATUS_DEFERRED,
        ]
        try:
            queued = db[constants.COLL_PROPOSALS].count_documents(
                {"status": {"$in": _unapproved_statuses}})
        except Exception as _e:  # noqa: BLE001
            print(
                f"researcher: unapproved-proposal depth check failed "
                f"(non-fatal): {_e}. Continuing cycle.",
                flush=True)
            queued = 0
        if queued >= max_queued_jobs:
            print(
                f"researcher: cycle skipped - {queued} unapproved proposal(s) "
                f"already in queue (pending_judge / pending_user / deferred) "
                f">= max_queued_jobs={max_queued_jobs}. "
                f"Review or reject existing proposals before new ones are added.",
                flush=True)
            return None

    # ---- Step 2: fetch arxiv papers -----------------------------------
    try:
        if arxiv_fetcher is None:
            papers = fetch_recent_arxiv_papers(
                research_query,
                days_back=days_back,
                max_papers=max_papers)
        else:
            papers = arxiv_fetcher(research_query, days_back, max_papers)
    except Exception as e:  # noqa: BLE001
        print(
            f"researcher: arxiv fetch failed: {type(e).__name__}: {e}. "
            f"Continuing with no papers (proposal will be codebase-only).",
            flush=True)
        papers = []

    # ---- Step 3: codebase context -------------------------------------
    codebase = fetch_codebase_context(db)

    # ---- Step 4: build prompts + record research notes ----------------
    system_prompt, user_prompt = build_researcher_prompts(
        papers, codebase, budget, days_back=days_back)

    # Stamp a note per cited paper so the proposal can backref them.
    research_note_ids: List[Any] = []
    for p in papers[:5]:
        note_id = _record_research_note(
            db, cycle_id,
            source_ref=f"arxiv:{p['arxiv_id']}",
            text=f"{p['title']}\n{(p.get('summary') or '')[:2000]}",
            source_type="arxiv_abstract")
        research_note_ids.append(note_id)
    # Plus one for the codebase snapshot.
    _record_research_note(
        db, cycle_id,
        source_ref="codebase://experiment_designs+models",
        text=json.dumps(codebase, default=str)[:6000],
        source_type="codebase_observation")

    # ---- Step 5: LLM call (with self-critique retries) ---------------
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_prompt}]
    total_input_tokens = 0
    total_output_tokens = 0
    parsed: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    last_failures: List[str] = []

    for attempt in range(max_revisions + 1):
        try:
            resp = call_anthropic_researcher(
                anthropic_client, system_prompt, messages)
        except Exception as e:  # noqa: BLE001
            last_error = f"anthropic call failed: {type(e).__name__}: {e}"
            print(f"researcher: {last_error}", flush=True)
            break

        total_input_tokens += resp["input_tokens"]
        total_output_tokens += resp["output_tokens"]

        try:
            candidate = parse_proposal_response(resp["text"])
        except ValueError as e:
            last_error = f"parse failed: {e}"
            print(
                f"researcher: attempt {attempt + 1}: {last_error}",
                flush=True)
            messages.append({"role": "assistant", "content": resp["text"]})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as JSON. "
                    "Return ONLY the JSON proposal object - no markdown "
                    "fences, no commentary.")})
            continue

        # Canonicalize source_papers[*].url BEFORE running checks. We
        # don't trust the LLM to produce well-formed URLs (it routinely
        # malforms them - missing protocol, wrong host, etc.), so we
        # derive the canonical URL from arxiv_id directly. This also
        # ensures the dashboard's "view paper" links work.
        _canonicalize_paper_urls(candidate)

        # Pre-rubric self-check. probe_urls=True so we HTTP-HEAD every
        # canonical arxiv URL before submission - catches the
        # LLM-hallucinated-arxiv-id failure mode (the LLM can substitute
        # a plausible-looking arxiv_id that doesn't actually resolve,
        # even when given a list of real papers in context). Costs
        # ~100ms per paper, plenty cheap.
        check_result = pre_rubric_checks.run_all(
            candidate,
            monthly_budget_usd=monthly_budget_usd,
            spent_so_far_usd=budget["spent_this_month_usd"],
            probe_urls=True,
        )
        if check_result.all_passed:
            parsed = candidate
            print(
                f"researcher: attempt {attempt + 1}: proposal passes "
                f"all pre-rubric checks.",
                flush=True)
            break

        # Failed checks. If we still have retries left, send the
        # failure list back to the LLM. Otherwise abandon.
        last_failures = [
            f"check {r.check_id}: {r.reason}"
            for r in check_result.failed]
        last_error = "; ".join(last_failures)
        if attempt >= max_revisions:
            print(
                f"researcher: attempt {attempt + 1}: still failing checks "
                f"after {max_revisions} revisions. Abandoning cycle.",
                flush=True)
            break
        print(
            f"researcher: attempt {attempt + 1}: pre-rubric failures "
            f"{[r.check_id for r in check_result.failed]}; asking LLM to revise.",
            flush=True)
        messages.append({"role": "assistant", "content": resp["text"]})
        messages.append({
            "role": "user",
            "content": build_revision_prompt(last_failures)})

    cost_usd = _estimate_cost_usd(total_input_tokens, total_output_tokens)

    if parsed is None:
        print(
            f"researcher: cycle aborted after {total_input_tokens + total_output_tokens} "
            f"tokens (cost ${cost_usd:.4f}). Last error: {last_error}",
            flush=True)
        return None

    # ---- Step 6: submit proposal --------------------------------------
    proposal_doc = _build_proposal_doc(
        parsed,
        cycle_id=cycle_id,
        cost_usd=cost_usd,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        research_notes_used=research_note_ids,
    )
    inserted = db[constants.COLL_PROPOSALS].insert_one(proposal_doc)
    proposal_doc["_id"] = inserted.inserted_id

    print(
        f"researcher: drafted proposal {inserted.inserted_id} - "
        f"title={proposal_doc['title']!r}, cost=${cost_usd:.4f}, "
        f"tokens={total_input_tokens + total_output_tokens}.",
        flush=True)
    return proposal_doc


# ---- Worker loop ---------------------------------------------------------


def research_loop(
    db,
    anthropic_client,
    *,
    poll_interval_seconds: int = constants.DEFAULT_RESEARCH_CYCLE_INTERVAL_SECONDS,
    monthly_budget_usd: float = 250.0,
    max_proposals_per_day: int = 1,
    max_queued_jobs: int = constants.DEFAULT_MAX_QUEUED_JOBS,
    research_query: str = DEFAULT_RESEARCH_QUERY,
    should_stop_fn=None,
):
    """Periodic cycle runner. One proposal at most per cycle; gated
    by the daily + monthly caps + job-queue depth in research_one_cycle."""
    print(
        f"researcher: starting loop, cycle interval = "
        f"{poll_interval_seconds}s, max_proposals_per_day = "
        f"{max_proposals_per_day}, monthly_budget = ${monthly_budget_usd:.2f}, "
        f"max_queued_jobs = {max_queued_jobs}",
        flush=True)
    while True:
        if should_stop_fn is not None and should_stop_fn():
            print("researcher: stop signal received; exiting.", flush=True)
            return
        try:
            research_one_cycle(
                db, anthropic_client,
                monthly_budget_usd=monthly_budget_usd,
                max_proposals_per_day=max_proposals_per_day,
                max_queued_jobs=max_queued_jobs,
                research_query=research_query)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(
                f"researcher: uncaught cycle error: "
                f"{type(e).__name__}: {e}\n{tb}",
                flush=True)
        # Cooperative sleep: the research cycle interval is typically
        # hours, but we want SIGTERM/SIGINT responsiveness in seconds.
        # Sleep in 5s chunks + check the stop flag between them.
        slept = 0
        while slept < poll_interval_seconds:
            if should_stop_fn is not None and should_stop_fn():
                print(
                    "researcher: stop signal received during cycle "
                    "sleep; exiting.",
                    flush=True)
                return
            chunk = min(5, poll_interval_seconds - slept)
            time.sleep(chunk)
            slept += chunk
