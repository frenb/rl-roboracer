"""Pydantic schemas for the MadScientist Mongo collections.

These are the source of truth for what shapes the workers write + the
dashboard reads. We use Pydantic v2 (not bare dicts) so:

  * Field names + types are documented in one place.
  * Workers fail loudly on malformed documents rather than silently
    propagating bad data through the lifecycle.
  * The dashboard server can borrow these schemas later (Pydantic ->
    JSON Schema export) to validate the chat-reply parser's output.

Three collections defined here, mirroring constants.COLL_*:

  * Proposal              -> db.proposals
  * ResearchNote          -> db.research_notes
  * JudgeRubricVersion    -> db.judge_rubric_history

Plus a handful of nested types used as fields of the above.

All `*_at` timestamps are stored as timezone-aware UTC datetimes so
pymongo round-trips them as BSON Dates and the dashboard's `new Date(...)`
parses them as ISO-8601 with a Z suffix.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover - pydantic only available in container
    # Fall-through stub so the module is importable in environments that
    # don't have pydantic installed (e.g., the bare host running the
    # trainer's tests). The runtime container HAS pydantic via the
    # madscientist requirements.txt.
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self, *args, **kwargs):
            return self.__dict__

    class ConfigDict(dict):  # type: ignore
        pass

    def Field(default=None, **kwargs):  # type: ignore  # noqa: N802
        return default


from . import constants


# ---- Nested types --------------------------------------------------------


class PaperReference(BaseModel):
    """A cited paper attached to a proposal or research note.

    The Researcher is required (via pre-rubric check H) to populate BOTH
    `section_refs` with concrete locators (e.g., 'Section 4.2', 'Eq. 12',
    'Theorem 3', 'Algorithm 1', 'Fig. 5', '§3.1', 'Page 7') AND
    `supporting_evidence` with a 1-3 sentence excerpt or paraphrase of
    the specific passage that supports this proposal's hypothesis. This
    lets the human reviewer jump straight to the relevant page in the
    paper instead of skim-reading the abstract.
    """
    arxiv_id: str = Field(..., description="e.g., '1709.10089' or '2104.06129v2'")
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    section_refs: List[str] = Field(
        default_factory=list,
        description="Concrete locators within the paper. e.g., "
                    "'Section 4.2', 'Eq. 12', 'Theorem 3', 'Algorithm 1', "
                    "'Fig. 5', '§3.1', 'Page 7'. Pre-rubric check H "
                    "requires at least one entry containing a digit or "
                    "an explicit anchor word.")
    supporting_evidence: str = Field(
        default="",
        description="1-3 sentence excerpt or paraphrase of the cited "
                    "passage explaining WHY this paper supports the "
                    "current proposal's hypothesis. Required (>=40 "
                    "chars) by pre-rubric check H.")
    url: Optional[str] = None


class ExperimentArm(BaseModel):
    """One arm of an experimental comparison.

    The `fields` dict overlays onto a base experiment_design / reward_design
    referenced by `experiment_design_id` / `reward_design_id` - mirrors how
    apply_to_main_kwargs() works in rl_agent/experiment_designs.py.
    """
    name: str = Field(..., description="Slot label - 'base', 'exp1', 'exp2', etc.")
    description: str = ""
    # Either reference an existing design + delta, or specify all fields inline.
    experiment_design_id: Optional[str] = None
    experiment_design_fields: Dict[str, Any] = Field(default_factory=dict)
    reward_design_id: Optional[str] = None
    reward_design_fields: Dict[str, Any] = Field(default_factory=dict)


class ProposedSchemaExtension(BaseModel):
    """A NEW experiment_designs SCHEMA field this proposal needs added.

    Researcher MVP-1B can only propose experiments using existing knobs
    (pre-rubric check E rejects unknown keys). Phase 1C-Full relaxes
    that: proposals can declare new fields here, the Judge accepts them
    based on this declaration, and the Cursor SDK orchestrator
    implements the SCHEMA addition + trainer-kwarg plumbing before
    queueing TRAIN jobs.

    Example for a DAPG-style aux BC loss experiment:
        ProposedSchemaExtension(
            name="aux_bc_loss_weight",
            type="float",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            doc="DAPG-style aux BC loss weight in the SAC actor update.",
            paper_ref="1709.10089",
            section="_section_bc",
        )

    Field shape mirrors experiment_designs.SCHEMA entries 1:1 so the
    Cursor agent just needs to insert this as a new dict entry.
    """
    name: str = Field(
        ...,
        description="The schema key (snake_case, no dots).")
    type: str = Field(
        ...,
        description="One of 'int' | 'float' | 'bool' | 'enum' | 'list[int]'.")
    default: Any = Field(
        ...,
        description="Trainer's built-in default if this knob isn't overridden.")
    min_value: Optional[float] = Field(
        default=None,
        description="Soft lower bound; None = no bound. Mapped to SCHEMA's 'min'.")
    max_value: Optional[float] = Field(
        default=None,
        description="Soft upper bound; None = no bound. Mapped to SCHEMA's 'max'.")
    doc: str = Field(
        ...,
        description="One-line human description shown as the dashboard tooltip.")
    paper_ref: Optional[str] = Field(
        default=None,
        description="arxiv id for the paper this knob ties to.")
    section: str = Field(
        default="_section_rl_loop",
        description="Which SCHEMA section to insert under "
                    "(_section_rl_loop / _section_bc / _section_replay / "
                    "_section_optimizer / _section_network).")


class PrimaryCriterionParsed(BaseModel):
    """Machine-readable form of the proposal's primary success criterion.

    The Researcher writes both the human-readable `SuccessCriteria.primary`
    (a free-form sentence the Judge can sanity-check) AND this structured
    counterpart that the outcome ingester evaluates mechanically without
    parsing natural language.

    Example: "avg_return(exp2) - avg_return(base) >= 10%" becomes
        {
          "metric": "avg_return",
          "arm_a": "exp2",
          "arm_b": "base",
          "comparator": ">=",
          "threshold": 0.10,
          "threshold_kind": "relative"
        }

    When primary_parsed is None, the ingester still computes per-arm
    summary stats but leaves OutcomeResult.primary_criterion_met as None
    so the operator's manual review on the dashboard isn't preempted.
    """
    metric: str = Field(
        ...,
        description="One of: 'avg_return', 'avg_goals_per_episode', "
                    "'avg_speed', 'avg_episode_length', "
                    "'avg_steering_angle_ratio'")
    arm_a: str = Field(
        ...,
        description="The variant arm whose mean we compare against the baseline")
    arm_b: str = Field(
        ...,
        description="The baseline arm (typically 'base')")
    comparator: str = Field(
        ...,
        description="One of '>=', '<=', '>', '<' for the delta direction")
    threshold: float = Field(
        ...,
        description="Numeric threshold the delta must hit. Interpretation "
                    "depends on threshold_kind.")
    threshold_kind: str = Field(
        default="relative",
        description="'relative' (threshold is fraction of arm_b's mean) "
                    "or 'absolute' (threshold is in metric's raw units)")


class SuccessCriteria(BaseModel):
    """How we'll judge whether the hypothesis was supported.

    `primary` is the single human-readable statement the Judge reviews.
    `primary_parsed` (optional) is the structured form the outcome
    ingester evaluates mechanically. `secondary` lists supporting checks.
    """
    primary: str = Field(
        ...,
        description="A measurable statement, e.g. 'avg_return(exp2) - avg_return(base) >= 10%'")
    primary_parsed: Optional[PrimaryCriterionParsed] = Field(
        default=None,
        description="Structured counterpart to `primary` for mechanical "
                    "outcome evaluation. None = ingester computes summary "
                    "stats only; operator decides manually.")
    secondary: List[str] = Field(default_factory=list)


class JudgeReview(BaseModel):
    """Output of the Judge worker. Populated when Judge transitions a
    proposal from pending_judge to pending_user.
    """
    overall: Literal[
        "strong_accept", "accept", "weak_accept", "weak_reject", "reject"
    ]
    scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Axis name -> 0-5 score per the rubric. Keys defined by the "
                    "active rubric version; see judge_rubric_history.")
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    suggested_revisions: List[str] = Field(default_factory=list)
    judged_at: datetime.datetime
    rubric_version: int = Field(..., description="Which judge_rubric_history.version was applied")
    judge_model: str = Field(..., description="Model identifier - e.g., 'claude-opus-4-7'")
    judge_token_count: Optional[int] = None
    judge_cost_usd: Optional[float] = None


class UserDecision(BaseModel):
    """The user's reply (parsed from email or dashboard click)."""
    at: datetime.datetime
    by: str = Field(default=constants.AGENT_USER)
    action: Literal["approve", "approve_with_revisions", "reject", "defer"]
    note: str = ""
    revision_applied: bool = Field(
        default=False,
        description="True if action=approve_with_revisions AND the orchestrator "
                    "actually folded the Judge's suggested_revisions into the spec.")
    source: Literal["email", "dashboard", "api", "manual"] = "email"


class CostBreakdown(BaseModel):
    """Total API spend attributed to this proposal across its lifecycle.

    Tracked separately so the dashboard's budget gauge can attribute
    spend to the right worker.
    """
    madscientist_usd: float = 0.0
    judge_usd: float = 0.0
    cursor_usd: float = 0.0
    other_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return (self.madscientist_usd + self.judge_usd
                + self.cursor_usd + self.other_usd)


class PerArmResult(BaseModel):
    """Per-arm summary statistic in the outcome record."""
    name: str
    n_trials: int = 0
    mean: Optional[float] = None
    ci_low_95: Optional[float] = None
    ci_high_95: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None


class OutcomeResult(BaseModel):
    """Populated by outcome_ingester when all TRAIN jobs for the proposal
    reach status DONE.
    """
    primary_criterion_met: Optional[bool] = None
    primary_delta: Optional[float] = None
    primary_p_value: Optional[float] = None
    per_arm: List[PerArmResult] = Field(default_factory=list)
    verdict: Literal["supported", "rejected", "inconclusive"] = "inconclusive"
    notes: str = ""
    computed_at: datetime.datetime
    n_jobs_succeeded: int = 0
    n_jobs_failed: int = 0


class AuditEvent(BaseModel):
    """One row in proposal.audit_events. Appended by every state transition.

    `by_agent` should be one of constants.AGENT_* identifiers. `detail`
    is freeform JSON-serializable structure (mostly used for logging
    error messages + diffs).
    """
    at: datetime.datetime
    by_agent: str
    event: str = Field(..., description="Short verb - 'drafted', 'judged', 'approved', etc.")
    detail: Dict[str, Any] = Field(default_factory=dict)


# ---- Top-level documents -------------------------------------------------


class Proposal(BaseModel):
    """One row in db.proposals.

    Long lifecycle, many states. Fields are listed in roughly the order
    they get populated (metadata -> scientific content -> judge -> user
    decision -> implementation -> training -> outcome).
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # ---- Metadata --------------------------------------------------------
    title: str
    status: Literal[
        "pending_judge", "pending_user", "approved", "rejected", "deferred",
        "implementing", "pr_open", "training", "done", "failed", "cancelled"
    ] = "pending_judge"
    created_at: datetime.datetime
    updated_at: datetime.datetime

    # ---- Provenance ------------------------------------------------------
    # git_sha + branch the MadScientist worker saw at draft time. The
    # implementation may bump main between proposal and PR; this stamp
    # makes the proposal reproducible.
    git_sha_at_proposal: Optional[str] = None
    git_branch_at_proposal: Optional[str] = None
    # Cited papers + freeform research-note backrefs.
    source_papers: List[PaperReference] = Field(default_factory=list)
    research_note_ids: List[Any] = Field(
        default_factory=list,
        description="List of research_notes._id values used to build this proposal")

    # ---- Scientific content ----------------------------------------------
    hypothesis: str
    motivation: str = ""
    code_changes_summary: str = ""
    experiment_arms: List[ExperimentArm] = Field(default_factory=list)
    n_seeds_per_arm: int = 1
    num_iterations_per_seed: int = 5000
    expected_wall_time_hours: Optional[float] = None
    success_criteria: SuccessCriteria

    # Phase 1C-Full: NEW SCHEMA fields this proposal needs added to
    # rl_agent/experiment_designs.SCHEMA before training jobs can run.
    # Empty list = the proposal only references existing knobs (the
    # common case; Phase 1C-MVP path auto-queues directly). Non-empty =
    # the orchestrator routes to the Cursor SDK agent which edits the
    # codebase and opens a PR before queueing.
    proposed_schema_extensions: List[ProposedSchemaExtension] = Field(
        default_factory=list)

    # ---- Judge -----------------------------------------------------------
    judge_review: Optional[JudgeReview] = None

    # ---- User decision ---------------------------------------------------
    decision: Optional[UserDecision] = None

    # ---- Implementation tracking -----------------------------------------
    implementation_started_at: Optional[datetime.datetime] = None
    implementation_branch: Optional[str] = None
    implementation_pr_url: Optional[str] = None
    implementation_log: List[str] = Field(
        default_factory=list,
        description="Streamed activity chunks from the Cursor SDK agent. "
                    "Appended in real time so the dashboard tab can tail it.")
    implementation_finished_at: Optional[datetime.datetime] = None
    implementation_failure_reason: Optional[str] = None

    # ---- Training jobs ---------------------------------------------------
    training_job_ids: List[Any] = Field(
        default_factory=list,
        description="ObjectIds of TRAIN jobs queued for this proposal")

    # ---- Outcome ---------------------------------------------------------
    results: Optional[OutcomeResult] = None

    # ---- Cost ------------------------------------------------------------
    cost: CostBreakdown = Field(default_factory=CostBreakdown)

    # ---- Audit log -------------------------------------------------------
    audit_events: List[AuditEvent] = Field(default_factory=list)


class ResearchNote(BaseModel):
    """One row in db.research_notes.

    Append-only. The MadScientist worker writes these as it reads papers
    + inspects the codebase + observes past results. Acts as both the
    agent's working memory AND a potential future vector-store target
    for similarity-based retrieval ("what have we already seen about
    DAPG-style aux losses?").
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    at: datetime.datetime
    cycle_id: Any = Field(..., description="ObjectId grouping notes from same MadScientist cycle")
    source_type: Literal[
        "arxiv_abstract", "arxiv_full", "web_page",
        "codebase_observation", "results_observation", "user_message"
    ]
    source_ref: str = Field(
        ...,
        description="Stable identifier for the source - 'arxiv:1709.10089', "
                    "'https://...', 'rl_agent/robotaxi.py:1462', 'proposal:<oid>'")
    text: str
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional vector embedding for similarity search. None until "
                    "we wire up a vector store; the field is reserved here so "
                    "future embedders can backfill without a schema migration.")
    tokens_used: int = 0
    cost_usd: float = 0.0


class JudgeRubricVersion(BaseModel):
    """One row in db.judge_rubric_history.

    A versioned snapshot of the rubric the Judge worker applies. New
    versions are appended (not edits) so historical Judge reviews remain
    interpretable when the rubric evolves.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    version: int = Field(..., description="Monotonic integer; 0 = stub placeholder")
    effective_from: datetime.datetime
    rubric_markdown: str = Field(
        ...,
        description="Full text of madscientist/JUDGE_RUBRIC.md at this version")
    rubric_axes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed axes for structured scoring. Each entry has at "
                    "least {name, description, scale_min, scale_max}")
    authored_by: str = Field(
        ...,
        description="'deep_research_initial' for v1, then 'self_revising_v<N>' once Phase 3")
    git_sha: Optional[str] = None
