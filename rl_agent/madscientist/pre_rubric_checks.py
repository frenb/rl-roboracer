"""Pre-rubric deterministic checks from JUDGE_RUBRIC.md section 2.

Eight checks (A-H) that run BEFORE the LLM-scoring step. Each is
deterministic and pure-Python (with one optional network probe in
check A). The Judge worker calls run_all(proposal, ...) on every
proposal before forking out to the rubric LLM call; a single failure
short-circuits the proposal to status=rejected with the failed check
recorded in proposal.judge_review.concerns.

These are the cheap fail-fast checks - they catch the largest classes
of garbage proposals (hallucinated papers, missing controls, schema-
key typos, over-budget plans, edits to safety-critical paths, vague
paper citations) without spending a single LLM token.

Each check returns a CheckResult(passed, reason). run_all() aggregates
into an AllChecksResult with .all_passed and .failed convenience
properties.

Import-side-effects: this module imports rl_agent.experiment_designs
to know which keys are valid in experiment_design_fields. That module
is pure-stdlib (no tf-agents), safe to import in the lightweight
madscientist container.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


# Guard the import so this module is still importable on a host
# without the trainer code (e.g., running unit tests on a laptop
# without the bind mount). Inside the madscientist container the
# import succeeds.
try:
    from rl_agent.experiment_designs import SCHEMA as _EXPERIMENT_DESIGNS_SCHEMA
except ImportError:
    _EXPERIMENT_DESIGNS_SCHEMA = {}


# ---- Constants -----------------------------------------------------------

# The three pluggable reward-function names (mirrors
# rl_agent/reward_designs.py). A proposal's
# experiment_arms[*].reward_design_fields keys MUST be a subset.
REWARD_FUNCTION_NAMES = ("reward_standard", "reward_success", "reward_failure")


# Reward-invariant secondary metrics the rubric requires when a
# reward-design proposal is on the table. Mirrors
# rl_agent/robotaxi.py's per-trial array names + the env's
# get_course_raw_counters keys.
REWARD_INVARIANT_METRICS = (
    "avg_goals_per_episode",
    "avg_speed",
    "avg_episode_length",
    "avg_steering_angle_ratio",
)


# Code-path substrings that a MadScientist proposal must NEVER mention
# editing in its code_changes_summary. These are the trainer's safety
# net + the dashboard (which is human-maintained). Pre-rubric check F
# rejects any proposal whose summary mentions any of these.
SAFETY_CRITICAL_PATTERNS = (
    "_emergency_pause_handler",
    "_emergency_state",
    "_get_job_lifecycle_state",
    "_restore_paused_active_dir",
    "_detect_resume_for_train_job",
    "_has_learner_checkpoint_for_job",
    "move_all_jobs_data",
    "RpcClient.__init__",
    "rl_agent/api.py",
    "virtual_endpoint/virtual.py",
    "dashboard/",
)


# Check H: concrete-locator regex for section_refs. Each cited paper's
# section_refs entries must contain at least one of these tokens (or
# a bare digit) to count as "concrete" - i.e., locator info a human
# reviewer can use to flip directly to the relevant passage.
# Examples that PASS:  "Section 4.2", "Eq. 12", "Theorem 3",
#                      "Algorithm 1", "Fig. 5", "§3.1", "Page 7",
#                      "Appendix B", "Table 2", "Lemma 1.3"
# Examples that FAIL:  "Langevin-type diffusion variance analysis"
#                      "Bellman recursion dependence"
#                      "the variance bound"
_CONCRETE_LOCATOR_RE = re.compile(
    r"(\d+(\.\d+)?)"                # any digit (covers '4', '4.2', '12')
    r"|\b("
    r"section|sec\.?|chapter|chap\.?|appendix|app\.?"
    r"|eq(uation)?\.?|theorem|thm\.?|lemma|proposition|prop\.?|corollary"
    r"|algorithm|algo\.?|figure|fig\.?|table|tab\.?"
    r"|page|p\.|pg\.?|§"
    r")\b",
    re.IGNORECASE,
)

# Check H: supporting_evidence minimum char length. Short enough that
# a 1-sentence paraphrase fits ("Eq. 12 bounds variance as O(1/sqrt(B)).")
# Long enough that one-word entries ("yes", "true") get rejected.
_MIN_SUPPORTING_EVIDENCE_CHARS = 40


# Default cost estimator constants. Tune from observed data later.
# Conservative: assume each training iter is 0.5s real wall time and
# costs ~$1/hr in API spend + compute proxy. With num_envs > 1 wall
# time is lower per iter; we don't try to model that here since this
# is a back-of-envelope budget check, not a billing system.
_SECONDS_PER_ITERATION = 0.5
_USD_PER_TRAINING_HOUR = 1.0


# ---- Result types --------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single check. `reason` populated only on failure."""
    check_id: str
    passed: bool
    reason: Optional[str] = None


@dataclass
class AllChecksResult:
    """Aggregator across all 7 checks."""
    all_passed: bool
    results: List[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        if self.all_passed:
            return "All 7 pre-rubric checks passed."
        ids = ", ".join(r.check_id for r in self.failed)
        return f"FAILED pre-rubric checks: {ids}"


# ---- Helper: tolerant attribute access -----------------------------------
#
# Proposals may arrive as either Pydantic Proposal instances OR raw
# dicts (e.g., fresh out of Mongo). The checks below should work
# against either. _get walks both.


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---- Individual checks ---------------------------------------------------


def check_a_arxiv_resolution(proposal, *, probe_urls: bool = False) -> CheckResult:
    """A: every entry in source_papers has a syntactically-valid arxiv_id;
    optionally probe each URL to confirm it returns 200.

    probe_urls=False (default) does only the syntactic check - safe for
    unit tests + CI. The Judge worker in Phase 1 sets probe_urls=True
    so hallucinated papers (arxiv ids that look real but don't exist)
    are also caught.

    Returns PASS when:
      * proposal has no source_papers (pure-codebase proposal), OR
      * every source paper has a syntactically valid arxiv_id, AND
      * if probe_urls=True, every probe returns HTTP 200.
    """
    papers = _get(proposal, "source_papers") or []
    if not papers:
        return CheckResult("A", True, None)

    # Arxiv ids come in three flavors:
    #   * New: "2104.06129" or "2104.06129v2"
    #   * Old: "cs.LG/0405123"
    #   * v3+ suffix: "1709.10089v3"
    # Pattern accepts all three with at most one v\d+ suffix.
    arxiv_re = re.compile(r"^[\w.\-]+/?[\w.\-]+(v\d+)?$")

    for p in papers:
        arxiv_id = _get(p, "arxiv_id")
        if not arxiv_id:
            return CheckResult("A", False, "paper entry missing arxiv_id")
        if not arxiv_re.match(str(arxiv_id)):
            return CheckResult("A", False, f"malformed arxiv_id: {arxiv_id!r}")

        if probe_urls:
            # Lazy import so callers that don't probe can avoid the
            # httpx dependency at import time.
            try:
                import httpx
            except ImportError:
                return CheckResult("A", False,
                    "probe_urls=True requested but httpx unavailable")
            url = f"https://arxiv.org/abs/{arxiv_id}"
            try:
                r = httpx.head(url, follow_redirects=True, timeout=10.0)
            except Exception as e:  # noqa: BLE001
                return CheckResult("A", False,
                    f"arxiv probe for {arxiv_id} failed: {e}")
            if r.status_code != 200:
                return CheckResult("A", False,
                    f"arxiv probe for {arxiv_id}: HTTP {r.status_code}")

    return CheckResult("A", True, None)


def check_b_hypothesis_and_criterion(proposal) -> CheckResult:
    """B: proposal.hypothesis and proposal.success_criteria.primary
    are both non-empty strings.

    Failures here usually mean a malformed proposal from the researcher
    worker (Phase 1 will validate before write, but defense-in-depth).
    """
    hyp = _get(proposal, "hypothesis", "")
    if not isinstance(hyp, str) or not hyp.strip():
        return CheckResult("B", False, "proposal.hypothesis is empty")

    sc = _get(proposal, "success_criteria")
    primary = _get(sc, "primary", "") if sc is not None else ""
    if not isinstance(primary, str) or not primary.strip():
        return CheckResult("B", False, "proposal.success_criteria.primary is empty")

    return CheckResult("B", True, None)


def check_c_experiment_arms(proposal) -> CheckResult:
    """C: at least 2 experiment_arms AND exactly one with name='base'.

    No control = no comparison; the outcome ingester has nothing to
    diff against. Multiple bases would be ambiguous (which one is the
    primary criterion against?).
    """
    arms = _get(proposal, "experiment_arms") or []
    if len(arms) < 2:
        return CheckResult("C", False,
            f"need >=2 experiment_arms, got {len(arms)}")
    base_count = sum(
        1 for a in arms
        if (_get(a, "name", "") or "").lower() == "base")
    if base_count != 1:
        return CheckResult("C", False,
            f"need exactly 1 arm named 'base', got {base_count}")
    return CheckResult("C", True, None)


def check_d_budget(
    proposal,
    *,
    monthly_budget_usd: float = 250.0,
    spent_so_far_usd: float = 0.0,
) -> CheckResult:
    """D: estimated wall-time cost fits the remaining monthly budget.

    Cost model is rough: n_arms x n_seeds x num_iterations_per_seed x
    0.5s per iter x $1/hr. This is a sanity check, not billing-grade.
    The real spend gets tracked in proposal.cost as the workers run.

    Caller passes the budget cap (from env BUDGET_USD_PER_MONTH) and
    the running month-to-date spend (sum of proposal.cost.total_usd
    across this month's proposals). Defaults are conservative.
    """
    arms = _get(proposal, "experiment_arms") or []
    n_arms = max(1, len(arms))
    n_seeds = max(1, int(_get(proposal, "n_seeds_per_arm", 1) or 1))
    n_iters = max(1, int(_get(proposal, "num_iterations_per_seed", 1) or 1))

    total_iter_seconds = n_arms * n_seeds * n_iters * _SECONDS_PER_ITERATION
    cost_estimate_usd = (total_iter_seconds / 3600.0) * _USD_PER_TRAINING_HOUR
    remaining = monthly_budget_usd - spent_so_far_usd

    if cost_estimate_usd > remaining:
        return CheckResult("D", False,
            f"estimated cost ${cost_estimate_usd:.2f} > remaining "
            f"${remaining:.2f} of monthly budget ${monthly_budget_usd:.2f}")
    return CheckResult("D", True, None)


def check_e_schema_keys(
    proposal,
    *,
    schema_keys: Optional[List[str]] = None,
) -> CheckResult:
    """E: every key in experiment_design_fields / reward_design_fields
    across all arms maps to a known field.

    Phase 1C-Full extension: keys declared in
    proposal.proposed_schema_extensions are ALSO accepted, since the
    Cursor SDK orchestrator will add them to SCHEMA before the
    training jobs run. The Researcher uses this to propose
    experiments that need novel knobs (e.g., aux_bc_loss_weight for
    DAPG); the Judge passes them through; the orchestrator's Cursor
    path then implements the schema additions before queueing.

    schema_keys override exists for tests; default reads from
    rl_agent.experiment_designs.SCHEMA (with all "_section_*" headers
    excluded).
    """
    if schema_keys is None:
        valid_design_keys = {
            k for k in _EXPERIMENT_DESIGNS_SCHEMA.keys()
            if not k.startswith("_section_")
        }
    else:
        valid_design_keys = set(schema_keys)
    valid_reward_keys = set(REWARD_FUNCTION_NAMES)

    # Phase 1C-Full: harvest the names of any proposed extensions.
    # These keys are accepted as if they were already in SCHEMA -
    # the orchestrator's Cursor agent is responsible for actually
    # adding them before training jobs can run.
    extensions = _get(proposal, "proposed_schema_extensions") or []
    proposed_keys = set()
    for ext in extensions:
        ext_name = _get(ext, "name")
        if isinstance(ext_name, str) and ext_name:
            proposed_keys.add(ext_name)

    bad: List[str] = []
    arms = _get(proposal, "experiment_arms") or []
    for a in arms:
        arm_name = _get(a, "name", "?")
        for k in (_get(a, "experiment_design_fields") or {}).keys():
            if k not in valid_design_keys and k not in proposed_keys:
                bad.append(f"arm '{arm_name}': unknown experiment_design key {k!r}")
        for k in (_get(a, "reward_design_fields") or {}).keys():
            if k not in valid_reward_keys:
                bad.append(f"arm '{arm_name}': unknown reward_design key {k!r}")

    if bad:
        return CheckResult("E", False, "; ".join(bad))
    return CheckResult("E", True, None)


def check_f_safety_critical(proposal) -> CheckResult:
    """F: code_changes_summary doesn't mention any safety-critical
    code path.

    A naive substring match - good enough since the patterns are
    long, distinctive identifiers (not common English words). False
    positives would manifest as the Judge rejecting an otherwise-good
    proposal that happened to mention one of these names in passing;
    we accept that tradeoff for the safety win.
    """
    summary = (_get(proposal, "code_changes_summary", "") or "")
    if not isinstance(summary, str):
        return CheckResult("F", True, None)
    hits = [p for p in SAFETY_CRITICAL_PATTERNS if p in summary]
    if hits:
        return CheckResult("F", False,
            f"code_changes_summary mentions safety-critical paths: {hits}. "
            f"These must remain human-maintained; if your hypothesis needs "
            f"changes here, implement them manually first then re-propose.")
    return CheckResult("F", True, None)


def check_g_reward_invariant_secondary(proposal) -> CheckResult:
    """G: if any arm overrides reward_design_fields, at least one
    reward-invariant secondary metric must be in success_criteria.secondary.

    Goodhart insurance: reward-design experiments are at risk of
    measuring nothing more than "we shifted the reward shape and the
    reward went up". A reward-invariant metric (avg_goals_per_episode,
    avg_speed, avg_episode_length, avg_steering_angle_ratio) tests
    whether the policy actually got better, not just better-at-the-
    reward.
    """
    arms = _get(proposal, "experiment_arms") or []
    touches_reward = any(
        bool(_get(a, "reward_design_fields"))
        or bool(_get(a, "reward_design_id"))
        for a in arms
    )
    if not touches_reward:
        return CheckResult("G", True, None)

    sc = _get(proposal, "success_criteria")
    secondaries = _get(sc, "secondary") if sc is not None else []
    secondaries = secondaries or []

    # We check whether any reward-invariant metric name appears as a
    # substring of any secondary statement. Matches both bare
    # "avg_goals_per_episode" and statements like
    # "avg_goals_per_episode increases by >= 5%".
    secondaries_text = " ".join(str(s) for s in secondaries).lower()
    has_invariant = any(
        m in secondaries_text for m in REWARD_INVARIANT_METRICS)
    if not has_invariant:
        return CheckResult("G", False,
            f"reward-design proposal needs >=1 reward-invariant secondary "
            f"metric. None of {list(REWARD_INVARIANT_METRICS)} mentioned "
            f"in success_criteria.secondary={secondaries!r}.")
    return CheckResult("G", True, None)


def check_h_paper_evidence(proposal) -> CheckResult:
    """H: every cited paper has CONCRETE section_refs AND non-empty
    supporting_evidence.

    The Researcher's earlier failure mode was citing a paper with
    section_refs like ['Langevin-type diffusion variance analysis']
    - thematic descriptors rather than locators. A human reviewer
    can't flip directly to a "Langevin-type diffusion variance
    analysis"; they CAN flip to "Section 4.2" or "Eq. 12". This
    check enforces:

      * Every PaperReference has >=1 section_refs entry matching
        _CONCRETE_LOCATOR_RE (digit, Section, Eq., Theorem, ...).
      * Every PaperReference has supporting_evidence of >=40 chars
        explaining WHY this passage supports the proposal's
        hypothesis.

    PASSES vacuously when no source_papers are cited (pure-codebase
    proposals are a legitimate path).
    """
    papers = _get(proposal, "source_papers") or []
    if not papers:
        return CheckResult("H", True, None)

    problems: List[str] = []
    for idx, p in enumerate(papers):
        arxiv_id = _get(p, "arxiv_id", "?")
        section_refs = _get(p, "section_refs") or []

        # Tolerate either str (rare LLM mistake) or list[str].
        if isinstance(section_refs, str):
            section_refs = [section_refs]

        concrete_refs = [
            s for s in section_refs
            if isinstance(s, str) and _CONCRETE_LOCATOR_RE.search(s)
        ]
        if not concrete_refs:
            problems.append(
                f"paper {arxiv_id!r}: section_refs lacks a concrete locator "
                f"(need at least one entry with a digit or a token like "
                f"'Section', 'Eq.', 'Theorem', 'Algorithm', 'Fig.', "
                f"'Table', '§', 'Page'). Got: {section_refs!r}")

        evidence = _get(p, "supporting_evidence", "")
        if not isinstance(evidence, str):
            evidence = ""
        evidence = evidence.strip()
        if len(evidence) < _MIN_SUPPORTING_EVIDENCE_CHARS:
            problems.append(
                f"paper {arxiv_id!r}: supporting_evidence is too short "
                f"({len(evidence)} chars, need >={_MIN_SUPPORTING_EVIDENCE_CHARS}). "
                f"Quote or paraphrase the specific passage that supports "
                f"this proposal's hypothesis.")

    if problems:
        return CheckResult("H", False, "; ".join(problems))
    return CheckResult("H", True, None)


# ---- Aggregator ----------------------------------------------------------


def run_all(
    proposal,
    *,
    monthly_budget_usd: float = 250.0,
    spent_so_far_usd: float = 0.0,
    probe_urls: bool = False,
    schema_keys: Optional[List[str]] = None,
) -> AllChecksResult:
    """Run all eight checks and return their combined result.

    Independent: a failure in one doesn't short-circuit the others -
    we want a full picture so the resulting judge_review.concerns can
    enumerate every issue rather than one-at-a-time bisecting with
    the operator.
    """
    results = [
        check_a_arxiv_resolution(proposal, probe_urls=probe_urls),
        check_b_hypothesis_and_criterion(proposal),
        check_c_experiment_arms(proposal),
        check_d_budget(
            proposal,
            monthly_budget_usd=monthly_budget_usd,
            spent_so_far_usd=spent_so_far_usd),
        check_e_schema_keys(proposal, schema_keys=schema_keys),
        check_f_safety_critical(proposal),
        check_g_reward_invariant_secondary(proposal),
        check_h_paper_evidence(proposal),
    ]
    return AllChecksResult(
        all_passed=all(r.passed for r in results),
        results=results,
    )
