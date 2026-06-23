"""Constants for the MadScientist agent system.

Status enum + collection names + default configuration knobs. Imported by
schemas, seed, the worker stubs, and (later) the dashboard server route
that serves the activity feed.
"""

# ---- Mongo collection names ----------------------------------------------

# New collections created by seed.py. Each is keyed under db = robotaxi so
# they live alongside the existing jobs/models/leaderboard_scores/etc.
COLL_PROPOSALS = "proposals"
COLL_RESEARCH_NOTES = "research_notes"
COLL_JUDGE_RUBRIC_HISTORY = "judge_rubric_history"


# ---- Proposal lifecycle states -------------------------------------------
#
# Status transitions form this DAG (terminal states marked *):
#
#   pending_judge
#         |
#         v
#   pending_user
#     |       |
#     v       v
#   approved  rejected*
#     |
#     v
#   implementing
#     |
#     v
#   pr_open
#     |
#     v
#   training
#     |
#     v
#   done*
#
# Plus two "wildcard" terminal states reachable from anywhere:
#   failed*    - any worker hit an unrecoverable error
#   cancelled* - operator clicked Cancel mid-flight
#
# `deferred` is a soft state used when the user replies "later"; reverts
# to pending_user when the user revisits.

STATUS_PENDING_JUDGE = "pending_judge"
STATUS_PENDING_USER = "pending_user"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_DEFERRED = "deferred"
STATUS_IMPLEMENTING = "implementing"
STATUS_PR_OPEN = "pr_open"
STATUS_TRAINING = "training"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# Convenience groupings for dashboard queries.
STATUSES_TERMINAL = (
    STATUS_REJECTED, STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)
STATUSES_AWAITING_USER = (STATUS_PENDING_USER, STATUS_DEFERRED)
STATUSES_IN_FLIGHT = (
    STATUS_APPROVED, STATUS_IMPLEMENTING, STATUS_PR_OPEN, STATUS_TRAINING)
STATUSES_ALL = (
    STATUS_PENDING_JUDGE, STATUS_PENDING_USER, STATUS_APPROVED,
    STATUS_REJECTED, STATUS_DEFERRED, STATUS_IMPLEMENTING, STATUS_PR_OPEN,
    STATUS_TRAINING, STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)


# ---- Judge verdict tones (rubric output) ---------------------------------
#
# Overall verdict labels used by the Judge worker. Mapped to user-facing
# colors / icons on the dashboard's pending-decision card.

JUDGE_STRONG_ACCEPT = "strong_accept"
JUDGE_ACCEPT = "accept"
JUDGE_WEAK_ACCEPT = "weak_accept"
JUDGE_WEAK_REJECT = "weak_reject"
JUDGE_REJECT = "reject"

JUDGE_VERDICTS = (
    JUDGE_STRONG_ACCEPT, JUDGE_ACCEPT, JUDGE_WEAK_ACCEPT,
    JUDGE_WEAK_REJECT, JUDGE_REJECT)


# ---- Outcome verdicts (after training finishes) --------------------------

OUTCOME_SUPPORTED = "supported"     # primary_criterion_met == True
OUTCOME_REJECTED = "rejected"        # explicit fail of the criterion
OUTCOME_INCONCLUSIVE = "inconclusive"  # underpowered / noisy / partial data

OUTCOMES_ALL = (OUTCOME_SUPPORTED, OUTCOME_REJECTED, OUTCOME_INCONCLUSIVE)


# ---- Agent identity strings (used in audit_events.by_agent) --------------
#
# Distinct from job_id / proposal_id - identifies WHICH worker process
# made a state transition, for debugging multi-worker race conditions
# later.

AGENT_RESEARCHER = "madscientist.researcher"
AGENT_JUDGE = "madscientist.judge"
AGENT_ORCHESTRATOR = "madscientist.orchestrator"
AGENT_OUTCOME_INGESTER = "madscientist.outcome_ingester"
AGENT_EMAIL_BRIDGE = "madscientist.email_bridge"
AGENT_USER = "user"
AGENT_SYSTEM = "system"


# ---- Default configuration -----------------------------------------------
#
# All knobs are read from env vars (see docker/madscientist/Dockerfile +
# docker-compose.yml service block). These constants are the FALLBACK
# defaults applied when the env var is missing or unparseable.

DEFAULT_MAX_PROPOSALS_PER_DAY = 1
DEFAULT_BUDGET_USD_PER_MONTH = 250.0
DEFAULT_RESEARCH_CYCLE_INTERVAL_SECONDS = 6 * 60 * 60  # 6h
DEFAULT_OUTCOME_POLL_INTERVAL_SECONDS = 5 * 60          # 5min
DEFAULT_AUTO_REJECT_AFTER_HOURS = 96                    # 4 days
# Maximum number of NOT_STARTED jobs allowed in the queue before the
# researcher skips a proposal cycle. Prevents the agent from piling up
# more experiments than the trainer can realistically run.
DEFAULT_MAX_QUEUED_JOBS = 3

# Hard guardrails - the worker REFUSES to start a cycle if any of these
# would be exceeded by the cycle. Mostly a backstop against budget bugs.
HARD_CAP_PROPOSALS_PER_DAY = 5
HARD_CAP_BUDGET_USD_PER_MONTH = 1000.0
