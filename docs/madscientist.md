# The Mad Scientist

The **Mad Scientist** (`rl_agent/madscientist/`) is an opt-in autonomous
RL-research agent: it reads recent arXiv papers + the codebase, drafts
experiment **proposals**, LLM-judges them against a rubric, emails you for
approval, then queues the approved experiments as TRAIN jobs and tallies the
results back into the proposal. It runs as a **separate Docker container**
(`docker/madscientist/`, compose service `madscientist`), shares MongoDB with the
trainer + dashboard, and is **off unless `MADSCIENTIST_ENABLED=true`**.

**Process model** (`madscientist/main.py`). Five daemon worker threads poll
MongoDB, each on its own cadence:

| Worker | File | Role |
|---|---|---|
| Researcher | `researcher.py` | Phase 1B — draft proposals from arXiv + codebase context |
| Judge | `judge.py` | Phase 1A — score a proposal against the rubric via Claude |
| Orchestrator | `orchestrator.py` | Phase 1C — turn an *approved* proposal into TRAIN jobs |
| Outcome ingester | `outcome_ingester.py` | Phase 1E — compute per-arm stats + verdict when jobs finish |
| Email bridge | `email_bridge.py` | Phase 1D — email proposals with signed magic-link buttons |

**Lifecycle DAG** (`constants.py`):

```
pending_judge → pending_user → approved → implementing → training → done
                     ↘ rejected                       (or → pr_open via Cursor path)
        (+ wildcard terminals: failed / cancelled; soft: deferred)
```

**Collections** (`madscientist/schemas.py`, Pydantic v2): `proposals` (the
central document — hypothesis, `experiment_arms`, `success_criteria`,
`judge_review`, `decision`, `training_job_ids`, `results`, `cost`, and an
append-only `audit_events` trail), `research_notes` (append-only working memory),
and `judge_rubric_history` (versioned rubric snapshots — new versions appended,
never edited). It also reads/writes the trainer's `jobs`, `experiment_designs`,
`reward_designs`, `models`, `leaderboard_scores`.

**Researcher** (model `RESEARCHER_MODEL`, default Claude Opus). One cycle per
`RESEARCH_CYCLE_INTERVAL_SECONDS` (6h). Gated by a daily proposal cap
(`MAX_PROPOSALS_PER_DAY`), a monthly budget (`BUDGET_USD_PER_MONTH`), and a
**queue-depth cap** (`MAX_QUEUED_JOBS`, skip if too many unreviewed proposals).
It fetches recent arXiv (last 30 days), a compact codebase snapshot (the
`experiment_designs.SCHEMA`, named designs, top global-best models, recent
proposal verdicts), then self-critiques up to 2 revisions against the
**pre-rubric checks**, HTTP-probing every cited arXiv id to catch hallucinated
papers, before inserting a `pending_judge` proposal.

**Judge** (model `JUDGE_MODEL`). Processes one `pending_judge` proposal per poll.

1. **Pre-rubric checks A–H** (`pre_rubric_checks.py`) run *first* and cheaply: A
   arXiv ids resolve (HTTP-probed), B hypothesis + primary criterion present, C
   ≥2 arms with exactly one `base`, D fits the budget, E every
   design/reward field key is known or declared as a `proposed_schema_extension`,
   F `code_changes_summary` names no **safety-critical path**, G reward-touching
   arms declare a reward-invariant secondary metric (Goodhart insurance), H every
   cited paper has a concrete section ref + evidence. Any failure → reject
   **without** calling the LLM (saves ~$0.50/rejection).
2. **LLM scoring** against the full rubric (`JUDGE_RUBRIC.md`) on **8 axes**
   (0–5): hypothesis specificity/falsifiability, novelty, significance,
   statistical power & baseline rigor, Goodhart resistance, paper faithfulness,
   implementation feasibility, cost & reproducibility.
3. **Mechanical verdict** (`compute_verdict`) — the LLM emits only the axis
   scores; the verdict is computed from the normalized sum against fixed
   thresholds (≥0.875 strong_accept … <0.350 reject), with a hard override that
   **any axis scored 0 forces reject**. Not trusting the LLM with the verdict
   prevents score inflation.

Reject → `rejected`; otherwise → `pending_user`.

**Email + approval** (`email_bridge.py`). For each `pending_user` proposal it
sends a rich HTML email (judge scores, arms table, success criteria, source-paper
cards) with **Approve / Reject / Defer** buttons. Each button is an
HMAC-`MADSCIENTIST_TOKEN_SECRET`-signed magic link to
`<DASHBOARD_PUBLIC_URL>/madscientist/act?token=…` (TTL `DECISION_TOKEN_TTL_SECONDS`,
7d). The dashboard verifies the HMAC (`crypto.timingSafeEqual`) and applies the
decision single-use (gated on status ∈ `{pending_user, deferred}`). If
`MADSCIENTIST_TOKEN_SECRET` is empty the buttons are omitted (dashboard-only).

**Orchestrator → jobs** (`orchestrator.py`). One `approved` proposal per poll:
for each `(arm × seed)` it creates a derived `experiment_designs` doc
(`auto:<pid8>:<arm>`, cloning the base + overlaying the arm's field overrides,
stamping `num_iterations` from `num_iterations_per_seed`) and inserts a
`NOT_STARTED` TRAIN `jobs` doc mirroring the New-job form, **linked back via
`proposal_id` (str of ObjectId) + `proposal_arm`**. The trainer's normal poll
loop then picks them up unchanged. Proposals with `proposed_schema_extensions`
instead take the **Cursor path** (`cursor_orchestrator.py`): a Cursor cloud agent
writes the new SCHEMA knob + plumbing and opens a PR (`pr_open`).

**Outcome feedback** (`outcome_ingester.py`, no LLM). Once **all** of a proposal's
linked jobs are terminal, it groups each job's best-model `avg_return` by arm,
computes per-arm mean/median/stddev + a bootstrap 95% CI, evaluates the parsed
primary criterion (`delta = mean(arm_a) − mean(arm_b)` vs a relative/absolute
threshold), and writes a verdict — `supported` / `rejected` / `inconclusive` —
flipping the proposal to `done`.

**Key env vars:** `MADSCIENTIST_ENABLED`, `ANTHROPIC_API_KEY` (required when
enabled), `BUDGET_USD_PER_MONTH`, `MAX_PROPOSALS_PER_DAY`, `MAX_QUEUED_JOBS`,
`MAX_JOBS_PER_PROPOSAL`, `MADSCIENTIST_TOKEN_SECRET` (must match the dashboard
service), `DASHBOARD_PUBLIC_URL`, `SMTP_*` / `NOTIFICATION_EMAIL`, and
`CURSOR_API_KEY` / `CURSOR_TARGET_REPO` for the codegen path. Dashboard tab:
`/madscientist` (`dashboard/madscientist.html`); API under
`/madscientist/*` in `server.ts`.

**Known limitations / gotchas:**

- **`pr_open` is a dead-end.** Nothing detects a merged PR and queues training,
  and the container mounts the repo read-only while the trainer runs the
  host-bind-mounted `./rl_agent` — so a merged codegen PR doesn't reach the
  trainer without a host `git pull` + restart. The Cursor path is unproven
  (disabled by default). See `docs/madscientist-autonomous-codegen.md`.
- **Verdict ignores the CIs.** The outcome ingester records per-arm bootstrap
  95% CIs and stddev but decides `supported`/`rejected` purely on the
  point-estimate delta vs threshold; `primary_p_value` is always `None`.
- **`AUTO_REJECT_AFTER_HOURS`** is defined (compose + `constants.py`) but no
  worker reads it — there is no auto-expiry sweep; deferred/pending proposals
  persist indefinitely.
- **Enum/collection-name duplication** between Python (`constants.py`) and
  TypeScript (`server.ts`) — kept in sync by hand (drift risk).
- **Naming drift:** several docstrings/prompts say "7 pre-rubric checks A–G"
  even though check **H** (paper evidence) is implemented and enforced.
- **LLM spend only** is tracked in `proposal.cost`; training compute isn't
  reconciled into the budget accounting.
- Model ids (`claude-opus-4-7`, `composer-2.5`) are placeholders that must match
  the deployment.
