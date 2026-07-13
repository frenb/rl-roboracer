# MadScientist Autonomous Code-Generation — Design & Plan

Status: **proposal / draft for review**
Owner: (you)
Last updated: 2026-06-13

## 1. Goal

Let the MadScientist research agent expand the **range of experiments it can
propose** by autonomously modifying the codebase — not just tuning existing
hyperparameters. New capabilities (new `experiment_designs.SCHEMA` knobs,
training-loop behavior, replay strategies, eventually reward functions and
algorithms) are implemented by a code-writing agent, shipped as **separate
GitHub branches/PRs**, and only run after passing the guardrails below.

Human-in-the-loop is preserved at two gates:

1. **Proposal approval** — a human approves the scientific proposal
   (`pending_user → approved`) *before* any code is written.
2. **Code gate** — CI must pass; the PR then auto-merges on green (chosen
   policy). CI quality is therefore the safety boundary for the code itself.

## 2. Key finding: ~80% of this already exists

This is not a greenfield build. The pipeline is largely implemented and just
isn't wired end-to-end or enabled.

| Capability | Status | Evidence |
|---|---|---|
| Researcher can emit `proposed_schema_extensions` | **Exists** | `rl_agent/madscientist/researcher.py:363–368`, output schema `:422`, `:449` |
| Proposal model carries impl fields | **Exists** | `schemas.py:359–379` (`proposed_schema_extensions`, `implementation_branch`, `implementation_pr_url`, `implementation_log`) |
| Orchestrator routes schema-ext proposals to a code agent | **Exists** | `orchestrator.py:408` (`_needs_cursor_path`), `:477`, `:524` (`_run_cursor_path`) |
| Cursor cloud agent edits SCHEMA + `robotaxi.py` and opens a PR | **Exists** | `cursor_orchestrator.py` — `Agent.create(... cloud=CloudAgentOptions(repos=[...], auto_create_pr=True))` `:354–365`; branch `xp-<id>/<slug>` `:181`; forbidden paths `:140–144` |
| Lifecycle states `implementing → pr_open → training → done` | **Exists** | `constants.py:48–69` |
| Two human gates (proposal approval, PR review) | **Exists** | email magic-links / dashboard `/madscientist/decide`; PR review on GitHub |
| Safety scaffolding | **Partial** | forbidden-paths (prompt-only), `MAX_JOBS_PER_PROPOSAL`, budget/rate caps, pre-rubric check E |

## 3. Gaps (what blocks it from working end-to-end)

The orchestrator documents the central gap itself:

> "After PR opens: status -> pr_open. Training jobs are NOT queued until a
> follow-up operation (Phase 1C-Full v2) confirms the PR has been merged"
> — `orchestrator.py:438–444`

- **G1 — No merge→train loop.** Proposals sit at `pr_open` forever; nothing
  detects the merge and queues training. *(Phase 1 below.)*
- **G2 — No host code-sync.** The trainer runs the **bind-mounted host
  `./rl_agent`**; the madscientist container mounts the repo **read-only**. A
  merged PR doesn't reach the trainer until the host repo is `git pull`ed
  **and the trainer is restarted** (Python imports the new SCHEMA/`main()` at
  process start). *(Phase 1.)*
- **G3 — Researcher is biased away from code changes.** It's told to "Prefer
  tuning EXISTING SCHEMA knobs … Leave `proposed_schema_extensions = []` if you
  can" (`researcher.py:363`, `:449`). Its action space is numeric SCHEMA knobs,
  not arbitrary capabilities. *(Phase 2.)*
- **G4 — The Cursor path has never run for real.** Tests mock the SDK;
  `MADSCIENTIST_ENABLED` defaults false; `cursor-sdk` install + `CURSOR_API_KEY`
  + `CURSOR_TARGET_REPO` unverified. *(Phase 0.)*
- **G5 — Pre-merge safety is soft.** Forbidden paths are prompt-only; no CI
  gate / branch protection enforced before a merge can trigger training.
  *(Phase 1 + 3.)*

## 4. Phased plan

- **Phase 0 — Validate the existing Cursor path** in a sandbox/fork: enable
  madscientist, install `cursor-sdk`, set keys + `CURSOR_TARGET_REPO`, author
  one tiny schema-extension proposal, confirm it produces a real PR.
- **Phase 1 — Close the merge→train loop** (detailed in §5). *Critical path.*
- **Phase 2 — Expand & un-bias the researcher's action space**: soften the
  prefer-existing bias; generalize `proposed_schema_extensions` →
  `proposed_code_changes` (capability description + target files + **required
  acceptance tests**); update the Cursor prompt, pre-rubric checks, and judge
  rubric to evaluate code-change feasibility/blast-radius/test-plan.
- **Phase 3 — Safety hardening**: CI forbidden-path test, branch protection,
  mandatory agent-authored tests, rollback automation, dashboard controls.
- **Phase 4 — Close the scientific loop**: feed `outcome_ingester` verdicts
  into `research_notes`; git_sha provenance ties results to merged code.

## 5. Phase 1 spec — Merge→Train loop

Decisions baked in: **auto-merge on green CI**; **`git pull` by default,
rebuild when deps change**.

### 5.1 Hard constraint
Deployment of merged code requires **host-level** actions (git pull + restart;
rebuild if deps changed) the sandboxed agent can't do. Design mirrors the
existing desired-gym / Unity-supervisor pattern: the agent **requests** a
deploy; a **host watcher** executes it and reports back.

### 5.2 Components
1. **CI workflow** — `.github/workflows/ci.yml`: `pytest rl_agent/madscientist`
   + trainer import/schema smoke. Marked a **required** check via branch
   protection on `main` (this is what makes auto-merge-on-green safe).
2. **Auto-merge enablement** — extend `cursor_orchestrator.py` (or the
   merge-watcher) to enable GitHub auto-merge on the PR
   (`gh pr merge --auto --squash` / REST `enable_auto_merge`).
3. **Merge-watcher worker** — new `rl_agent/madscientist/merge_watcher.py`
   (mirrors `orchestrate_loop`): polls each `pr_open` proposal's
   `implementation_pr_url` via GitHub API (`GITHUB_TOKEN`); on `merged` →
   record `implementation_merged_sha`, transition `pr_open → deploying`, write
   a deploy request; on `closed`/CI-red-timeout → `failed`.
4. **Host deploy executor** — new `scripts/Deploy-Watcher.ps1` (host,
   long-running): polls deploy requests; **waits for trainer idle** (no
   `IN_PROGRESS` job); `git pull --ff-only origin main`; if merged diff touches
   `requirements.txt`/`Dockerfile*`/`docker-compose*.yml` → `Restart-Stack.ps1
   -Build`, else `Restart-Stack.ps1`; verify trainer up on merged sha; post
   `deploy_done{proposal_id, deployed_sha}`.
5. **Post-deploy queueing** — orchestrator/merge-watcher reuses the existing
   `_queue_jobs` / `_run_auto_queue_path` (`orchestrator.py:365,484`) to queue
   TRAIN jobs; transition `deploying → training`. **Provenance assertion:**
   queued jobs' `trainer_git_sha` must equal the merged sha.

### 5.3 Data-model changes
- `constants.py`: add `STATUS_DEPLOYING`; update DAG to
  `pr_open → deploying → training`; add `AGENT_MERGE_WATCHER`.
- `schemas.py` `Proposal`: add `implementation_merged_at`,
  `implementation_merged_sha`, `deploy_started_at`, `deploy_finished_at`,
  `deployed_sha`, `deploy_failure_reason`.
- New `deploy_requests` collection (or dashboard in-memory state à la
  `/set_desired_gym`).

### 5.4 Sequence (happy path)
1. Cursor agent opens PR + enables auto-merge → `pr_open`.
2. CI green → GitHub auto-merges to `main`.
3. Merge-watcher → `deploying`, writes deploy request (`target_sha`).
4. Deploy-Watcher waits idle → `git pull` → (rebuild if deps else)
   `Restart-Stack` → trainer up on merged sha → `deploy_done`.
5. Orchestrator queues TRAIN jobs (asserts git_sha) → `training`.
6. Trainer runs → `outcome_ingester` verdict → `done` (existing).

### 5.5 Safety / failure handling
- **Serialize deploys** (global lock): one deploy/restart at a time.
- **Idle-gating:** never restart while a job is `IN_PROGRESS` (protects running
  batches).
- **Deploy failure** (pull conflict / build fail / trainer won't boot):
  proposal `failed`; **auto-rollback** to prior good sha + restart; notify.
- **Training crash on merged code:** flag + open a revert PR (Phase 3 hook).
- **CI red:** PR never merges; auto-reject after `AUTO_REJECT_AFTER_HOURS`, or
  (Phase 3) feed logs back to the Cursor agent to self-fix.

### 5.6 Testing
- Unit-test `merge_watcher` with a mocked GitHub client (merged/closed/pending),
  mirroring how `cursor_orchestrator` tests mock the SDK.
- Unit-test post-merge queueing reuses `_queue_jobs` + the sha guard.
- `Deploy-Watcher.ps1` dry-run mode (log actions, skip real restart).

## 6. Open decisions
- Target repo & branch protection config (this repo vs. dedicated fork).
- CI suite scope (how much to run on every agent PR vs. nightly).
- Host deploy privilege scoping (the Deploy-Watcher is the most
  security-sensitive new surface; only act on merged-to-main shas).
- Budget for code-writing agent runs (pricier than research/judge LLM calls).
- Phase 2 action-space scope: schema-knobs + tied behavior only, vs. arbitrary
  capability proposals.

## 7. Highest-leverage first build
Because auto-merge makes **CI the only code gate**, the first thing to build is
the **CI suite + a forbidden-path CI test** (fail the build if a PR touches
`_emergency_pause_handler`, `dashboard/`, `virtual_endpoint/`, etc. — turning
the prompt-only rule in `cursor_orchestrator.py:140–144` into an enforced
check). Everything else in Phase 1 leans on it.
