# Curriculum Builder (cross-robot)

**Status:** Phase 1 (core) done, Nov 17, 2026; see "Phases". The OpenBeni track depends on `robot-ingestion-and-openbeni-plan.md` Phase E (an `openbeni` TRAIN → EXPORT → EVAL job working through the queue).

**Goal:** one cross-robot way to define and run a curriculum: an ordered list of training stages, each with a pass condition, that the platform trains stage by stage, advancing automatically. It replaces the car's `curriculum_stages` experiment-design field and is the only place curricula are authored, for `roboracer`, `openbeni` and later `microduck`. For OpenBeni it covers low-level control (balance, driving, running, jumping) up to obstacle avoidance and path planning.

**Decisions (Nov 17, 2026):**

- **Single source of truth.** Curricula live only in the builder (`db.curricula`). Experiment designs no longer carry stage lists; jobs reference a curriculum run instead.
- **Robots:** `roboracer` (migration of the existing track curriculum) and `openbeni` (new skill curricula) first; `microduck` follows on the same `mjlab_core` path.
- **Composition (OpenBeni):** hybrid. Low-level stages fine-tune one network; from obstacle avoidance up, low-level skills are frozen and a separate high-level policy learns to command them.
- **Gating:** automatic. A stage passes when its evaluation meets the stage's condition; otherwise it retries with more iterations, then stops the run and flags it.
- **One job per stage, for every robot (Nov 17, 2026).** Stage changes never happen inside a running job. What a stage inherits from the one before it (weights, optimizer and entropy state, replay buffer, normalizer statistics) is declared per stage in the curriculum and restored by the next job; see "Handoff between stages". The car's in-job `CurriculumScheduler` is retired.

### Scope: what counts as a curriculum

The builder owns every **user-authored sequence of training conditions**: which environment settings, task, reward or starting checkpoint apply, in what order, and what has to be true to move on.

It does *not* own mechanisms that run *within* one stage as part of an algorithm or task. Those stay as experiment-design fields or task code:

- the JSRL guide→learner handoff schedule (`rl_agent/jsrl.py`);
- mjlab task-internal schedules (`fallen_reset_curriculum`, terrain levels, Beni's jump-frequency ramp), which stay switchable via the existing `curricula` experiment-design toggle.

---

## Today

| Robot | Mechanism | Where | Notes |
|---|---|---|---|
| `roboracer` | Track-geometry ladder inside one TRAIN job | `curriculum_stages` / `curriculum_start_stage` experiment-design fields → `CurriculumScheduler` in `rl_agent/robotaxi.py`; `_apply_stage` reconfigures Unity's `TrackGenerator` live | Gate: top-3 goals/ep of the latest greedy eval ≥ `advance_goals` for `consecutive` cycles. 4 active experiment designs use it. |
| `roboracer` | Same stage list read by other features | `_resolve_eval_curriculum_stages` / `_eval_over_curriculum` (EVAL `eval_all_stages`, Models-tab Curriculum sweep), DEMO recording, dashboard (`jobs.html`, `models.html`, `experiment_designs.html`, `components.js`, `server.ts` schema fallback), `dashboard/tools/audit_experiment_designs.js`, MCP `create_job` (`eval_all_stages`), seed data (`scripts/install/seed/seed.json`) | All read stages via the job's experiment design. |
| `roboracer` | Manual stage picker | Unity `CurriculumStageButtons` (key `C`) | Trainer-less debug convenience. |
| `roboracer` | Curricula as sequences of jobs | MadScientist (`researcher.py`) proposes "arms/jobs with decreasing `corner_radius`" | Ad hoc, no runner. |
| `openbeni` / `microduck` | Warm start between jobs | `init_checkpoint` on a job (`mjlab_core/launch.py`), shape check `check_warm_start` | No runner, no gates. |

---

## Concepts

- **Curriculum**: an ordered list of stages for one robot, versioned like reward/experiment designs.
- **Stage**:
  - `settings`: robot-specific stage knobs, validated by the robot's **stage schema** (published by its worker, like the experiment-design schema). Car: `corner_radius`, `chicanes_{north,east,south,west}`. OpenBeni: command ranges (`cmd_lin_vel_x_max`, `cmd_ang_vel_z_max`, `standing_fraction`), jump fields.
  - `gym`, `reward_design_id`, `experiment_design_id`, `max_iterations` (or `max_steps` for the car). Unset fields inherit from the previous stage, so a stage that only changes track geometry lists just that.
  - `init`: `previous` (from the last passed stage; the default), `model:<id>`, or `none`.
  - `carry`: which training state comes from `init` (see "Handoff between stages"). Defaults per robot.
  - `kind`: `train` (trains the robot's network) or `skill` (OpenBeni: trains a high-level policy over frozen skills listed in `skills`).
  - `gate`: list of `{metric, op, value}`, plus `consecutive` (evaluations in a row that must pass) and `eval_episodes`. The last stage may omit the gate (train to budget). The job stops early as soon as the gate passes, so `max_*` is a ceiling, not a fixed cost.
  - `on_fail`: `{retries, extra_iterations}`; after retries the run stops as `FAILED_GATE`.
- **Curriculum run**: one execution of a curriculum (a snapshot of its stages) plus run options (`start_stage`, `priority`). Records per stage: job id, model id, evaluation metrics, pass/fail, and when it passed.
- **Skill**: a model promoted from a passed stage, with its command interface recorded, reusable by later stages and other curricula.

### Data model (Mongo)

```text
curricula        {_id, robot, name, version, archived, stages: [Stage], created_at, updated_at}
curriculum_runs  {_id, curriculum_id, curriculum_version, stages (snapshot), options, status,
                  current_stage, results: [{stage_id, attempt, job_id, model_id, eval, passed, at}],
                  created_at, updated_at}
jobs             + curriculum_run_id, curriculum_stage_id, curriculum_attempt,
                   stage_settings (resolved), gate (resolved),
                   handoff: {source_job_id, checkpoint, carry: [...], artifacts: {...}}
models           + curriculum_run_id, curriculum_stage_id,
                   skill: {name, command_interface, source_run_id, frozen_pt, onnx}
experiment_designs  - curriculum_stages, curriculum_start_stage  (removed; see migration)
```

---

## Runner

A robot-agnostic package `rl_agent/curriculum/` (`model.py`, `validate.py`, `runner.py`), run as a small `curriculum-runner` service (Python + pymongo, no GPU) so it works the same for the TF/Unity car worker and the mjlab workers.

1. **Starting a run** resolves stage 1 (or `start_stage`) into a job document (`stage_settings`, `gate`, designs, budget, `handoff`) and queues it.
2. **During the job** the trainer evaluates the gate at its normal evaluation points and finishes the job early once the gate has passed `consecutive` times. It writes the last evaluation's gate metrics and a `stop_reason` (`gate_passed` | `budget`) onto the job.
3. **When the job finishes** the runner records the result, then:
   - pass → promote the model to a skill if the stage says so; resolve and queue the next stage with its `handoff`;
   - fail with retries left → queue a continuation of the same stage from this job (`carry: all`) with `extra_iterations`;
   - fail with no retries → run status `FAILED_GATE`.
   If the job has no gate metrics (e.g. it crashed after training), the runner queues an EVAL job and decides on its result.
4. **Idempotent:** every decision is derived from Mongo state, so a restart re-derives the same next action instead of double-queuing; the runner also sweeps active runs on start.
5. **Queue order:** the next stage goes to the front of the shared queue by default, so no other job slips between stages and the stage boundary costs only the job's startup time; `priority: normal` appends to the end instead.

### Handoff between stages

Two kinds of things pass from one job to the next.

**1. Parameters the curriculum declares.** Stage settings, designs and budgets are resolved by the runner when it queues a stage (stage values over inherited values over robot defaults) and written onto the job as plain fields. Workers read them like any other job field; the curriculum is the only place they're authored.

**2. Training state produced by the previous job**, selected by the stage's `carry` list. The runner writes the source job, checkpoint path and the chosen items into `job.handoff`; the worker restores only those.

| `carry` item | Car (SAC/AWAC, TF) | mjlab (PPO, PyTorch) | Allowed when |
|---|---|---|---|
| `weights` | actor, critics, target critics | actor, critic | same network shapes (obs/action specs and hidden dims) |
| `optimizer` | actor/critic/alpha optimizer state, train step counter | Adam state, iteration counter | `weights` carried |
| `entropy` | learned α (log-alpha) | action-noise std (part of actor) | `weights` carried |
| `normalizer` | (none today) | observation normalizer statistics | `weights` carried |
| `replay` | Reverb online table (checkpoint) | n/a (on-policy) | same gym (same observation spec) **and** same reward design (stored rewards would be stale otherwise) |
| `jsrl_guide` | frozen BC guide weights | n/a | JSRL enabled in both stages |

Defaults: car `[weights, optimizer, entropy, replay]` when gym and reward are unchanged, else `[weights]`; mjlab `[weights, optimizer, entropy, normalizer]`. The runner validates the list against the "allowed when" column before queuing, and rejects the curriculum at save time if a stage requests an impossible carry.

**Artifacts layout.** A job's handoff artifacts live under its own job directory (car: `<tempdir>/<job_id>/learner/` checkpoint + `reverb_checkpoint/`; mjlab: the run's `model_<iter>.pt`, which already includes optimizer state). The next job **copies** them into its own directory before starting, so each job's directory stays self-contained and a failed stage can be retried from the same source. Old handoff artifacts are pruned when a run finishes (keep the final stage and any promoted skill).

**What's gained and lost against today's in-job ladder (car):** carrying `weights + optimizer + entropy + replay` reproduces the learner state exactly; what's new per stage is process startup (Unity reconnect, demo load + demo-table prefill, which isn't serialised today), plus the replay copy. Stage changes become visible, retryable units with their own model, metrics and video.

### Gate metrics

Each robot's evaluator reports named metrics; gates refer to them by name.

- **Car:** `eval_goals_top3` (today's signal: top-3-by-goals mean of the latest greedy eval), `eval_goals_mean`, `crash_rate`, `lap_time`.
- **mjlab (`mjlab_core/evaluate.py` gains per-task hooks):** `fall_rate`, `mean_return`; velocity tasks: linear/angular tracking error; recovery success (reuse `beni_rl.recovery_evaluation`); jump success and apex height (reuse `beni_rl.jump_evaluation`); navigation: goal-reached rate, collision rate, time to goal.

---

## Track 1: Migrate the car's curriculum

The training outcome of existing car curricula should match today's; what changes is that each track stage becomes its own job.

1. **Stage schema for `roboracer`:** `corner_radius`, `chicanes_*` (`curvature_difficulty` dropped from new stages; still read on old snapshots for logging). Old `advance_goals` + `consecutive` map to `gate: [{metric: eval_goals_top3, op: ">=", value}], consecutive`.
2. **Cross-job warm start in the car trainer (new).** Today the trainer can only resume *its own* paused job (`is_resume_val`: tf-agents `Checkpointer` in `<tempdir>/<job_id>/learner/`, and Reverb restores from `reverb_checkpoint/` at server construction). Add a seeded start: copy the `handoff` items from the source job's directory into the new job's, then take the resume path for those items only, with the new stage's track geometry. Skip BC pretrain and the pre-loop eval when `weights` are carried; redo the demo-table prefill (it isn't checkpointed); restore the JSRL guide when `jsrl_guide` is carried. The resume path once failed silently for months (the Reverb `set_checkpoint` bug), so this gets an explicit test: after a seeded start, the online table size, `train_step`, α and a fixed-batch actor output must match the source job's final values.
3. **Gate early stop in the car trainer.** `CurriculumScheduler` is replaced by a gate check at each eval cycle using the same signal (top-3 goals/ep of the latest greedy eval, `consecutive` cycles); when it passes, the job saves its final checkpoint and handoff artifacts and ends with `stop_reason: gate_passed`. `curriculum_stages_val` / `curriculum_start_stage_val` are removed; the run option `start_stage` replaces the latter.
4. **Fixed-geometry jobs:** `corner_radius` / `chicanes_*` on an experiment design for a job with no curriculum keep working as-is for now (a single fixed stage is not a curriculum). Open question below on moving them to stage settings everywhere.
5. **Other readers move to the curriculum:** EVAL `eval_all_stages` and the Models-tab Curriculum sweep (sweep the stages of the model's `curriculum_run_id`), DEMO recording, the dashboard job/model views, the audit tool, MCP `create_job`, and the seed data.
6. **MadScientist** proposes curricula through the builder (create a curriculum + start a run) instead of chains of arms with decreasing `corner_radius`.
7. **Unity `CurriculumStageButtons`** stays a manual debug tool, with its presets loaded from a chosen curriculum instead of hard-coded stages.
8. **Migration script:** for each experiment design with `curriculum_stages` (4 today), create a `roboracer` curriculum with the same stages, remove the fields from the design, and record the mapping (`migrated_from_experiment_design`). Old jobs and models keep resolving their stages through that mapping for sweeps. Remove both fields from the experiment-design schema (`rl_agent/experiment_designs.py` and the `server.ts` fallback).
9. **Parity check:** run one migrated 5-stage curriculum as 5 chained jobs with full carry, against the same design run in-job on the current code (before step 3 lands). Compare steps-to-advance per stage and final eval goals/ep; record the per-boundary startup cost from the trainer's `[startup-timing]` breakdown.

---

## Track 2: OpenBeni skill curricula

### Low-level chain (`train` stages, one job each)

OpenBeni drives on its wheels, so "walk → run" is a widening speed range rather than a gait change.

| # | Stage | Gym | Main settings | Pass condition (starting guess, to tune) |
|---|---|---|---|---|
| 1 | Balance + recovery | `Beni-Velocity-Flat` | Standing / near-zero commands; fallen resets on | fall rate < 5%; recovery success > 90% |
| 2 | Slow driving | `Beni-Velocity-Flat` | `vx ∈ ±0.5`, `ωz ∈ ±1.5` | tracking error < 0.15 m/s; fall rate < 5% |
| 3 | Full-speed driving | `Beni-Velocity-Flat` | `vx ∈ ±2.0`, `ωz ∈ ±5.0` (task default) | tracking error < 0.25 m/s; fall rate < 5%; recovery success > 85% |
| 4 | Jumping | `Beni-Jump-Flat` | Recovery gate + jump curriculum (existing) | jump success > 80%; fall rate < 10% |

Stages 1–3 need the OpenBeni stage-schema fields for command ranges, applied in `rl_agent/openbeni/cfg_hook.py` to `cfg.commands["twist"].ranges`. Stage 4 uses the jump fields from the OpenBeni plan (Phase C: `jump_warmup_iters`, `jump_ramp_iters`, `jump_probability`). The output of stage 4 is the **low-level skill**: one GRU policy that obeys velocity, pose and jump commands.

### High-level layers (`skill` stages)

A new task family in `rl_agent/openbeni/tasks/` (not in the submodule) where:

- The **low-level policy is frozen** and runs inside the environment every control step (50 Hz), batched on the GPU, with its GRU state reset per environment on `done`.
- The **high-level policy acts at a lower rate** (e.g. 10 Hz); its action is the low-level command `(vx, ωz, base height, jump trigger)`. Pitch, roll and gimbal stay at defaults initially.
- High-level observations are task-specific (goal position, obstacle sensing), not the 40-D proprioception.
- It trains with the same `rsl_rl` PPO runner, so the worker, export and evaluation apply unchanged. A `skill` stage's export contains only the high-level network; the low-level skill is referenced by model id.

| # | Stage | Task | Observations | Pass condition |
|---|---|---|---|---|
| 5 | Go to goal | `Beni-Nav-Goal-Flat` | goal position/heading in body frame, base velocity | goal reached > 90% within time limit |
| 6 | Obstacle avoidance | `Beni-Nav-Obstacles` | + obstacle sensing (privileged nearest-obstacle positions first, then ray casts / height scan) | goal reached > 85%; collisions < 5% |
| 7 | Obstacle hopping | `Beni-Nav-Obstacles-Low` | + low obstacles that must be jumped | goal reached > 80%; jump-over success > 70% |
| 8 | Path planning | `Beni-Nav-Maze` | waypoints from a classical planner (A* over an occupancy grid) fed to the stage-6/7 policy | goal reached > 80% on unseen maps |

Path planning starts classical (the planner produces waypoints, the learned policy follows them locally). A learned planner can come later if the classical one is the bottleneck.

---

## Interfaces

- **Dashboard:** a Curricula tab with a stage list editor (gym / designs / budget / settings from the robot's stage schema / gate / `carry` checkboxes, with invalid carries greyed out and explained) and a run view showing each stage's job, metrics against the gate, and pass/fail. The Experiment Designs tab loses the stage-list editor and links to Curricula instead.
- **Jobs tab: collapsible curriculum runs.** One job per stage would otherwise flood the table (a 5-stage run with a retry is 6+ rows), so jobs that belong to a curriculum run are nested under one row for the run.
  - **Run row (collapsed by default):** curriculum name and version, robot, run status (`RUNNING`, `PASSED`, `FAILED_GATE`, `CANCELLED`), "stage 3 / 5 · Slow driving", a progress bar across stages with the current stage's own progress inside it, the current job's status, the latest gate metric against its threshold (e.g. `goals_top3 7.2 / 9`), total steps and elapsed time. The active stage is readable without expanding.
  - **Expanded:** one child row per stage job in stage order, each with the usual job columns plus a Stage column (`2 / 5 · attempt 2`), a gate result pill (passed / failed / running, with the metric), `stop_reason`, and what was carried in. Stages not queued yet show as greyed placeholder rows, so the remaining ladder is visible.
  - **Behaviour:** pagination and newest-first sorting work on top-level rows, with a run sorted by its latest job activity, so active runs stay near the top. Expand state is remembered per run (localStorage). Robot / type / status filters match a run if any of its jobs match. Deep links to a stage job (`roboracer:open-job`) expand its run and scroll to it.
  - **Run actions:** cancel run (cancels the queued or running stage job and stops the runner), retry the failed stage, open in the Curricula tab. Deleting a run row asks before deleting its jobs.
  - **Toolbar toggle** "Group curriculum jobs" (on by default) switches back to the flat list.
  - **Implementation:** Tabulator `dataTree` (`dataTreeChildField`, `dataTreeStartExpanded: false`), exposed through `TableView` in `components.js` as a `tree` option. The page builds the tree client-side from `/get_jobs` (which already returns `curriculum_run_id` / `curriculum_stage_id`) plus a new `/get_curriculum_runs`; change-stream updates on `curriculum_runs` refresh run rows in place.
- **Models tab:** the same grouping: a run row with one model per passed stage, promoted skills badged.
- **MCP console:** `list_curricula`, `get_curriculum`, `create_curriculum`, `update_curriculum`, `set_curriculum_archived`, `get_curriculum_stage_schema`, `start_curriculum_run`, `get_curriculum_run`, `cancel_curriculum_run`. `create_job` loses stage fields; `eval_models` sweeps take a curriculum run or curriculum id. `list_jobs` groups jobs by curriculum run by default (one entry per run with its stage jobs inside; `flat: true` for the old list) and accepts a `curriculum_run` filter.

---

## Phases

1. **Core.** Data model, validation (including `carry` rules), stage-schema publishing per robot, runner service, MCP tools, tested with fake jobs (no GPU). The Jobs-tab grouping (collapsible run rows) ships here too, before the first real chained runs add jobs to the table.
   *Done (Nov 17, 2026):* `rl_agent/curriculum/` (model, runner, Mongo store, CLI, service; `test_curriculum.py` in CI), the `curriculum-runner` compose service, dashboard routes, console tools (`*_curriculum*`, `list_jobs` `group_curricula`), and Jobs-tab run rows with Cancel / Retry. Two departures from the text above:
   - What a robot supports is declared in a `curriculum:` block of its `robots/<id>/robot.yaml` (enabled, budget unit, checkpoint field, restorable carry items, gate metrics, stage settings) rather than published by the worker. Only `openbeni` is enabled: its warm start restores the whole rsl_rl state, so carry is all-or-nothing there (`carry_atomic`), and it has no stage settings yet (phase 3). The car stays off until phase 2, so a car curriculum can't silently start each stage from scratch.
   - With no early stop yet, a gated stage trains to its budget and is then judged by an EVAL job the runner queues on its checkpoint (`eval_results` against the gate); `consecutive > 1` is refused until a trainer supports early stop. A missed gate retries from the failed attempt's checkpoint with `on_fail.extra_budget`, then stops the run as `FAILED_GATE`.
2. **Car migration (Track 1).** Seeded start with handoff restore, gate early stop, migration of the 4 designs, readers moved over, fields removed from the experiment-design schema. *Done when* the parity check passes and no code reads `curriculum_stages` from experiment designs.
3. **OpenBeni low-level chain.** Command-range stage fields, velocity/recovery/jump metrics, seed `openbeni-lowlevel`. *Done when* a run goes from stage 1 to stage 4 unattended and its final model exports and evaluates.
4. **Skill library + high-level environment.** Skill promotion, frozen low-level wrapper, `Beni-Nav-Goal-Flat` (stage 5).
5. **Obstacles, jumping obstacles, path planning** (stages 6–8).
6. **Dashboard Curricula tab**, MadScientist and Unity picker switched over, `microduck` stage schema.

Phases 2 and 3 are independent once phase 1 lands.

## Open questions / risks

- **Fixed track geometry outside curricula:** keep `corner_radius` / `chicanes_*` on experiment designs for single-stage jobs (current plan), or move them to stage settings everywhere so a plain job is a one-stage curriculum. The latter is purer but touches every car job.
- **Car stage-boundary cost:** unmeasured. Each boundary pays trainer startup (Unity reconnect, demo load and demo-table prefill) plus copying the Reverb checkpoint (75k-step table; camera observations make it the largest artifact). The parity check measures it; if it's too slow, checkpoint the demo table too and hard-link rather than copy the Reverb files.
- **Gate early stop for mjlab:** `rsl_rl` doesn't evaluate during training. Add a runner callback in `train_entry.py` that evaluates the gate every `gate_every` iterations on a small set of eval environments, or gate on training-episode statistics as the car's JSRL gate does; pick after measuring eval cost on the GPU.
- **Sensing in mjlab 1.6:** confirm which ray-cast / height-scan sensors exist; otherwise implement ray casts against obstacle geometry with MuJoCo Warp's ray functions.
- **Low-level skill robustness:** stages 1–4 train on flat ground with no obstacles; contact with obstacles may need a low-level fine-tune with obstacles present.
- **Forgetting during fine-tune chaining:** later gates re-check earlier skills (e.g. stage 3 also gates on recovery success).
- **GPU sharing:** the queue runs one job at a time across robots, so curriculum runs interleave with other jobs; the `priority` option mitigates this.
- **Gate thresholds** for OpenBeni are starting guesses; calibrate them from the pretrained `policy.pt` evaluation (OpenBeni plan Phase E step 1).
