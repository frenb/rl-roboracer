# Training internals

How the trainer evaluates, regularizes, schedules difficulty, detects stuck cars, stores demonstrations and survives pause and resume.

## Greedy evaluation policy (mode of the SAC distribution)

Collection and evaluation use **different** policies:

- **Collect** uses the stochastic policy (`tf_agent.collect_policy`) — it
  *samples* from the actor's distribution so exploration noise drives the car
  into new states. Required for learning.
- **Eval** uses a **greedy wrapper** around the same actor network, so it takes
  the distribution's deterministic **mode** instead of a sample:

```1869:1871:rl_agent/robotaxi.py
    tf_eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)
```

**Why.** tf-agents' `SacAgent` sets *both* `policy` and `collect_policy` to the
**same** stochastic `ActorPolicy` (a tanh-squashed Gaussian). Evaluating with
that unwrapped policy would sample actions with full exploration noise, so a
single unlucky sampled steering value could crash the car mid-corner — making
`AverageReturn` / episode-length carry the policy's exploration *variance*
rather than its learned *competence* (this showed up as spuriously short eval
episodes). Wrapping it in `GreedyPolicy` removes that noise from the
measurement.

**What "greedy" actually computes.** `greedy_policy.GreedyPolicy` doesn't add a
separate argmax head — it replaces each action distribution the wrapped policy
returns with a deterministic distribution located at that distribution's
**mode**, so `policy.action()` returns `distribution.mode()` instead of
`distribution.sample()`. For SAC the action distribution is a
`TransformedDistribution`: a Gaussian `Normal(μ, σ)` (μ, σ from the actor
network) pushed through a `Tanh` bijector (then scaled/shifted to the action
bounds). tf-agents evaluates the transformed mode as
`bijector.forward(base.mode())`, and a Normal's mode is its mean μ, so the
greedy action is:

```
a_greedy = scale · tanh(μ) + shift        # the "mean action", squashed to bounds
```

i.e. the σ (spread) output is ignored at eval time and only the mean μ of each
action dimension (steering, acceleration) is used. Note this is the
bijector-forward-of-the-base-mode (the standard SAC deterministic eval action),
**not** a Jacobian-corrected mode of the squashed density — the tanh warp
changes the density, but SAC evaluation conventionally uses `tanh(μ)` as the
deterministic action, which is what tf-agents returns here.

**Consequences elsewhere:**

- **Best-model export is greedy too.** When an eval sets a new max return the
  saved policy is `PolicySaver(tf_eval_policy)`, so the deployed artifact takes
  `tanh(μ)` — a deployed car drives deterministically instead of shipping
  exploration noise into production, and it behaves exactly like what the eval
  measured.
- **Eval `goals/ep` reflects greedy driving.** The `eval/…` goal-count scalars
  (see the eval-vs-collect metric split) and the curriculum's top-3 gate metric
  are all computed from these greedy rollouts, so stage advancement is gated on
  deterministic competence, not sampling luck.

## AWAC actor regularization (advantage-weighted BC)

`AwacSacAgent` (`rl_agent/awac_sac_agent.py`) is a drop-in subclass of tf-agents'
`SacAgent` that adds an **advantage-weighted imitation term to the actor loss**,
computed on a *separate* batch sampled from the protected expert-demo replay
table every gradient step. Where plain BC-pretrain-then-SAC let the policy
**drift off** the demonstrations (the best checkpoint was the BC baseline; SAC
then degraded it), AWAC keeps shaping the **policy** directly throughout
training — so the actor inherits the expert's survival / goal-chaining without
being permanently chained to the expert's *slowness*.

**Fully opt-in / inert by default.** The AWAC term only fires when **both**
`awac_lambda > 0` **and** a demo iterator has been attached via
`set_demo_iter(...)`. Otherwise `actor_loss` is bit-identical to the base
`SacAgent`, so the default training path is unchanged.

**What the term computes** (`actor_loss`, `awac_sac_agent.py:97`):

```
A(s, a_demo) = Q_min(s, a_demo) − Q_min(s, a_π)      # advantage; V(s) ≈ Q(s, a~π)
w            = min(exp(A / awac_beta), awac_weight_clip)   # stop-grad AWAC weight
awac_bc      = mean( w · ‖a_π − clip(a_demo)‖² )     # advantage-weighted MSE
actor_loss   = sac_loss + λ(t) · awac_bc
```

- `Q_min` is SAC's pessimistic min-of-twin-critics (`_critic_min`), matching how
  SAC values actions. `A` is clipped to `[−10, 10]` before the exponent.
- The advantage weight `w` means the policy **only imitates expert actions the
  critic rates better than its own current action** (`A > 0`). Expert actions
  the policy already beats get `w ≈ 0`, so AWAC adds survival competence without
  dragging the policy back to expert speed.
- **MSE, not tanh-squashed `log_prob`.** A log-prob imitation loss blows up to
  `inf`/`nan` when the expert action sits at the action-space bounds (steering
  `±1`, accel near the `0.1` floor) — this was the source of an iteration-1 NaN.
  MSE toward the reparameterized policy sample `a_π` is bounded with well-behaved
  gradients and still pulls the policy toward the (advantage-weighted) expert
  action (TD3+BC-style).

**λ schedule** (`_current_lambda`): constant `awac_lambda`, or — when
`awac_lambda_decay_steps > 0` — a linear decay from `awac_lambda` toward
`awac_lambda_min` over that many train steps (strong early survival shaping,
then hand off to RL to refine speed).

**Why the demo iterator lives in a module-level registry** (`_DEMO_ITERS`, keyed
by `id(agent)`): a `tf.data` iterator is a `Trackable`, so as an agent attribute
the Learner's checkpointer would try to serialize it and raise
`UnimplementedError (Op:SerializeIterator)`. Keeping it off the agent's
Trackable graph keeps checkpointing working while the iterator stays usable
in-graph.

**Wiring** (`rl_agent/robotaxi.py::main`):

1. The knobs are experiment-design fields (`rl_agent/experiment_designs.py`
   `SCHEMA`): `awac_lambda`, `awac_beta`, `awac_weight_clip`,
   `awac_lambda_decay_steps` (→ kwargs `awac_lambda_val`, …). All default to the
   plain-SAC values (`awac_lambda_val=0.0`).
2. `main()` constructs `AwacSacAgent(...)` with those knobs and `demo_iter=None`
   when `awac_lambda_val > 0` (the demo table doesn't exist yet).
3. After the demo replay table is built, a demo-only `tf.data` iterator is
   attached via `tf_agent.set_demo_iter(iter(_demo_only_ds))` **before** the
   Learner first runs (so `actor_loss` is traced with the iterator live).

## Curriculum training (performance-gated track difficulty)

`CurriculumScheduler` (`rl_agent/robotaxi.py:974`) walks a TRAIN job through a
sequence of **track-geometry stages** of increasing difficulty, advancing only
when the policy demonstrates competence at the current stage. It exists because
starting a fresh policy on the hardest geometry (tight corners + chicanes) rarely
learns — the curriculum lets it master easy corners first, then ratchets up.

**Stage list.** `curriculum_stages` is an experiment-design field
(`experiment_designs.py` `SCHEMA`, kwarg `curriculum_stages_val`): a JSON list of
stage dicts. Each stage carries the track knobs plus its advancement gate:

- `corner_radius` (float, m) — turn tightness; smaller = harder.
- `chicanes_north` / `_east` / `_south` / `_west` (int) — absolute per-edge
  chicane counts (the real difficulty axis).
- `curvature_difficulty` (float) — **DEPRECATED**, logging / back-compat only;
  no longer drives chicane count.
- `advance_goals` (float) — goals/ep threshold to advance off this stage.
- `consecutive` (int, default 1) — how many *consecutive* eval cycles must clear
  the threshold before advancing (variance insurance).

The **last stage omits `advance_goals`** — it's terminal, so training continues
on the hardest geometry indefinitely.

**How advancement is gated.** At each eval cycle the training loop calls
`_curriculum.update(_eval_topk_metric, step)` (`robotaxi.py:3019`). The gate
metric is deliberately **not** the collect env's noisy running mean — it's the
**top-3-by-goal-count mean of ONLY the most recent greedy eval**
(`_EVAL_TOPK = 3`, `robotaxi.py:2989`):

- **Greedy eval** signal reflects learned competence, not collect-time
  exploration variance (see the greedy-policy section above).
- **Only the latest eval** → the gate tracks the *current* policy, not a window
  lagged by earlier weaker evals.
- **Top-3** rewards demonstrated best-case driving and is robust to a couple of
  unlucky early crashes; per-eval noise is absorbed by the stage's `consecutive`
  requirement.

`update()` increments a consecutive-clears counter when the metric meets
`advance_goals` (resetting it on any miss), and advances one stage once the
counter reaches `consecutive`.

**Applying a stage** (`_apply_stage`) forwards the stage geometry to Unity's
`TrackGenerator` via `configure_env(...)` on the collect env **and**, in
multi-actor runs, the dedicated single-gym `eval_env` — so eval always measures
the *current* stage's geometry rather than whatever stage the eval env was built
at. It also logs `curriculum/stage`, `curriculum/corner_radius`,
`curriculum/curvature_difficulty`, and `curriculum/chicanes_total` scalars to
TensorBoard.

**Starting higher up the ladder.** `curriculum_start_stage` (kwarg
`_curriculum_start`) pins the initial stage index (clamped into range). Setting
it to `len(stages) − 1` trains directly on the final/hardest geometry — since the
terminal stage has no `advance_goals`, the scheduler is then a no-op and the run
stays there. `main()` reads the start-stage geometry to configure the env before
the first eval, then hands the stages to the scheduler.

Notes:

- The gym itself has **no** internal stage progression — it renders whatever
  geometry the reset message carries. All curriculum progression is
  **trainer-driven**; the Unity `CurriculumStageButtons` picker (toggle `C`) is a
  manual, trainer-less convenience only.
- A standalone EVAL job does not run the scheduler; to evaluate across stages use
  the Models-tab **Curriculum sweep** (Start/End stage) — see
  `_eval_over_curriculum` and the eval modal.

## Stuck / wedged-car detection

Episode termination is decided **in Python** (`donut_course.has_failed` →
`base_course.check_if_moving`); Unity only resets when the trainer sends a
RESTART. `check_if_moving` inspects the car's per-step world positions
(`_position_history`, one `[x, y, z]` in metres per env step) over a trailing
window and decides whether the car is still making progress.

**Net-displacement test (2026-08-09).** The old logic returned "moving" the
moment the current position differed from *any* sample in the window by
≥ 0.0001 m (0.1 mm) — i.e. it required a car to be *perfectly frozen* to be
flagged stuck. A car high-centered / wedged against geometry with the policy
still applying throttle (spinning wheels) and sweeping the steering
micro-oscillates far more than 0.1 mm, so it always cleared that bar and was
never reset; it burned steps until the 10 000-step cap (`has_too_many_steps`),
dumping thousands of near-zero-reward wedged transitions into the buffer
(observed on the `w-course` hairpin neck). The check now measures how far the
car has travelled from its **current** spot across the whole window: a wedge
stays inside a small ball (jitter / steer-sweep don't move the chassis), while
a slow-but-progressing car escapes the radius. This is immune to
steering/wheel-spin jitter.

Two env-tunable knobs (defaults preserve the historical 200-step window):

- `ROBOTAXI_STUCK_WINDOW` (default `200`) — trailing per-step positions to
  inspect. The car is never flagged stuck until this many steps have elapsed,
  giving a legitimately slow car (e.g. easing through a chicane) a multi-second
  runway to prove progress.
- `ROBOTAXI_STUCK_RADIUS_M` (default `0.5`) — minimum net travel (metres) from
  the current position over the window to count as "moving". Below it the car
  is treated as wedged → `is_stuck` → `has_failed` → RESTART. Raise (e.g.
  `1.0`) to reset sooner; lower (e.g. `0.25`) to be more lenient.

Notes:

- Applies to **all** courses (`donut`, `donut_no_hint`, `w-course`, `simple`);
  once active it will raise `is_stuck` / `crashes_total` counts.
- It's a Python-only change — no Unity rebuild — but the trainer must be
  restarted to load it; a running job keeps the old behaviour until then.

## The TFRecord corpus: layout, `TFRECORDS_DIR`, and housekeeping

Expert demonstrations are stored as one directory per DEMO job —
`job_<id>/`, holding one or more `.tfrecord` files. Every consumer (BC
pretraining, the AWAC actor regularizer, `analyze_demo_metrics.py`) reads
this same tree.

### Host path vs. container path — the distinction that bites

One folder, two names, and conflating them breaks the trainer:

| | Path | Who uses it |
| --- | --- | --- |
| Host (Windows) | `LATEST\tfrecords\` | Compose, as the bind-mount **source** |
| Container | `/tfrecords` | **every line of Python** |

`sim-controller` is the only service that mounts the volume — the
dashboard and Mad Scientist never touch it, they pass job ids around
instead.

The Python side is hardcoded to the **container** path on purpose:
`robotaxi.py` builds `/tfrecords/job_<id>` strings for both the write side
(`collect_expert_demos`) and the read side (`demo_record_dirs`), and
`analyze_demo_metrics.py` resolves a bare job id against `/tfrecords`.
Because the mount **target** is fixed, those literals always resolve no
matter where the host folder actually sits. Never rewrite them to a host
path such as `C:\...\tfrecords` — there is no `C:` drive inside the
container, and every demo-backed job would fail to find its corpus.

### Relocating the corpus with `TFRECORDS_DIR`

The host side *is* configurable. Compose reads:

```yaml
      - type: bind
        source: ${TFRECORDS_DIR:-../tfrecords}
        target: /tfrecords
```

so with the variable unset the corpus stays a sibling of the repo (the
[Setup](setup.md#3-create-sibling-data-directories) default) and nothing changes.
To keep it somewhere else — a roomier drive, a shared scratch disk — set
it in `.env`:

```ini
TFRECORDS_DIR=D:/rl-data/tfrecords
```

Three rules for the value: **forward slashes** (backslashes in a `.env`
invite escaping surprises), **no trailing slash** (a trailing separator on
a bind source is a well-known source of Compose grief), and either an
absolute path or one relative to the repo root. Only the host side moves;
the target stays `/tfrecords`, so no Python changes and no re-collection.

Confirm the substitution actually happened before bringing the stack up:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml config |
  Select-String tfrecords -Context 1,1
```

Then apply it with `docker compose ... up -d`. A bare `restart` is **not**
enough — mounts are fixed at container creation, so `sim-controller` has
to be recreated to pick up a new source.

### Filenames inside a job directory

`read_files_from_directory()` reads *every* file in the job dir and
concatenates them, so three naming generations coexist harmlessly:

| Pattern | Era |
| --- | --- |
| `000000trajectories.tfrecord` | original — one write at end of run |
| `<batch>_<sub>trajectories.tfrecord` | periodic flush (`flush_every_n_steps`) |
| `<batch>_g<gym>_<sub>trajectories.tfrecord` | current — `g<gym>` added for concurrent gyms |

`<batch>` is the curriculum stage (`batch_number`), `<sub>` the flush
counter, `<gym>` the Unity instance index. The concatenate-everything read
is why `collect_expert_demos()` insists callers pass a **distinct**
`batch_number` per stage and bakes `gym_index` into the name: reusing
either would silently clobber a sibling stage's steps rather than adding
to them.

### Related environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `TFRECORDS_DIR` | Compose (host) | Host directory to bind-mount at `/tfrecords`. Defaults to `../tfrecords`. |
| `ROBOTAXI_DEFAULT_DEMO_JOB_ID` | `sim-controller` | Overrides which DEMO job supplies the default expert corpus — step 2 of `_resolve_default_demo_job_id()`, below the job doc's own `default_demo_job_id` and above the per-course `COURSE_DEFAULT_DEMO_JOB_IDS` map. |
| `ROBOTAXI_OBSERVATION_SIZE` | `sim-controller` | Fallback active observation width the demo reader slices down to. Normally irrelevant — `do_job()` sets it per-job via `set_observation_size()` from the course. |

Note the asymmetry: `TFRECORDS_DIR` answers *where the corpus is*, the
other two answer *which records to load and how wide they are*. Only the
first one belongs in `.env`; the other two are container-side overrides
for one-off experiments.

### Housekeeping

The corpus is a **sibling** of the repo, not inside it, so git never sees
it and there is nothing to gitignore — but equally, nothing backs it up.
It grows roughly linearly with DEMO hours (currently ~0.5 GB across 30 job
dirs, the largest ~110 MB), so it is slow-growing rather than urgent.

Moving a corpus to another machine is a plain folder copy: `job_<id>`
directories are self-contained and carry no absolute paths. Deleting one,
however, orphans any job document still referencing that id — a TRAIN job
pointed at a missing source resolves to an empty dataset. Prefer trimming
via `demo_job_ids` / `skip_default_demo` on the job doc over deleting
directories.

## Analyzing DEMO-collection metrics (speed / episodes / crashes)

`rl_agent/analyze_demo_metrics.py` recovers per-run driving statistics for a
**DEMO** (`collect_expert_demos`) job straight from its stored tfrecords.

### Background

A DEMO job drives the car with a scripted heuristic and records every
transition to `/tfrecords/job_<id>/` for later use as an AWAC/BC expert prior.
Unlike TRAIN jobs, a DEMO run:

- does **not** write TensorBoard / course metrics (no `metrics/` event files,
  no `avg_speed` / `crashes_per_1k_steps` / `avg_goals_per_episode` scalars —
  those come from the TRAIN loop's `update_stats`, which DEMO collection never
  enters), and
- only prints per-episode counts to stdout, which lands in `/tmp/trainer.log`
  and is **overwritten by the next trainer launch**.

So once you've moved on to training, the only durable record of how well a demo
run actually drove is the tfrecords themselves. This tool reads them back.

### Rationale — measured vs. reconstructed

- **Speed is measured directly.** `observation[1]` is the signed car speed in
  m/s (see `DonutCourse.observation_spec`: `obs[0]`=angle-to-goal,
  `obs[1]`=speed, `obs[17]`=forward raycast clearance). The tool reports
  per-stage and overall mean/median/percentiles plus `%reverse`, `%stopped`,
  and `%cruise`.
- **Episodes / goals-per-episode / crashes are reconstructed** (estimates, not
  ground truth), because the tfrecords store single-step transitions with
  synthetic step-types — episode boundaries aren't persisted. Rows are in
  temporal order within each stage file, so:
  - an **episode boundary** is a respawn: speed dips below `--speed-stop` after
    having driven above `--speed-moving` (with `min_force >= 0.2` the demo car
    only sits at ~0 m/s on a fresh spawn);
  - **goals/episode** is inferred from the per-stage goal budget ÷ reconstructed
    episode count — the collector advances a stage after `--goal-budget` goals
    and truncates each episode at `--goal-cap` goals, so a crash-free stage
    lands on exactly `ceil(goal_budget / goal_cap)` episodes (e.g. 4 for
    100/30). Landing on that minimum is itself the signal that the run was
    essentially crash-free;
  - a **crash-ended episode** is flagged when the forward clearance at the
    episode's last step is in the bottom decile (car against a wall) rather than
    a clean cap/budget cutoff on open road.

### Usage

Runs inside the `sim-controller` container (it imports `collect_training_data`
and reads the `/tfrecords` volume):

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec -T sim-controller `
  bash -lc "cd /python_ws/src && PYTHONPATH=/python_ws/src python -u analyze_demo_metrics.py <JOB_ID>"
```

The positional argument accepts a bare job id, `job_<id>`, or a full
`/tfrecords/...` path:

```powershell
# by job id
... python -u analyze_demo_metrics.py 6a63d44dc9d3ac4a1af57a0a

# override the collection's budget/cap or the respawn speed thresholds
... python -u analyze_demo_metrics.py 6a63d44d --goal-budget 100 --goal-cap 30 --speed-stop 0.3
```

Output is an `OVERALL` block followed by one block per curriculum stage (the
leading zero-padded number in each `<stage>_g<gym>_<subbatch>trajectories.tfrecord`
filename). Keep `--goal-budget` / `--goal-cap` in sync with
`robotaxi.GOALS_PER_STAGE` and `donut_course.GOALS_PER_EPISODE_CAP` if those
change, so the goals-per-episode estimate stays accurate.

## Testing Reverb buffer save/restore on pause-resume

On pause the trainer now saves **both** the Learner checkpoint (actor + critic
+ optimizer state + `train_step`) **and** a Reverb replay-buffer snapshot (and
it also writes a periodic Reverb snapshot during training, so an ungraceful
crash recovers too). On resume it restores the buffer and **skips
`initial_collect`**, so the old demo-heavy distributional shift on resume is
gone. The test procedure below validates that this restore actually fires; if
the snapshot is ever missing (Layer 1 = `NOT_FOUND`), resume falls back to the
legacy behaviour — buffer refills from demo prefill + `initial_collect` — which
is what the "Without Reverb restore" rows below describe.

### Layer 1 — File existence (2 minutes)

After pausing a job, verify the Reverb snapshot file was written:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "ls -lh /tmp/active/<job_id>/learner/reverb_checkpoint/ 2>/dev/null || echo NOT_FOUND"
```

**Pass:** file exists and contains a path.
**Fail:** file missing — the save didn't fire (check logs for the `Reverb checkpoint save failed` warning).

### Layer 2 — Buffer size on first TRAIN line after resume (10 minutes)

Compare `buffer_size=X/300000` on the very first `TRAIN end:` line after the
job is resumed:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "grep 'TRAIN end' /tmp/trainer.log | head -3"
```

| Code version | Expected first buffer_size |
|---|---|
| Without Reverb restore | ~300 (demo prefill warmup only) |
| With Reverb restore | ≈ pre-pause value (e.g. 250,000) |

Also check the trainer log for these lines in order:

```
main: Reverb table restored from /tmp/active/.../learner/reverb_checkpoint
main: Skipping initial_collect — Reverb buffer already warm (size=250000)
TRAIN begin: iter=X/...
```

**Pass:** buffer_size on first TRAIN ≈ pre-pause value, `initial_collect` skipped.
**Fail:** buffer_size starts at ~300 and initial_collect runs — restore didn't fire.

### Layer 3 — Return trajectory continuity (1–2 hours)

Inspect `eval_curve.csv` around the pause step:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "cat /tmp/active/<job_id>/eval_curve.csv"
```

**Pass:** no visible dip in `avg_return` immediately after the resume step — the
policy continues smoothly from its pre-pause level.
**Fail (old behaviour):** a dip of 10–30% in the eval return for 2–3 eval cycles
after resume, caused by the critic adapting to the temporarily demo-heavy buffer.

### Recommended test sequence

1. Start any queued training job.
2. Wait for ~5,000 iterations (`buffer_size` should be 50,000+).
3. Click **Pause** on the Jobs tab.
4. Run the Layer 1 check — confirm the snapshot file was written.
5. Set the job back to `NOT_STARTED` (click **Resume** in the Jobs tab).
6. Check the first `TRAIN end:` line (Layer 2) — confirm `buffer_size` ≈ pre-pause value.
7. Let it reach the next eval cycle and inspect `eval_curve.csv` (Layer 3) for a smooth return trajectory.

Total wall time: ~15–20 minutes once the implementation is in place.

### Monitoring-counter continuity across pause/resume

A pause/resume restores training state (actor/critic/optimizer + `train_step`
via the Learner checkpoint, and the Reverb buffer via its snapshot) but the
**cumulative monitoring counters** live only in memory and are rebuilt at 0 by
the fresh process. Because TensorBoard plots them against the restored
`train_step` (~170k), they used to **sawtooth back to 0** at the resume step
even though no training progress was lost. Two families were affected:

- tf-agents `py_metrics` on the collect actor: `Metrics/EnvironmentSteps`,
  `Metrics/NumberOfEpisodes`.
- Course lifetime stats surfaced from the collect env: `avg/max` goals-per-
  episode, `avg/max` speed + accel, `avg_steering_angle_ratio`,
  `crashes_per_1k_steps`, `avg_steps_per_episode`, and the per-location
  traversal/crash counts.

**Fix (2026-08-09):** the trainer now snapshots these to
`learner_dir/resume_counters.json` on pause **and** on every periodic Reverb
checkpoint (so an ungraceful crash also benefits), and re-seeds them on resume
so the curves continue instead of resetting. See `_save_resume_counters` /
`_seed_resume_counters` / `restore_course_counters` in `robotaxi.py`,
`RobotaxiEnv.restore_course_cumulative`, and
`BaseCourse.restore_cumulative_counters`.

Details / limitations:

- The course SUM-type totals (`steps_total`, `goals_per_episode_total`, …) are
  **split evenly across the current actor count** before restore, so both the
  summed aggregate (`read_course_raw_counters`) and every mean-aggregated ratio
  derived from them stay continuous; the already-aggregated mean/max fields
  (`crashes_total`, per-location counts, `max_*`) are passed through unchanged.
- The rolling **`*_last_30`** metrics and their backing arrays are deliberately
  **not** restored — they re-stabilize within ~30 episodes.
- Only the **collect** env's course is re-seeded; the separate greedy **eval**
  env's `eval/…` course counters still restart at 0 (eval is periodic and
  those re-stabilize quickly).
- Effective from the **next** resume after the code is in place: the snapshot
  is written by a *running* trainer that has the feature, so the first resume
  onto it finds no file and still starts at 0 (one last sawtooth); every resume
  after that is continuous. Keep `--num-envs` constant across a resume for the
  split to stay exactly proportional.
- Windowed metrics `AverageReturn` / `AverageEpisodeLength` are last-N buffers
  (not lifetime totals), so they simply refill and are intentionally not
  seeded. `max_avg_return` (best-model gate) is separately restored from Mongo.
