# rl-roboracer

RL training stack for the robotaxi Unity gym.

## Layout

```
rl-roboracer/
├── docker-compose.yml         # the whole stack
├── docker/
│   ├── ros_server/            # ROS noetic + gRPC bridge build context
│   └── sim_controller/        # CUDA + tf-agents training image build context
├── rl_agent/                  # tf-agents Python code that drives training
├── dashboard/                 # Node/Express monitoring dashboard (port 80)
├── protos/                    # source of truth for the gRPC contract
└── scripts/                   # PowerShell helpers
```

The repo expects to live next to a few sibling data folders (created in
Setup below). Trainer `/tmp` is a Compose named volume (`tmpdata`), not a
host bind mount. The Unity gym binary lives in-repo at
`unity\Builds\latest\`, not as a sibling.

```
LATEST/
├── rl-roboracer/      # this repo
├── saved_models/      # tf-agents checkpoints   (bind-mounted)
├── mongodb/           # mongo data dir          (bind-mounted)
└── tfrecords/         # demonstration trajectories (bind-mounted)
```

## Setup

Do these once, in order, on a Windows host with an NVIDIA GPU. Scripts
are PowerShell.

### 1. Install host software

1. **Git.**
2. **[NVIDIA Game Ready or Studio driver](https://www.nvidia.com/Download/index.aspx)**
   new enough for WSL 2 CUDA (required by `sim-controller`;
   [R495 or later](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   on the Windows host — do not install a Linux NVIDIA driver inside WSL).
3. **[Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)**
   with the **WSL 2** engine. There is no separate “enable GPU”
   checkbox — [GPU-PV](https://docs.docker.com/desktop/features/gpu/)
   is on when the NVIDIA host driver is current, WSL is updated
   (`wsl --update`), and Settings → **General** has **Use the WSL 2
   based engine** checked (default on a WSL 2 machine; the row is
   hidden if it is the only engine). Settings → **Resources → WSL
   Integration** is only needed if you will run `docker` from a Linux
   distro. Compose reserves one NVIDIA device for `sim-controller`.
   Confirm with `docker version`, `docker compose version`, and:

   ```powershell
   docker run --rm --gpus all nvidia/cuda:11.0.3-runtime-ubuntu20.04 nvidia-smi
   ```
4. **[Unity Hub](https://unity.com/download)**, then install **Unity
   Editor 2020.3.11f1** (changeset `99c7afb366b3`). That is the version
   in `unity/ProjectSettings/ProjectVersion.txt`; a newer 2020.3 patch
   will rewrite the project. In Hub → Installs → the editor's modules,
   add **Windows Build Support (Mono)**. Hub itself is unpinned.

### 2. Clone the repo

Parent directory name can be anything; compose bind-mounts
`../saved_models`, `../mongodb`, and `../tfrecords` relative to the
repo, so those three folders must sit next to `rl-roboracer`.

```powershell
cd <parent>   # e.g. Documents\agents\robots\LATEST
git clone https://github.com/frenb/rl-roboracer.git
cd rl-roboracer
```

### 3. Create sibling data directories

```powershell
New-Item -ItemType Directory -Force `
  ..\saved_models, ..\mongodb, ..\tfrecords | Out-Null
```

Empty dirs are enough to start. Copy demo tfrecords or checkpoints into
those folders when you have them. Optional: `Copy-Item .env.example .env`
if you will enable the Mad Scientist (off by default).

### 4. Build the Docker images

First-time `docker compose build` compiles the three local images and
tags them with the names compose already uses (pulling the `FROM`
bases as it goes). From the repo root:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml build
```

| Service | Compose image | Source |
|---|---|---|
| `ros-server` (+ `ros-server-1..3`) | `docker_ros-server:thin` | Built from `docker/ros_server/Dockerfile` (`FROM ros:noetic-ros-base-focal`) |
| `sim-controller` | `sim_controller:latest` | Built from `docker/sim_controller/Dockerfile` (`FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04`; pip installs `tensorflow==2.7.0rc1`, `tf-agents[reverb]==0.11.0rc0`) |
| `madscientist` | `madscientist:latest` | Built from `docker/madscientist/Dockerfile` (`FROM python:3.11-slim`) |
| `dashboard` | `node:20` | Pulled; no local Dockerfile |
| `mongo` | `bitnami/mongodb:6.0` | Pulled |
| `mongo-express` | `mongo-express` | Pulled (tag unpinned) |

`Start-Stack.ps1` later pulls any remaining public images (`mongo`,
`mongo-express`, `node:20`) on first `up`.

The compose `image:` name `docker_ros-server:thin` is the tag applied to
the full `Dockerfile` build. Incremental rebuilds (edit ROS packages
without reinstalling apt) are optional: tag a `Dockerfile` build as
`docker_ros-server:working`, then
`docker build -f docker/ros_server/DockerfileThin -t docker_ros-server:thin docker/ros_server`.

### 5. Build and promote the Unity gym

1. Unity Hub → **Add** → open the `unity\` folder of this repo with
   Editor **2020.3.11f1**.
2. Confirm **File → Build Settings** has `Assets/Scenes/w-course-jetracer.unity`
   enabled (it is the play scene in `EditorBuildSettings`).
3. **File → Build Settings → Build**, platform **Windows**, into a
   dated subfolder of `unity\Builds\` (e.g.
   `unity\Builds\2026.09.05-jetracer\`). Do not build straight into
   `latest\`.
4. Promote that folder so the launchers can find it:

```powershell
.\scripts\PromoteLatestBuild.ps1
```

This archives any previous `unity\Builds\latest\` and renames the new
dated folder to `latest\`. The launchers always read from `latest\` and
copy it into per-instance directories (`unity\Builds\instances\0..N-1\`)
so Unity's "force single instance" mutex does not block multi-actor
runs.

After this, use **Running an experiment** for day-to-day start/stop.

## Running an experiment

The default workflow runs **4 parallel Unity clients** all feeding one shared
SAC learner. Three PowerShell scripts in `scripts\` handle the full lifecycle:

| Script | Purpose |
|---|---|
| `scripts\Start-Stack.ps1` | Bring the docker stack + N Unity clients up |
| `scripts\Stop-Stack.ps1`  | Kill Unity clients, their supervisor windows, and tear the docker stack down |
| `scripts\Restart-Stack.ps1` | Stop-Stack followed by Start-Stack |

All three accept `-N <int>` to override the default of 4 actors, plus
`-StaggerSeconds`, `-Popup`, `-SkipUnity`, and `-WaitForRosServersSeconds` —
see each script's help block for details.

### Bring everything up

```powershell
.\scripts\Start-Stack.ps1
```

This will:

1. `docker compose -f docker-compose.yml -f compose/scale.yml up -d` — starts
   the base services plus the `ros-server-{1..3}` overlay services.
2. Wait ~8s for ros-servers' internal `start.sh` to finish (rosmaster +
   ROS-TCP listener), so the first Unity handshake doesn't see a closed port.
3. Sync `unity\Builds\latest\` into 4 per-instance copies via robocopy and
   spawn 4 supervised Unity clients (each in its own PowerShell window).

Each client gets a unique `--ros-port` (10000+i) and `--unity-port` (5005+i)
so the bidirectional ROS-TCP-Connector protocol routes correctly per actor.

Once all four Unity windows are up, kick off training:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  bash -c 'cd /python_ws/src && python -u robotaxi.py --num-envs 4 2>&1 | tee robotaxi.out'
```

Two important pieces in that command:

- `python -u` forces unbuffered stdout. When Python detects that stdout
  is a pipe (which it is when piped to `tee`), it switches from
  line-buffered to ~8 KB block-buffered, holding back lines until the
  buffer fills. With `-u`, every `[actor-N]` line lands in `robotaxi.out`
  (and the dashboard's live log view) the moment it's emitted.
- `| tee robotaxi.out` feeds the dashboard's live log panel —
  `dashboard/src/server.ts` tails `/python_ws/src/robotaxi.out` over a
  WebSocket. The `compose/scale.yml` overlay disables sim-controller's
  default auto-run of the single-env trainer (so it doesn't compete with
  your multi-env exec for MongoDB jobs), which means without the `tee`
  the file stays stale and the dashboard panel shows old data.

TensorBoard at `http://localhost:6006/` will show one run with `metrics/`,
`eval/`, `train/`, and `learner/train/` summaries. The dashboard at
`http://localhost:80/` browses past runs (archived to `/tmp/jobsdata/` by
the new TRAIN job's startup cleanup).

### Bring everything down

```powershell
.\scripts\Stop-Stack.ps1
```

Order: supervisors die first (so they stop respawning Unity), then Unity
clients, then `docker compose down`. Each step is best-effort — re-running
on an already-clean state is a no-op, not an error. The running Python
training (whether the default one in the container's `command:` or one
started via `docker compose exec`) is killed automatically when its
container is stopped.

### Restart the stack

```powershell
.\scripts\Restart-Stack.ps1
```

Useful after edits that need a fresh container state (changes to
`docker-compose.yml`, network config, or container `command:` lines). For
pure `rl_agent/` Python edits, the bind mount picks them up live — no
restart needed, just re-run `python robotaxi.py --num-envs 4`.

### Trajectory rollout viewer (2 actors)

The policy candidate-path fan (`TrajectoryRolloutViz` in Unity, **T** to
toggle) stays empty unless the trainer is started with
`ROLLOUT_VIZ_ENABLED=1`. The compose `command:` never sets that flag, so a
container auto-start or a `pkill` that lets sim-controller respawn
`robotaxi.py` will not show trajectories.

Use the scale overlay (no auto-trainer), two Unity clients, then exec the
trainer yourself:

```powershell
.\scripts\Restart-Stack.ps1 -N 2
```

When both Unity windows are up:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  bash -c 'export ROLLOUT_VIZ_ENABLED=1 && cd /python_ws/src && python -u robotaxi.py --num-envs 2 2>&1 | tee /tmp/trainer.log'
```

Match `--num-envs` to `-N`. Start a TRAIN or EVAL job from the dashboard.
Watch actor 0 (or both; `ROLLOUT_VIZ_ALL_ACTORS` defaults on) from the
**top-down Main Camera**. Press **T** if the fan is hidden.

EVAL uses the greedy SavedModel, so the lines almost overlap — that still
means the viz is working. TRAIN shows the real spread.

If the stack is already up with scale.yml and no trainer, skip
`Restart-Stack` and run only the `export ROLLOUT_VIZ_ENABLED=1` command.
Do not run a second `robotaxi.py` on top of an existing one. More knobs:
`docs/trajectory-rollout-viz.md`.

### Unity-side-only lifecycle

For the common case of "I edited `RunClientWrapper.ps1` / re-promoted a
Unity build / want to re-grid the windows" without churning Docker
(which would lose sim-controller's warm reverb buffer + MongoDB state),
there's a parallel set of scripts that touch only the Unity side:

| Script | Purpose |
|---|---|
| `scripts\Start-Clients.ps1`  | Launch N Unity clients + supervisor tabs (assumes Docker is up) |
| `scripts\Stop-Clients.ps1`   | Kill Unity + supervisors only (equivalent to `Stop-Stack.ps1 -KeepDocker`) |
| `scripts\Restart-Clients.ps1`| Stop-Clients followed by Start-Clients |

Same parameters as the `*-Stack` versions (`-N`, `-StaggerSeconds`,
`-Popup`, `-GridCols`, `-GridRows`, `-Minimized`, `-UseWindowsTerminal`).

### Common variations

```powershell
# Single-actor smoke test (one Unity client, no parallelism). When the
# scale.yml overlay is loaded sim-controller's auto-run is disabled, so
# even with -N 1 you start the trainer manually:
.\scripts\Start-Stack.ps1 -N 1
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  bash -c 'cd /python_ws/src && python -u robotaxi.py 2>&1 | tee robotaxi.out'

# Tile small popup windows for quick visual inspection of multi-actor runs
.\scripts\Start-Stack.ps1 -Popup

# Bring docker up but skip the Unity launch (e.g. to debug a single instance)
.\scripts\Start-Stack.ps1 -SkipUnity

# Recycle just the Unity clients without touching docker (keeps reverb /
# MongoDB / tensorboard state warm)
.\scripts\Restart-Clients.ps1

# Custom 4-wide single-row layout for ultrawide monitors
.\scripts\Start-Clients.ps1 -N 4 -GridCols 4 -GridRows 1

# Foreground supervisors (not minimized) for actively watching debug output
.\scripts\Start-Clients.ps1 -Minimized:$false

# Force separate-windows fallback (skip Windows Terminal even if installed)
.\scripts\Start-Clients.ps1 -UseWindowsTerminal:$false
```

### Service map

| Service          | Port              | What it does                                              |
|------------------|-------------------|-----------------------------------------------------------|
| `ros-server`     | 10000 / 50051     | Actor 0: ROS-TCP socket / gRPC virtual endpoint           |
| `ros-server-1`   | 10001 / 50052     | Actor 1: same, scale-overlay                              |
| `ros-server-2`   | 10002 / 50053     | Actor 2: same, scale-overlay                              |
| `ros-server-3`   | 10003 / 50054     | Actor 3: same, scale-overlay                              |
| `mongo`          | 27017             | Job / model / leaderboard storage                         |
| `mongo-express`  | 8081              | Mongo admin UI                                            |
| `sim-controller` | 6006              | Tensorboard for the live training run                     |
| `dashboard`      | 80                | Golden Layout UI (iframes Tensorboard, logs, jobs, models) |
| `dashboard`      | 8080              | WebSocket tail of `rl_agent/robotaxi.out`                 |

## Rebuilding the gRPC stubs

If you change `protos/virtual_endpoint/proto/ros_service.proto`:

```powershell
pip install grpcio-tools
.\scripts\gen_protos.ps1
```

(Or `./scripts/gen_protos.sh` on bash.)

## Notes

- First-time image build is **Setup §4**. Incremental `ros-server` rebuilds
  via `DockerfileThin` (layer on `docker_ros-server:working`) are optional.
- MongoDB data, saved models, and tfrecords live as siblings of this repo
  so they can be regenerated, swapped, or wiped without touching git
  history. The gym binary is `unity\Builds\latest\`. Trainer `/tmp` is
  the Compose volume `tmpdata`.

---

## Developer notes

### Testing Reverb buffer save/restore on pause-resume

On pause the trainer now saves **both** the Learner checkpoint (actor + critic
+ optimizer state + `train_step`) **and** a Reverb replay-buffer snapshot (and
it also writes a periodic Reverb snapshot during training, so an ungraceful
crash recovers too). On resume it restores the buffer and **skips
`initial_collect`**, so the old demo-heavy distributional shift on resume is
gone. The test procedure below validates that this restore actually fires; if
the snapshot is ever missing (Layer 1 = `NOT_FOUND`), resume falls back to the
legacy behaviour — buffer refills from demo prefill + `initial_collect` — which
is what the "Without Reverb restore" rows below describe.

#### Layer 1 — File existence (2 minutes)

After pausing a job, verify the Reverb snapshot file was written:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "ls -lh /tmp/active/<job_id>/learner/reverb_checkpoint/ 2>/dev/null || echo NOT_FOUND"
```

**Pass:** file exists and contains a path.
**Fail:** file missing — the save didn't fire (check logs for the `Reverb checkpoint save failed` warning).

#### Layer 2 — Buffer size on first TRAIN line after resume (10 minutes)

Compare `buffer_size=X/300000` on the very first `TRAIN end:` line after the
job is resumed:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "grep 'TRAIN end' /python_ws/src/robotaxi.out | head -3"
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

#### Layer 3 — Return trajectory continuity (1–2 hours)

Inspect `eval_curve.csv` around the pause step:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c "cat /tmp/active/<job_id>/eval_curve.csv"
```

**Pass:** no visible dip in `avg_return` immediately after the resume step — the
policy continues smoothly from its pre-pause level.
**Fail (old behaviour):** a dip of 10–30% in the eval return for 2–3 eval cycles
after resume, caused by the critic adapting to the temporarily demo-heavy buffer.

#### Recommended test sequence

1. Start any queued training job.
2. Wait for ~5,000 iterations (`buffer_size` should be 50,000+).
3. Click **Pause** on the Jobs tab.
4. Run the Layer 1 check — confirm the snapshot file was written.
5. Set the job back to `NOT_STARTED` (click **Resume** in the Jobs tab).
6. Check the first `TRAIN end:` line (Layer 2) — confirm `buffer_size` ≈ pre-pause value.
7. Let it reach the next eval cycle and inspect `eval_curve.csv` (Layer 3) for a smooth return trajectory.

Total wall time: ~15–20 minutes once the implementation is in place.

#### Monitoring-counter continuity across pause/resume

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

### Analyzing DEMO-collection metrics (speed / episodes / crashes)

`rl_agent/analyze_demo_metrics.py` recovers per-run driving statistics for a
**DEMO** (`collect_expert_demos`) job straight from its stored tfrecords.

#### Background

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

#### Rationale — measured vs. reconstructed

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

#### Usage

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

### The `donut_no_hint` course (hint-free observation for sim→real)

`donut_no_hint` is a variant of the classic `donut` course whose observation
**drops the leading `dist_from_traj` element** (`obs[0]`, the signed
angle-to-next-goal), shrinking the observation from **32 → 31** dims.
Everything else — car dynamics, goals, raycasts, the expert controller, and the
sideslip angle (`goal_2`) — is identical.

**Why:** `dist_from_traj` is derived from the Unity course's invisible goal
objects (`SceneDataPublisher.GetAngleToGoal()`), a hint that will **not** exist
when the learned policy is transferred to a real *JetRacer ROS AI Robot*.
Training without it forces the policy to infer heading from the raycasts it
*will* have on the real car. `goal_2` (sideslip angle, forward-vector vs.
velocity-vector) is kept — that's a physical quantity derivable on the robot.

Implementation: `rl_agent/environments/courses/donut_course_no_hint.py`
subclasses `DonutCourse` and slices column 0 off the observation spec,
`get_empty_state()`, and `scene_data_array()`. It's registered in
`environments/courses/__init__.py` and instantiated by `robotaxi_env.py` when
`course_type == 'donut_no_hint'`.

#### Selecting the course per job

The dashboard **New-job** form has a **Course** selector (`trainer default`,
`donut`, `donut_no_hint`). The choice is stored as `course_type` on the job
document and threaded through the whole pipeline:

- `do_job()` resolves `_job_course_type` (job doc → `ROBOTAXI_COURSE_TYPE`
  env → `donut`), then points the demo/BC pipeline at the matching width via
  `collect_training_data.set_observation_size(...)` and passes
  `course_type=...` to every `make_env()` (DEMO/EVAL) and to `main()` (TRAIN).
- `main()` re-resolves the course (so direct callers work too) and builds the
  collect/eval envs and the actor/critic network at the right width.
- The Jobs table shows a **Course** column (legacy jobs are backfilled to
  `donut` on server startup); model records are stamped with `course_type`, and
  the Models tab's **Compat** check compares against the model's own course
  width. An EVAL launched from the Models tab inherits the model's course.

`trainer default` means "use the `ROBOTAXI_COURSE_TYPE` env var / `donut`" and
is the fallback for anything unmapped.

#### One 32-wide demo corpus, reused everywhere

There is a **single expert-demo corpus, always recorded at the full 32-wide
observation**. `donut_no_hint` does **not** need its own demos — the read path
drops the leading column at load time. Two halves make this work:

- **Read side** (`collect_training_data.py`): demos are always *parsed* at the
  full recorded width (`FULL_OBSERVATION_SIZE = 32`, see `feature_description`),
  then `convert_tfrecord_to_trajectory()` slices off the leading
  `_OBS_DROP_LEADING` column(s) to reach the **active** width (`observation_size`,
  set per-job via `set_observation_size()` from `COURSE_OBSERVATION_SIZES`:
  `donut`=32 → drop 0, `donut_no_hint`=31 → drop 1). So a 32-wide corpus feeds a
  31-wide job unchanged, with no re-collection.
- **Write side** (`robotaxi.py` DEMO branch): a DEMO job **always records at the
  full width**. Because a reduced-obs course shares its parent's dynamics,
  `_demo_recording_course()` maps `donut_no_hint` → `donut` for collection (see
  `DEMO_RECORDING_COURSE`); a `donut_no_hint` DEMO job transparently collects on
  `donut` and writes 32-wide records. This guarantees no 31-wide records are
  ever written, so the reuse reader can never hit a width mismatch.

#### Course-aware default demo source

When a TRAIN job doesn't name its own demo sources, `do_job()` picks a default
via `_resolve_default_demo_job_id()`:

1. job doc `default_demo_job_id`, else
2. `ROBOTAXI_DEFAULT_DEMO_JOB_ID` env var, else
3. per-course `COURSE_DEFAULT_DEMO_JOB_IDS` map.

Both `donut` and `donut_no_hint` map to the same classic 32-wide default
(`64168c1b58d4d8ccdb76e721`), reused via the read-time drop. Additional job-doc
knobs:

- `demo_job_ids` (list or comma string): extra DEMO-job ids concatenated onto
  the default into one combined expert dataset.
- `skip_default_demo` (bool): drop the baked default and use only
  `demo_job_ids`.
- `demo_source_counts`: explicit per-source step counts (parallel to the
  resolved source list).

If no source resolves at all (course mapped to `None`, or `skip_default_demo`
with no `demo_job_ids`), the job **fails fast** with an actionable message
(stamped as `FAILED` + `eval_error` by `run_jobs_loop`) rather than silently
loading incompatible data.

#### Compatibility / gotchas

- **Checkpoints are not cross-compatible.** A 32-wide (`donut`) policy can't
  resume/eval as 31-wide (`donut_no_hint`) or vice-versa — the network input
  dim differs. Start `donut_no_hint` as a fresh TRAIN.
- The per-job `set_observation_size()` mutates module-level state in
  `collect_training_data`; this is safe **only** because jobs run strictly
  sequentially in the singleton trainer. Concurrent multi-course jobs would
  need per-call plumbing instead.
- `analyze_demo_metrics.py` derives `SPEED_IDX` / `FWD_IDX` from
  `_OBS_DROP_LEADING`, so it reads correct indices for either course.

### Stuck / wedged-car detection

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

### Greedy evaluation policy (mode of the SAC distribution)

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

### AWAC actor regularization (advantage-weighted BC)

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

### Curriculum training (performance-gated track difficulty)

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

### The Mad Scientist (autonomous experiment orchestrator)

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

### Dashboard: architecture notes & limitations

The dashboard is an Express + static-HTML app (`dashboard/`, compose service
`dashboard` on **`node:20`**, `npm start` → `node ./build/index.js`). Pages are
static HTML with inline JS (edits are live on browser reload); only `server.ts`
changes need a **dashboard container restart** to recompile. `prestart` runs
`npm install` **only when `node_modules` is missing**, then always `tsc` — so a
restart is a fast offline compile rather than a network-dependent reinstall
(after bumping the base image, do a one-time `rm -rf node_modules && npm install`
so ABI-sensitive deps rebuild). A few architectural points are worth knowing:

- **Crash resilience.** The server installs process-level `uncaughtException` /
  `unhandledRejection` guards, and every Mongo request handler responds with a
  `500` instead of `throw err`. Historically a single transient Mongo error (a
  change-stream reconnect, replica-set blip, connection reset) threw inside an
  async driver callback → on Node ≥15 an unhandled rejection **terminated the
  process**; `restart: always` then bounced the container through a full
  `npm install` + `tsc` before serving again — the classic "dashboard freezes /
  locks up mid-demo". Errors are now logged and swallowed so one bad query can't
  take the whole UI down. The startup `MongoClient.connect` retries with backoff
  instead of throwing (no boot crash-loop if mongo lags the dashboard).
- **Change-gated GETs (per-client version cursor).** List endpoints
  (`/get_jobs`, `/get_models`, `/get_experiment_designs`, …) are gated by
  `needsUpdate(req, res, changed, coll)`. A MongoDB **change stream**
  (`superviseChangeStream`, requires Mongo be a replica set) bumps a
  monotonically-increasing **per-collection version** on every write, returned on
  every response (data *and* `NO_CHANGES`) via the **`X-Coll-Version`** header.
  A polling client sends **`?since=<last-seen-version>`** and gets `NO_CHANGES`
  unless it's actually behind. Because the decision is derived from the client's
  own cursor — not a single shared `changed` boolean — **N tabs poll cheaply
  without racing each other into a false `NO_CHANGES`**. The WebSocket change
  broadcast also carries the new version so a WS-applied change advances the
  cursor (no redundant refetch). Legacy fallbacks still work: `?force=true`
  always sends (used by one-shot on-demand loads like the New-job modal
  dropdowns), and a request with neither `since` nor `force` uses the old shared
  `changed` flag. This replaced the previous scheme where every polling tab had
  to pass `?force=true` and re-pull the **entire** `jobs` collection (no
  pagination, ~1k docs) every few seconds — the `jobs` and `weakness` tabs now
  ride the cursor. `/get_jobs` still returns the whole collection when it *does*
  send (no server-side pagination yet).
- **Undated jobs sort to the bottom.** Jobs created by the New-job form,
  `Clone-Job`, the Models-tab EVAL path, and trainer-completed jobs typically
  carry **no `create_date` / `update_date`** (only the old MadScientist
  orchestrator stamped `create_date`). Any recency sort keyed on those fields
  collapses undated jobs to epoch 0 and buries brand-new jobs. The Weakness tab
  works around this with an **ObjectId-embedded-timestamp fallback**
  (`jobTimeKey`) — other tabs that sort by date may still exhibit the "my
  just-finished job isn't at the top" symptom. The robust fix is to stamp
  `create_date` in `/add_job`.
- **Large job pickers.** With ~1k jobs, plain `<select>` job dropdowns are
  unwieldy; the Weakness tab replaced its selector with a searchable
  `<input list>` + `<datalist>` combobox (label→id map, ObjectId-recency sort).
  Native `<datalist>` rendering varies across browsers (weaker on Safari).
- **`/add_job` inserts `req.body` verbatim** with no server-side schema
  validation. This is deliberate (new job fields are additive and just flow to
  the trainer), but it means typos in field names fail silently rather than
  erroring — the trainer falls back to defaults.
- **Eval modal stage source.** The Models-tab Eval "Curriculum sweep" Start/End
  stage dropdowns are populated from the **selected model's experiment-design
  `curriculum_stages`** — gym docs carry no stage data. The raw per-edge chicane
  counts aren't exposed as free-form inputs; a curriculum sweep is the only way
  to eval the exact chicane geometry the trainer curriculum used.
- **Mad Scientist tab has no live budget gauge** — cost is shown per-proposal
  only; `/madscientist/decide` sets `status=approved` but does **not** trigger
  the orchestrator directly (the Python worker picks it up on its next poll).
