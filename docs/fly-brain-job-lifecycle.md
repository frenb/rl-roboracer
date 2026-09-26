# Fly-brain job lifecycle: ridge readout → SAC → running policy

Every step it takes to go from "new encoder idea" to "a trained fly-brain policy
driving in the gym", who or what performs each step today, how long it takes,
and where it has needed a human or a coding agent to step in. The goal is to
make the end-to-end complexity visible, and to list what would have to change
for the whole lifecycle to run as one unattended workflow.

Companion to [`fly-brain-driver-how-it-works.md`](fly-brain-driver-how-it-works.md),
which explains *what* each component does. This document is about *operating*
them. Figures are from the `fly_donut_flow` run on 2026-09-25/26 (job
`6ab728dfa2e93e59cc065c13`) unless stated otherwise.

---

## The short version

There are three phases, and they are less connected than the title suggests:

1. **Ridge readout (behaviour cloning).** Replay the expert demo corpus through
   the frozen brain, fit a linear readout from the 1,314-wide descending trace
   to `[acceleration, steering]`, and evaluate it as `FlyPyPolicy`. This is a
   **baseline and a diagnostic**. It tells you whether an encoder change put
   useful information into the brain, cheaply and offline.
2. **SAC training (reinforcement learning).** A TRAIN job on a fly course, where
   the trace *is* the observation and SAC learns the readout from reward.
   **SAC does not start from the ridge weights.** Nothing from phase 1 is loaded
   into phase 2; they share only the encoder and the brain.
3. **Running the policy.** An EVAL job that loads a saved SAC checkpoint and
   drives it on the same fly course, with the fly-brain service in the loop.

So the real dependency chain is **encoder → (ridge diagnostic, optional) → SAC →
eval**, and the ridge phase is a gate on whether the SAC run is worth its
~15 hours, not an input to it.

Today, about 20 distinct steps are spread across a PowerShell stack script, the
dashboard, MongoDB, two containers and ad-hoc Python. Roughly half of them are
manual. On the night of 2026-09-25, keeping one SAC job alive took nine separate
interventions, none of which were about the model.

---

## The pipeline at a glance

| # | Stage | Triggered by today | Duration | Produces |
|---|---|---|---|---|
| 0 | Bring up the stack | Human: `scripts/Start-Stack.ps1` | ~1 min + Unity warm-up | Containers, Unity clients, watchdog |
| 1 | Register the Unity gym build | Human: dashboard **Gyms** tab | seconds | `gyms` row |
| 2 | Encoder change (new cue, new populations) | Coding agent: code edits in 6 files | hours | New course type |
| 3 | Calibrate encoder gains | Coding agent: ad-hoc script, constant pasted into code | minutes | e.g. `FLOW_GAIN = 1.37` |
| 4 | Probe candidate neuron populations | Coding agent: `docker/fly_brain/probe_candidates.py` | minutes | Go / no-go on the populations |
| 5 | Replay the corpus through the brain | Coding agent: `python -m fly_brain.step5_readout`, on a private brain server | **46.6 min** (100 episodes) | Cached traces in `/tmp` |
| 6 | Fit the ridge and report held-out R² | Same script, seconds on a warm cache | seconds | `/tmp/fly_readout_*.npz` |
| 7 | Install the readout | Human: copy into `/saved_models/robotaxi/FlyPyPolicy/0/` | seconds | Deployable readout |
| 8 | Baseline EVAL (`FlyPyPolicy`) | Human: dashboard **Models** tab | minutes | Leaderboard row |
| 9 | Create the SAC TRAIN job | Human: dashboard **Jobs** tab (or Mongo edits) | seconds | `jobs` row, `NOT_STARTED` |
| 10 | Launch the trainer | Human: `run_trainer.sh` inside `sim-controller` | — | Trainer process polling the queue |
| 11 | Trainer startup | Trainer | **~6.5 min** cold | Env, networks, replay buffer |
| 12 | Training loop | Trainer | **~15 h** for 100k iterations | Checkpoints, eval curve |
| 13 | Save best models | Trainer, on every new best eval | seconds each | `SacAgent/<n>_step_<k>` + `models` row |
| 14 | Pause / resume / crash recovery | Human, dashboard, or watchdog | varies | Job resumes from checkpoint |
| 15 | EVAL the best checkpoint | Human: dashboard **Models** tab | minutes | Leaderboard row |
| 16 | Compare and run controls | Human | — | The actual result |

---

## Stage by stage

### 0. Bring up the stack

`scripts/Start-Stack.ps1 -N <clients>` starts the Docker Compose stack
(`mongo`, `dashboard`, `sim-controller`, `fly-brain`, the `ros-server-*`
bridges, `madscientist`), launches `N` supervised Unity clients in Windows
Terminal tabs, and starts `scripts/Watchdog.ps1`. It does **not** start the
trainer.

Fly courses must run with **one** Unity client and `--num-envs 1`, because the
`fly-brain` service holds exactly one brain. Two environments sharing it
interleave silently (see "Constraints that bite" in the how-it-works doc). The
stack scripts default to `N = 2`, and so does the watchdog's auto-restart, so a
fly run has to override this every time.

The `fly-brain` container loads the connectome in about 15 s, then (since
2026-09-26) runs a 20-step warm-up to compile its CuPy kernels before opening
port 50061.

### 1. Register the gym build

A Unity build lives under
`UnityBinary/<build-name>/robotaxi gym level 1.exe`. It gets a row in the
`gyms` collection via the dashboard's **Gyms** tab, and a job references it by
`gym_id`, `gym_name` and `gym_file_path`.

At job start the trainer POSTs the job's gym to `dashboard/set_desired_gym`.
Each Unity supervisor (`scripts/RunClientWrapper.ps1`) polls
`get_desired_gym`, and if the build differs from what it is running, it copies
the build into `unity/Builds/instances/<i>/` and **relaunches Unity**. See the
"gym switch race" below for why that timing matters.

A gym is not a course. `FlyBrain-wCourseJetRacer2026.09.22-v27` is a gym (the
Unity binary and its track geometry); `fly_donut_flow` is a course (the Python
side: encoder, observation, action spec).

### 2. Encoder change → new course type

A new cue (for example optic flow) means a new encoder class and a new course
type, and the course name is threaded through six places by hand:

| File | Change |
|---|---|
| `rl_agent/fly_brain/encoder.py` | The encoder class, its populations and gains |
| `rl_agent/environments/courses/fly_donut_course.py` | A course subclass that returns the new encoder |
| `rl_agent/environments/robotaxi_env.py` | A `course_type` branch that instantiates it |
| `rl_agent/collect_training_data.py` | `COURSE_OBS_KIND` and `COURSES_WITHOUT_DEMOS` |
| `dashboard/components.js` | The course dropdown |
| `dashboard/models.html`, `dashboard/jobs.html` | Observation sizes and validation hints |

Miss one and the failure shows up somewhere else, such as a dashboard that
refuses the job or a trainer that loads the wrong observation width.

### 3–4. Calibrate and probe

Gains are set so the 99th percentile of each cue over the demo corpus lands at
`MAX_AMOUNT = 0.8`. That percentile is computed by a throwaway script and the
result is **pasted into the code as a constant**. Change the encoder formula
and forget to recalibrate, and nothing errors; the cue just clips or sits in the
noise.

Before committing to a population, `docker/fly_brain/probe_candidates.py`
stimulates candidates left, right and bilaterally across seeds against a
noise-only baseline, to check that the population drives a distinct,
reproducible descending response.

Both steps are **baked into saved models**. A SAC checkpoint only works with the
exact encoder, gains and brain parameters it was trained against, and none of
these are recorded on the `models` row. Changing `FLOW_GAIN` after training
silently changes what every existing `fly_donut_flow` checkpoint sees.

### 5–6. Ridge readout (`step5_readout.py`)

```
docker compose exec -w /python_ws/src -e FLY_ENCODER=flow -e FLY_EPISODES=100 \
    sim-controller python -m fly_brain.step5_readout
```

1. Read `FLY_EPISODES × 1000` rows from the legacy expert corpus
   `/tfrecords/job_64168c1b58d4d8ccdb76e721` (31-D observations).
2. **Replay** every row through the brain, one episode at a time, resetting the
   brain (seeded by episode) and the encoder at each 1,000-row boundary. This is
   serial and costs ~10 ms per frame plus round-trips: **46.6 minutes** for 100
   episodes with the flow encoder.
3. Cache the traces to `/tmp/fly_step5_trace_<N>ep[_<encoder>].npz`, so every
   refit afterwards is free.
4. Fit a standardised ridge over a λ sweep (`1e-2 … 1e5`), splitting **by
   episode**: 80 episodes (80,000 frames) to fit, 20 held out.
5. Report held-out R² for the trace, against two controls: the encoder's own
   cues, and the raw 31-D observation.
6. Write `w`, `mu`, `sd`, `y_mean` to `/tmp/fly_readout_<N>ep[_<encoder>].npz`.

Two operational hazards:

- **The replay needs its own brain.** Replaying through the live `fly-brain`
  service while a SAC job is running would interleave the two workloads and
  corrupt both. The replay on 2026-09-25 ran against a second server started by
  hand on port 50062 and stopped afterwards.
- **The legacy corpus is a poor throttle target.** 72% of its actions are below
  the course's 0.05 throttle floor, so throttle R² is capped by the data, not
  the brain. Steering R² had plateaued by ~80 episodes (0.620 for the 4-cue
  encoder); more frames of the same corpus do not help.

Measured results:

| Encoder | R² acceleration | R² steering |
|---|---|---|
| 4-cue `RayEncoder` (`fly_donut`) | 0.176 | 0.620 |
| 6-cue `FlowEncoder` (`fly_donut_flow`) | **0.245** | **0.668** |

### 7–8. Install the readout and run the baseline

`FlyPyPolicy` loads `/saved_models/robotaxi/FlyPyPolicy/0/readout.npz`, so the
step-6 output has to be copied there by hand. It runs as an EVAL job with
`model_type: FlyPyPolicy`, **no location**, and course **`donut_no_hint`**,
because the policy does its own encoding and needs the 31-D ray vector.

`FlyPyPolicy` hard-codes the 4-cue `RayEncoder`. **The flow readout cannot be
evaluated as a policy yet**: its R² exists, but there is no driving baseline for
it. There is also only one slot (`FlyPyPolicy/0`), so installing a new readout
overwrites the previous one.

### 9. Create the SAC TRAIN job

The fields that matter on the `jobs` document:

| Field | Value for this run |
|---|---|
| `job_type` | `TRAIN` |
| `model_type` | `SacAgent` |
| `course_type` | `fly_donut_flow` |
| `gym_id`, `gym_name`, `gym_file_path` | the v27 build |
| `reward_design_id` | `Goal-count speed (v4, TIME_COST 0.0073)` |
| `experiment_design_id` | `AWAC + No-BC + reward_scale1 (no curriculum)` |
| `num_iterations` | 100000 |
| `status` | `NOT_STARTED` |

The experiment design's name mentions AWAC, but for any course in
`COURSES_WITHOUT_DEMOS` the trainer forces behaviour-cloning pre-training, demo
prefill, demo sampling and AWAC all to zero. This is pure SAC from scratch.

### 10–11. Launch and startup

```
docker compose -f docker-compose.yml -f compose/scale.yml exec -d sim-controller \
    bash /python_ws/src/run_trainer.sh
```

The trainer polls MongoDB for `NOT_STARTED` jobs. `run_trainer.sh` rotates the
previous `/tmp/trainer.log` into `/tmp/trainer-logs/` on every launch, so follow
the live log with `tail -F`, not `tail -f`.

Cold-start breakdown from the original launch of this job:

| Phase | Time |
|---|---|
| Unity handshake | 0.6 s |
| Network build | 0.1 s |
| **Expert TFRecord load (500k records)** | **200.3 s** |
| Agent build + Reverb setup | 3.4 s |
| Initial collect | 59.7 s |
| Learner + eval actor build | 36.0 s |
| First eval + pre-train eval | 84.1 s |
| **Total** | **384 s** |

More than half of startup is loading the legacy demo corpus, and for fly courses
**every row of it is thrown away**: demo prefill is zero and the corpus is 31-D
against a 1,314-D observation. The phase prints nothing for 200 s, which is
enough to make a healthy trainer look hung.

On a resume, initial collect is skipped (the replay buffer is restored from its
Reverb checkpoint), but the demo load still runs.

### 12–13. Training loop and model saving

Each iteration collects one env step on client-0, then runs one learner update.
Steady state is ~0.5 s per iteration (collect ~0.15 s, learner ~0.35 s), and
~0.54 s per iteration including periodic evals, which works out to about 15
hours for 100,000 iterations.

Every eval that sets a new best `AverageReturn` exports the **greedy** policy
with `PolicySaver` to `/saved_models/robotaxi/SacAgent/<n>_step_<k>` and inserts
a `models` row with `course_type`, observation and action specs, and reward and
experiment design provenance, and `is_global_best: true` (demoting the previous
best for that job). Saved so far for this job:

| Checkpoint | AverageReturn |
|---|---|
| `SacAgent/7644_step_4393` | 4.38 |
| `SacAgent/7645_step_11301` | 6.04 |
| `SacAgent/7646_step_13552` | 10.47 |
| `SacAgent/7647_step_16642` | **12.87** (current best) |

The learner also auto-checkpoints every 100 steps under
`/tmp/active/<job_id>/learner/`, which is what pause and crash recovery resume
from.

### 14. Pause, resume and recovery

```
NOT_STARTED ──► IN_PROGRESS ──► DONE
                    │    │
                    │    └──► FAILED ──(watchdog, if a checkpoint exists)──► NOT_STARTED
                    │
                    └──► PAUSE_REQUESTED ──► PAUSED ──(set NOT_STARTED)──► NOT_STARTED
```

- **Pause** is cooperative: the dashboard (or watchdog) sets `PAUSE_REQUESTED`;
  the trainer saves a Learner checkpoint and a Reverb checkpoint, writes
  `paused_at_step`, and sets `PAUSED`.
- **Resume** is `status = NOT_STARTED` **with `paused_at_step` still set**.
  That field is what tells the trainer to restore `/tmp/active/<job_id>/`
  instead of archiving it into `/tmp/jobsdata/<job_id>/` and starting fresh.
- **The watchdog** (`scripts/Watchdog.ps1`) polls the trainer log. On a wedge
  signature it pauses the job, kills the trainer, runs `Restart-Stack.ps1`, sets
  the job back to `NOT_STARTED` and relaunches the trainer. It also auto-resumes
  `FAILED` jobs that still have a checkpoint, at most twice per job.

### 15. Running the trained policy

EVAL from the **Models** tab: tick the `SacAgent/<n>_step_<k>` row and click
**+ Eval selected**. The job carries `model_type: SacAgent`, the checkpoint's
`location` and its `course_type` (`fly_donut_flow`).

What is easy to miss is that **the saved model is only the readout**. The
SavedModel maps a 1,314-wide trace to an action; the encoder and the brain are
supplied at run time by the course and the `fly-brain` service. Running the
policy therefore needs the same things training did: the `fly-brain` container
up and warm, the same encoder code and gains, a matching gym, and one Unity
client. The same applies to any deployment outside the gym. The car needs the
connectome service (166,700 neurons on a GPU) next to it, stepped once per
control step.

### 16. Compare and control

A result is four numbers at an equal step budget, compared on
`eval/goals_per_episode_this_eval` rather than `AverageReturn`: SAC on
`donut_no_hint`, the ridge readout, SAC on the real connectome, and SAC on a
degree-preserving **scrambled** connectome. The scrambled control does not exist
yet; without it, there is no evidence that the fly wiring matters rather than
just any large random reservoir.

---

## Where it needed intervention (2026-09-25/26)

Everything below happened to one SAC job in one evening. None of it was about
the model.

| # | What happened | Root cause | Status |
|---|---|---|---|
| 1 | Job paused at step 6,964 | Watchdog false positive: two slow collects (6.7 s, 13.3 s) after a transient `Apply force timed out`. The trainer had already recovered, but both lines were still inside the watchdog's 600-line window. | Open |
| 2 | Job sat `PAUSED` for ~2 hours with no trainer | Watchdog died during `Restart-Stack.ps1`, so its "resume job" and "relaunch trainer" steps never ran. Nothing reported it. | Open |
| 3 | Docker Desktop went down mid-diagnosis | Host | — |
| 4 | First resume hung, then failed with `DEADLINE_EXCEEDED` | Gym switch race: the trainer's own `set_desired_gym` call made the Unity supervisor relaunch client-0 while the trainer was connecting to it. | Open |
| 5 | Every start after a stack restart failed with `DEADLINE_EXCEEDED` on `FlyBrain.Step` | Fly-brain cold start: the first `Step` on a fresh container compiles CuPy kernels and overran the client's 10 s deadline. | **Fixed**: server warm-up before the port opens |
| 6 | A relaunch archived the checkpoint instead of resuming | Killing a trainer mid-resume left the job without `paused_at_step`. The next start treated it as fresh and moved `/tmp/active/<job>` to `/tmp/jobsdata/<job>`. Recovered by restoring the field. | Open |
| 7 | A healthy trainer was killed as "hung" | The 200 s silent demo load, which fly courses never use. | Open |
| 8 | The trainer looked dead while running | `run_trainer.sh` rotates the log; the terminal was on `tail -f`, following the rotated file. | Workaround: `tail -F` |
| 9 | Two Unity clients launched for a one-env course | `Restart-Stack.ps1` and the watchdog default to `N = 2`. | Open |

Earlier in the same project, getting to the point of launching this job also
needed a coding agent to edit six files for a new course, calibrate gains by
hand, run a private brain server for the replay, and edit MongoDB directly to
switch the job's gym.

---

## What a single unattended workflow needs

The lifecycle can be one workflow, but not by scripting the current steps in
order: most interventions above were failures *between* the steps. In rough
priority order:

### Make the existing stages safe to run unattended

1. **Resume state that survives a kill.** Derive "this is a resume" from the
   checkpoint on disk rather than from `paused_at_step` alone. The trainer
   already detects `IN_PROGRESS` jobs with a checkpoint at startup; extend that
   to `NOT_STARTED`, and never archive a directory that holds a newer checkpoint
   than the job's recorded step.
2. **Recovery that finishes or says so.** The watchdog must be idempotent and
   resumable: record each recovery step in MongoDB, and on restart pick up where
   it left off. A recovery that stalls should raise an alert, not leave a job
   paused with no trainer.
3. **A stable gym before the trainer connects.** Either switch the gym before
   the job is marked `IN_PROGRESS` and wait for the supervisor to report the new
   build as ready, or skip the switch when the build already matches.
4. **Per-course client count.** Fly courses declare `num_envs = 1`, and stack
   restarts and watchdog relaunches read it from the job instead of defaulting
   to 2.
5. **Watchdog tuned to recovery, not to history.** Only count slow collects
   that are still happening (for example, the last *k* iterations), so a
   transient stall that has already cleared does not trigger a full restart.
6. **Skip the demo load for courses without demos**, and print progress for any
   phase that runs longer than a few seconds.

### Turn the manual stages into jobs

7. **A `READOUT_FIT` job type.** Parameters: encoder, episode count, corpus.
   It starts its own brain server (never the live one), replays, fits, and
   writes the readout into a versioned model slot with its R² and controls
   recorded on the row. This replaces stages 5–7.
8. **`FlyPyPolicy` parameterised by encoder**, reading the encoder name and
   gains from the readout file, so any fitted readout can be evaluated.
9. **Encoder provenance on every model.** Record the encoder class, its gains
   and the brain parameters (`dt`, `tau`, `substeps`, `eye_drive`) on every
   `models` row, and refuse an EVAL whose runtime values differ.
10. **Calibration as code.** Compute gains from the corpus at course
    construction (or as part of `READOUT_FIT`) and store them with the readout,
    instead of pasting constants.
11. **Course registration in one place.** A single course registry that the
    trainer, `collect_training_data` and the dashboard all read, so adding a
    course is one entry rather than six edits.

### Chain the jobs

12. **Job dependencies.** A `depends_on` field and a gate on the parent's
    outcome, so one submission expands into:

    ```
    READOUT_FIT (flow encoder)
        └─► gate: R² steering ≥ previous encoder
              ├─► EVAL FlyPyPolicy (baseline)
              └─► TRAIN SacAgent on fly_donut_flow
                    └─► EVAL best checkpoint
                    └─► TRAIN SacAgent on scrambled connectome (control)
                          └─► EVAL best checkpoint
    ```

    The `madscientist` orchestrator already seeds experiment designs, queues
    TRAIN jobs from approved proposals and ingests completed outcomes, so it is
    the natural owner of this chain. The pieces it lacks are the new job types
    above and dependency-aware scheduling.

With 1–6 in place, the SAC and EVAL stages would have survived the night of
2026-09-25 without anyone touching them. With 7–12, the ridge diagnostic, SAC
training, the baseline, the control and the final eval become a single
submission.
