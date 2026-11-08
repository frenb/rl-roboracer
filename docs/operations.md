# Running the stack

Day-to-day start, stop and restart, the multi-actor layout, and the service map.

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
  bash /python_ws/src/run_trainer.sh --num-envs 4
```

`run_trainer.sh` is a thin wrapper — everything after it is forwarded to
`robotaxi.py` untouched. What the wrapper adds:

- **Your previous session's log survives.** It rotates `/tmp/trainer.log`
  into `/tmp/trainer-logs/trainer-<timestamp>.log` and keeps the 10 most
  recent (`TRAINER_LOG_KEEP` to change that). The old launch piped
  straight into `tee`, which opens with `O_TRUNC` — so every relaunch
  destroyed the log of the run you were probably relaunching *because of*.
- **A session banner** with the UTC timestamp, git branch and SHA, and the
  arguments used. A dashboard client that connects mid-run gets
  `tail -n 100`, so this is usually still on screen.
- `python -u` for unbuffered stdout. Python switches from line-buffered to
  ~8 KB block-buffered the moment it sees a pipe, which would hold back
  `[actor-N]` lines from both the log and the dashboard panel.
- `| tee /tmp/trainer.log`, which is what feeds that panel —
  `dashboard/src/server.ts` tails it over a WebSocket. The
  `compose/scale.yml` overlay disables sim-controller's auto-run of the
  single-env trainer (so it doesn't compete with your multi-env exec for
  MongoDB jobs), so without the `tee` the panel has nothing to show.

The panel tails with `tail -F`, which follows by name, so it keeps
streaming across the rotation and across a manual restart.

Use `/tmp/trainer.log`, not the older `robotaxi.out`. The panel was
switched to it on 2026-07-19 (see the comment in `server.ts`) because
`robotaxi.out` is only written by the base compose command, so the first
manual restart left the panel frozen on a stale file with no indication
anything was wrong.

TensorBoard at `http://localhost:6006/` will show one run with `metrics/`,
`eval/`, `train/`, and `learner/train/` summaries. The dashboard at
`http://localhost:80/` browses past runs (archived to `/tmp/jobsdata/` by
the new TRAIN job's startup cleanup).

#### What keeps the TensorBoard view honest

TB scans two directories — `/tmp/active` as the `current` experiment and
`/tmp/tb_compare` as `compare` — so *anything* left in either is drawn.
Two pieces of housekeeping keep that to what you asked for:

- **EVAL jobs archive their own summaries when they finish.** An eval writes
  `/tmp/active/<job_id>_eval/` so you can watch it live, then moves it to
  `/tmp/jobsdata/` on the way out. Until 2026-09-24 only the TRAIN startup
  cleanup swept `/tmp/active`, so every eval between two training jobs left a
  run behind permanently — 21 had piled up against a single real training run,
  and the live view was unreadable.
- **`prune_jobsdata` protects archives that a saved model points at.** It caps
  `/tmp/jobsdata` at `JOBSDATA_MAX_ARCHIVES` (default 100) by directory mtime,
  but a bucket referenced by any model in `db.models` is retained regardless of
  age, so `keep` is a floor rather than a cap. Recency alone was the wrong
  rule: eval buckets churn daily while the training runs worth comparing are
  old by definition, so model archives were being evicted by eval noise. A
  pruned bucket cannot be opened in the dashboard's **Compare in Analysis**
  view — the selection resolves to `missing` and the pane renders nothing.
  Model *weights* live on a separate mount (`/saved_models`) and were never at
  risk; only the scalar history is lost.

Both live in `rl_agent/robotaxi.py`, so changing either needs a trainer
restart, not just a container restart.

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
  bash -c 'export ROLLOUT_VIZ_ENABLED=1 && bash /python_ws/src/run_trainer.sh --num-envs 2'
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
  bash /python_ws/src/run_trainer.sh

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
| `mongo`          | 27017 (localhost) | Job / model / leaderboard storage                         |
| `mongo-express`  | 8081 (localhost)  | Mongo admin UI                                            |
| `sim-controller` | 6006              | Tensorboard for the live training run                     |
| `fly-brain`      | 50061             | Frozen fly connectome over gRPC (see [fly brain](fly-brain.md)) |
| `dashboard`      | 80                | Golden Layout UI (iframes Tensorboard, logs, jobs, models) |
| `dashboard`      | 8080              | WebSocket tail of `sim-controller:/tmp/trainer.log`       |

## Migrating off the legacy Bitnami Mongo image

The `mongo` service pins `bitnamilegacy/mongodb:6.0`. Broadcom retired the
Bitnami Docker Hub catalogue in August 2025, so `bitnami/mongodb:6.0` no
longer resolves and a cold `docker compose up` fails at the pull.
`bitnamilegacy` is that same image republished (MongoDB 6.0.13, UID 1001,
same entrypoint and data layout), so env vars, both bind mounts, and
existing `../mongodb` data carry over unchanged.

It is a stopgap — the legacy org is frozen and gets no further security
patches. Moving to stock `mongo:6.0` is not a one-line swap: the Bitnami
entrypoint bootstraps the replica set from `MONGODB_REPLICA_SET_MODE`, and
`dashboard/src/server.ts` drives its live UI from change streams, which
require a replica set. A standalone `mongod` breaks the dashboard silently
while ordinary CRUD keeps working, and no test covers it.
