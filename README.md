# robot-tycoon

RL training stack for the robotaxi Unity gym.

## Layout

```
robot-tycoon/
├── docker-compose.yml         # the whole stack
├── docker/
│   ├── ros_server/            # ROS noetic + gRPC bridge build context
│   └── sim_controller/        # CUDA + tf-agents training image build context
├── rl_agent/                  # tf-agents Python code that drives training
├── dashboard/                 # Node/Express monitoring dashboard (port 80)
├── protos/                    # source of truth for the gRPC contract
└── scripts/                   # PowerShell helpers
```

The repo expects to live next to a few sibling data folders:

```
LATEST/
├── saved_models/          # tf-agents checkpoints           (bind-mounted)
├── mongodb/               # mongo data dir                  (bind-mounted)
├── tfrecords/             # demonstration trajectories      (bind-mounted)
├── tmp/                   # tensorboard scratch + run logs  (bind-mounted)
└── UnityBinary/           # pre-built Unity gym executable
```

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

### One-time prerequisite

Build the Unity binary from the editor into a dated subfolder of `unity\Builds\`
(e.g. `unity\Builds\2026.05.09-multi-actor\`), then promote it to the location
the launchers expect:

```powershell
.\scripts\PromoteLatestBuild.ps1
```

This archives any previous `unity\Builds\latest\` and renames the new dated
folder to `latest\`. The launchers always read from `latest\` and copy it into
per-instance directories (`unity\Builds\instances\0..N-1\`) so that Unity's
"force single instance" mutex doesn't block multi-actor runs.

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

- `ros-server` is built from `image: docker_ros-server:thin`, which expects you to
  have first built the base layer (`Dockerfile`) as `docker_ros-server:working`,
  then built `DockerfileThin` on top. If you're starting from scratch, swap the
  compose `build:` block to use `Dockerfile` directly.
- The Unity gym binary, MongoDB data, saved models, tfrecords, and tensorboard
  scratch all live as siblings of this repo so they can be regenerated, swapped,
  or wiped without touching git history.
