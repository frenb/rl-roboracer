# robot-tycoon

RL training stack for the robotaxi Unity gym.

## Layout

```
robot-tycoon/
├── docker-compose.yml     # the whole stack
├── ros_server/            # ROS noetic + gRPC bridge (Docker build context)
├── rl_agent/              # tf-agents Python code that drives training
├── dashboard/             # Node/Express monitoring dashboard (port 80)
├── protos/                # source of truth for the gRPC contract
└── scripts/               # PowerShell helpers
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

From the repo root:

```powershell
docker compose up -d
.\scripts\RunClientWrapper.ps1
```

The wrapper launches the Unity executable from `..\UnityBinary\...\robotaxi gym level 1.exe`
(edit the path inside `RunClientWrapper.ps1` to point at the build you want)
and respawns it if it stops responding.

Service map:

| Service          | Port  | What it does                                              |
|------------------|-------|-----------------------------------------------------------|
| `ros-server`     | 10000 | TCP socket the Unity client connects to (ROS bridge)      |
| `ros-server`     | 50051 | gRPC virtual endpoint used by `rl_agent/api.py`           |
| `mongo`          | 27017 | Job / model / leaderboard storage                         |
| `mongo-express`  | 8081  | Mongo admin UI                                            |
| `sim-controller` | 6006  | Tensorboard for the live training run                     |
| `dashboard`      | 80    | Golden Layout UI (iframes Tensorboard, logs, jobs, models) |
| `dashboard`      | 8080  | WebSocket tail of `rl_agent/robotaxi.out`                 |

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
