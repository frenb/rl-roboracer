# Manual setup

Step-by-step development setup on Windows with an NVIDIA GPU. Most people should use the one-command installer in the [README](../README.md#quickstart) instead; it automates all of this.

> **Two Unity projects.** `unity/` is the original Unity 2020.3.11f1 project these steps build. `unity6.6/` is its Unity 6.6 port, which the newest gyms (`*-Unity6.6-*` on the [v0.1 release](https://github.com/frenb/rl-roboracer/releases/tag/v0.1)) are built from; open it with a Unity 6.6 editor instead. The installer downloads prebuilt gyms, so you only need an editor to change the simulator.

## Layout

```
rl-roboracer/
├── docker-compose.yml         # the whole stack
├── docker/
│   ├── fly_brain/             # frozen fly connectome service (Python 3.10)
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
repo, so those three folders must sit next to `rl-roboracer`. The
tfrecords location can be pointed elsewhere with `TFRECORDS_DIR` — see
[the TFRecord corpus](training.md#the-tfrecord-corpus-layout-tfrecords_dir-and-housekeeping).

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
if you will enable the Mad Scientist (off by default), want to keep the
tfrecords corpus somewhere other than `..\tfrecords` (`TFRECORDS_DIR`), or
want a Mongo password other than the default `example`
(`MONGO_ROOT_PASSWORD`). Mongo and mongo-express only listen on
localhost; `.env.example` explains how to change the password on an
existing database.

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

After this, use [Running the stack](operations.md) for day-to-day start/stop.

## Rebuilding the gRPC stubs

If you change `protos/virtual_endpoint/proto/ros_service.proto`:

```powershell
pip install grpcio-tools
.\scripts\gen_protos.ps1
```

(Or `./scripts/gen_protos.sh` on bash.)

## Notes

- First-time image build is [step 4](#4-build-the-docker-images). Incremental `ros-server` rebuilds
  via `DockerfileThin` (layer on `docker_ros-server:working`) are optional.
- MongoDB data, saved models, and tfrecords live as siblings of this repo
  so they can be regenerated, swapped, or wiped without touching git
  history. The gym binary is `unity\Builds\latest\`. Trainer `/tmp` is
  the Compose volume `tmpdata`.
