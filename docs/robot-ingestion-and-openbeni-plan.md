# Standardizing Robot Ingestion and Importing OpenBeni

**Status:** In progress on `microduck-phase0`. Part 1 (manifests + `mjlab_core` + dashboard/MCP generalization) is landed. Phase A is done: `beni_trainer` builds (torch 2.7.1+cu128, warp 1.17), 10-iteration `Beni-Velocity-Flat` and `Beni-Jump-Flat` runs train on GPU, `headless_export` writes `policy.onnx` + `policy.mnn` (ONNX checks out: `obs[1,40]` + `h_in[1,1,512]` → `actions[1,7]` + `h_out`), and the `beni-trainer` worker registers `openbeni` in `db.robots` with both gyms and the passthrough reward / default experiment designs. Phase B/C code is committed. Phase D is done (Nov 18): the dashboard reads `engine` / `supports` from `/get_robots` everywhere (`isDuckRow` is gone; MicroDuck-only wording now covers any mjlab robot, and the job form opens on each robot's first task), and CI runs the CPU-only `openbeni` tests (launch, experiment design / GRU-512 lock, gyms, baseline seed). The worker now registers upstream `assets/models/policy.pt` (the jump-stage policy, 15k updates, GRU-512) as an `openbeni` model on `Beni-Jump-Flat`. Remaining: Phase E. Its first EVAL (the baseline on `Beni-Jump-Flat`) is queued behind the car jobs; no `openbeni` job has run through the queue yet.

With three distinct morphologies — **`roboracer`** (JetRacer 4-wheel Ackermann/parallel-steer rover), **`microduck`** (14-DoF walking/roller biped), and **`openbeni`** (7-DoF wheel-legged jumper with a 2-DoF gimbal) — the platform's two-robot shortcuts (`isDuckRow`, hardcoded `['roboracer', 'microduck']` enums, and duplicated mjlab worker code) break cleanly in one place. This plan uses the `openbeni` import to establish a standard **Robot Adapter Contract** across the whole stack.

---

## Robot comparison

| Property | `roboracer` (JetRacer) | `microduck` | `openbeni` (Beni) |
|---|---|---|---|
| **Morphology** | 4-wheel rover (`robots/jetracer/jetracer.urdf`) | 14-DoF biped + optional rollers | 7-DoF wheel-legged (`L1/R1/L2/R2`, `LW/RW`, `head_yaw_joint`) |
| **Trainer stack** | TF 2 / tf-agents SAC + AWAC / RLPD / JSRL | PyTorch + `mjlab==1.3.0` + `rsl_rl` PPO | PyTorch 2.7 + `mjlab>=1.6.0` + `rsl_rl` PPO (`BeniVelocityOnPolicyRunner`) |
| **Simulator** | Unity 6.6 lockstep (`0.4 s`), ROS-TCP + gRPC | MuJoCo Warp (GPU) + Unity MuJoCo server | MuJoCo Warp (GPU); optional MuJoCo WASM browser viewer (`web/`) |
| **Actor architecture** | MLP / Conv+MLP (`84x84x3` + `31`-D) | Feedforward MLP (`61 → 14` on Walk) | Recurrent `RNNModel` GRU-512 (`40`-D actor / `50`-D critic → `7`-D) |
| **Curriculum** | Track geometry + JSRL BC-guide stages | Task-internal / terrain curricula | Two-phase: `Beni-Velocity-Flat` (adaptive recovery bank) → `Beni-Jump-Flat` (`recovery_gate` + jump normalizer priming) |
| **Export artifacts** | TensorFlow SavedModel (`saved_models/robotaxi/`) | `.pt` → `.onnx` | `.pt` → `.onnx` + `.mnn` (`beni_rl.headless_export`) |

---

## Part 1: Standardized Robot Ingestion Contract

Every robot registered in `rl-robotlab` conforms to four shared layers:

### 1. Declarative Robot Manifest (`robots/<robot>/robot.yaml`)

Replace scattered hardcoded constants (`DEPLOY_SHAPE`, `isDuckRow`, `ROBOT_LABEL`) with a committed folder per robot:

- `robots/jetracer/robot.yaml` (`id: "roboracer"`)
- `robots/microduck/robot.yaml` (`id: "microduck"`)
- `robots/openbeni/robot.yaml` (`id: "openbeni"`)

Each worker publishes its `robot.yaml` to MongoDB (`db.robots`) at startup alongside `env_spec` / `experiment_design_schema`:

- **Identity & UI:** `id`, `display_name`, `morphology`, `description`, `badge_color`
- **Engine & Queue:** `engine` (`tf_agents_unity` | `mjlab_rsl_rl`), `worker_queue_id`, `models_dir`, `supports`: `{ eval, export, video, weakness_map, demos, unity_clients }`
- **Model & Export Contract:** `checkpoint_format` (`saved_model` | `pt`), `export_formats` (`onnx`, `mnn`), `recurrent` (`bool`), `deploy_shapes` (per task prefix, for the Models-tab Compat column)
- **Assets:** relative paths to `urdf`, `mjcf`, `meshes/`, `motions/`

### 2. Shared `mjlab` Worker Core (`rl_agent/mjlab_core/`)

Instead of duplicating `rl_agent/duck/{worker,gyms,reward_design,experiment_design,evaluate,render_progress}.py` per robot:

- **Extract robot-agnostic `mjlab` scaffolding into `rl_agent/mjlab_core/`:**
  - Queue claiming (`job_queue.may_claim`), heartbeat, `JobLog`, pause/resume/cancel lifecycle, progress parsing, checkpoint clip + progression video stitching (`rl_agent/media.py`), headless rollout evaluation (`mean_return`, `fall_rate`, `mean_episode_length`), and `cfg_hook` loading (`ENV_CFG_HOOK` / `RL_CFG_HOOK`).
- **Keep per-robot plugins thin (`rl_agent/robots/<robot>.py` or `rl_agent/<robot>/`):**
  - Task discovery & friendly gym names (`gyms.py`)
  - Reward design `LIBRARY_MODULES` and wrapped-term support
  - Robot-specific `experiment_design` schema fields (e.g., Beni's recovery & jump curriculum knobs)
  - Checkpoint → deployment exporter (`onnx` / `mnn`) and model compatibility rules
- **One container per incompatible dependency lock:**
  - `microduck_rl` pins `mjlab==1.3.0` + `torch==2.9.1`; `open-beni` specifies `mjlab>=1.6.0` + `torch>=2.7,<2.8` + `MNN==3.5.0`. Each gets its own Dockerfile and optional compose overlay (`compose/duck.yml`, `compose/beni.yml`) sharing `./rl_agent` mounted at `/python_ws/src`.

### 3. Robot-Agnostic Dashboard & MCP Console

- **Dashboard (`/get_robots`):** Replace `isDuckRow = (r.robot !== 'roboracer')` and hardcoded `['roboracer', 'microduck']` lists across `dashboard/{jobs,models,gyms,reward_designs,experiment_designs,reward_specs}.html` and `dashboard/components.js` with capability flags from `db.robots` (`engine`, `supports.*`). Register each robot's media root under `/media` in `dashboard/src/server.ts`.
- **MCP Console (`mcp/console/src/`):** Source the `robot` enum from the robot registry/config so `list_jobs`, `create_job`, `list_models`, `list_gyms`, `get_experiment_design_schema`, `tail_trainer_log`, and `eval_models` support `openbeni` (and future robots) without per-tool special cases.

### 4. Reward Specs (`rl_agent/specs/targets.py`)

Add an `openbeni` target alongside `roboracer` and `microduck` so robot-agnostic Markdown Reward Specs compile, lint, and verify coverage across all three robots.

---

## Part 2: `openbeni` Import Plan

### Phase A — Submodule, Assets & Container

1. **Add `Mondo-Robotics/open-beni` as a git submodule** at `third_party/open-beni` (MIT license; document in `THIRD_PARTY_NOTICES.md`).
   - Brings in `assets/beni_base/{urdf/beni.urdf, mjcf/beni.xml, meshes/*.obj, textures/}`, `assets/motions/*.npz` (`base_jump_h0.18.npz`, `recovery_*_states_*.npz`), pretrained `assets/models/policy.{pt,onnx,mnn}`, and `python/beni_rl/`.
2. **Create `robots/openbeni/robot.yaml`** declaring Beni's asset paths, 7-DoF action order, `40`-D actor / `50`-D critic obs, GRU-512 recurrence, and `onnx`/`mnn` export artifacts.
3. **Add `docker/beni_trainer/Dockerfile` + `compose/beni.yml`:**
   - Install `beni_rl` with `uv` (`mjlab>=1.6.0`, `torch 2.7 cu128`, `onnxruntime`, `MNN==3.5.0`, `pymongo`, `pytest`, `imageio-ffmpeg`) and EGL/d3d12 GPU rendering.
   - Mount `../saved_models/openbeni` → `/openbeni_rl/logs` and `./rl_agent` → `/python_ws/src`.
   - Smoke test 10-iteration training on `Beni-Velocity-Flat` and `Beni-Jump-Flat` plus `beni_rl.headless_export` (`policy.onnx` + `policy.mnn`).

### Phase B — Shared `mjlab_core` Extraction & `openbeni` Worker

1. **Extract `rl_agent/mjlab_core/`** from `rl_agent/duck/`, keeping all existing `rl_agent/duck/test_*.py` checks passing.
2. **Implement `rl_agent/openbeni/`:**
   - `gyms.py`: sync `openbeni:Beni-Velocity-Flat` and `openbeni:Beni-Jump-Flat` into `db.gyms` (`robot: "openbeni"`, `reward_terms`, `reward_defaults`, `episode_s`).
   - `cfg_hook.py` + `train_entry.py`: intercept `mjlab.tasks.registry.load_env_cfg` / `load_rl_cfg` cleanly without modifying the `third_party/open-beni` submodule.
   - `evaluate.py` / `mjlab_core` evaluator support: handle `BeniVelocityOnPolicyRunner` and recurrent (`RNNModel`) hidden state reset across `done` environments.
   - `export`: invoke `beni_rl.headless_export` to emit `policy.onnx` (and `policy.mnn`), validating the exported ONNX graph with `onnxruntime`.

### Phase C — Training Curriculum, Reward Designs & Experiment Designs

1. **Stage 1 (`Beni-Velocity-Flat` — Locomotion + Adaptive Recovery):**
   - Uses `fallen_reset_curriculum` (`0.2 → 0.5` fallen resets over `warmup_iterations=1000`, `ramp_iterations=1000`) and `AdaptiveRecoverySampler` from `recovery_contact_states_train.npz`.
   - Expose PPO + recovery curriculum parameters in `openbeni`'s `experiment_design` schema while guarding the GRU-512 recurrent deployment invariant (`rnn_type=gru, rnn_hidden_dim=512`).
2. **Stage 2 (`Beni-Jump-Flat` — Recovery-Gated Jump Curriculum):**
   - Requires a Stage-1 checkpoint via `init_from` (`BeniVelocityOnPolicyRunner` verifies `recovery_gate` on held-out states, calls `prime_jump_normalizers` for the 4-D `jump_command`, then ramps jump frequency/height).
   - Expose jump curriculum fields (`jump_warmup_iters`, `jump_ramp_iters`, `jump_probability`) in `experiment_design` and seed `third_party/open-beni/assets/models/policy.pt` as a baseline model so Stage 2 can also start immediately.
3. **Reward Designs & Reward Specs:**
   - Seed `openbeni-passthrough`, register `beni_rl.mdp` / `beni_rl.recovery` / `beni_rl.jump` in `reward_design`, and ensure edits to wrapped jump terms (`jump.flat_reward`, `jump.contact_penalty`, `jump.jump_position_limits`) update inner weights/params cleanly without dropping the wrappers.
   - Add the `openbeni` target to `rl_agent/specs/targets.py`.

### Phase D — Dashboard, MCP & CI Generalization

1. Expose `/get_robots` in `dashboard/src/server.ts` and register `/saved_models/openbeni` in `MEDIA_ROOTS`.
2. Update dashboard pages and `components.js` to drive robot pills, columns, and job/model actions from `robot.engine` and `robot.supports` (removing `isDuckRow`).
3. Update `mcp/console` tools/tests and `.github/workflows/ci.yml` to cover `mjlab_core` + `openbeni` unit tests (CPU-only).
4. *(Optional later)* Wire `open-beni`'s MuJoCo WASM browser viewer (`web/`) to preview exported `openbeni` ONNX policies directly from the Models tab.

### Phase E — End-to-End Verification

1. Register and `EVAL` `open-beni`'s pretrained `policy.pt` as a reference baseline (`mean_return`, `fall_rate`).
2. Run a short `Beni-Velocity-Flat` `TRAIN` job from the console → verify live logs, checkpoint clips/GIFs, `progress_video`, `EXPORT` (`.onnx` + `.mnn` validated by `onnxruntime`), and `EVAL`.
3. Warm-start a `Beni-Jump-Flat` `TRAIN` job with `init_from` pointing to the Velocity checkpoint → verify `recovery_gate`, jump normalizer priming, jump curriculum, and ONNX/MNN export.

---

## Part 3: Remote Simulation

**Goal:** run training jobs on rented cloud GPUs as well as on this PC, from the same queue, dashboard and console. Scope for now: the shared groundwork, then remote workers for the mjlab robots (`openbeni`, `microduck`). These can be persistent rented pods (R2) or one Hugging Face Job per queued job (R3). Out of scope until these work: other on-demand platforms (SkyPilot / RunPod / Modal), and moving the car off this PC (it needs headless Linux Unity gyms next to the trainer, because the 0.4 s lockstep over ROS-TCP + gRPC shouldn't cross a network).

### What ties the stack to one machine today

| Coupling | Where | Remote consequence |
|---|---|---|
| One GPU, strict FCFS across robots | `rl_agent/job_queue.py` `may_claim` (no robot may start while another robot's job is `IN_PROGRESS`) | A remote GPU would sit idle waiting for the local one. |
| One worker per robot | `job_queue.beat` keys `db.workers` by robot id | A second `openbeni` worker overwrites the first one's heartbeat. |
| Artifacts on a bind mount | `../saved_models` mounted into trainers, ROS servers and the dashboard; the dashboard's `/media` serving and `/delete_model` read it directly | Remote checkpoints, clips and ONNX/MNN files are invisible to the dashboard and to later jobs (EVAL, EXPORT, warm start, curriculum carry). |
| Logs and TensorBoard on local volumes | The dashboard tails `/tmp/trainer.log` and per-robot `/shared_tmp/*_trainer.log`; TensorBoard reads `/tmp/active` and `/tmp/tb_compare` | No live log panel or curves for remote jobs. |
| Code via bind mount | `./rl_agent` → `/python_ws/src` in every worker | A remote host needs the code shipped at a known commit. |
| Mongo on the Compose network | `MONGO_URL` → `mongo:27017` | Remote workers need a private route to it. This is already just a connection string. |

### Phase R1: Shared groundwork

1. **Private network.** Join this PC and each remote host to a Tailscale tailnet. Remote workers use `MONGO_URL=mongodb://…@<pc-tailnet-name>:27017/`. Mongo is never exposed publicly; keep auth on and use a dedicated worker user with access to the `robotaxi` database only.
2. **Hosts and capacity in the queue.**
   - Workers register as `<robot>@<host>`, with `{robot, host, gpus, gpu_name, labels, last_seen}`.
   - `may_claim` changes from "one job at a time overall" to "one job per GPU slot". A job can carry an optional `host` / `labels` constraint, and car jobs keep `host: local` implicitly.
   - **Claims become atomic.** Today `MjlabWorker.next_job` only reads the queue, and treats any of the robot's `IN_PROGRESS` jobs as its own to resume. With two workers for one robot, the second would adopt the first's running job. Claiming changes to one `find_one_and_update` from `NOT_STARTED` to `IN_PROGRESS` that stamps `claimed_by`.
   - Pause, resume and crash recovery only touch jobs whose `claimed_by` is that worker. A job whose worker's heartbeat has expired goes back to `NOT_STARTED` for any worker to take.
3. **Artifact store.** A small `rl_agent/artifacts.py` (`put` / `get` / `url` / `delete`) behind one interface. Model records gain `artifact_uri` next to `location`. Workers upload checkpoints, clips, progress videos and ONNX/MNN files when each is written, and download inputs (warm-start checkpoint, EVAL / EXPORT checkpoint) into a local cache before use. The dashboard serves media through the store (signed URLs or a proxy) and deletes through it. Local jobs can keep writing to `../saved_models` with a `file://` URI, so the car path is unchanged. The backend is still open:

   | Backend | Pros | Cons |
   |---|---|---|
   | Cloudflare R2 | No egress fees (checkpoints and videos get read back a lot); S3 API | Another account and credentials; storage billed monthly |
   | AWS S3 / GCS | Mature; same region as some GPU providers | Egress fees when the dashboard or this PC downloads artifacts |
   | MinIO on this PC | Free; data stays here; S3 API, so it can switch to R2 later | Remote jobs upload over this PC's home uplink; the PC must stay on for remote jobs to start or finish |
   | Hugging Face Storage Buckets | HF Jobs mount them read-write (`-v hf://buckets/<ns>/<bucket>:/artifacts`), so R3 jobs write checkpoints as plain files | Not the S3 API: needs its own backend in `artifacts.py` (`huggingface_hub`); the dashboard must sync or proxy files to serve them |

   Recommendation: write against the S3 API (`boto3`, and `@aws-sdk/client-s3` in the dashboard) so the backend is a config choice. Start with MinIO for development, and choose R2 once real remote runs show the artifact volume. If R3 becomes the main remote path, add an HF Buckets backend instead of copying from bucket to S3.
4. **Remote logs and curves.**
   - Workers append log lines to a capped `job_logs` collection (job id, seq, text). The dashboard's log panel and the console's `tail_trainer_log` read it, falling back to the local file for local jobs.
   - TensorBoard event files sync to the artifact store every N minutes and when the job ends. The dashboard pulls a job's events into `/tmp/tb_compare/<job>` on demand, which is the mechanism the Compare view already uses.
5. **Images instead of bind mounts.** CI builds and pushes `beni_trainer` and `duck_trainer` images to GHCR, tagged with the git SHA. Each image bakes in `rl_agent` at that commit. The job (or worker) records which image ran it. Remote hosts pull by tag; local development keeps the bind mount.

*Done when:* a second `openbeni` worker on this PC, started with a different host name and the store backend, can take a job from the queue alongside the local one; and its checkpoints, clips, logs and curves show up in the dashboard while the car trains.

### Phase R2: Persistent remote mjlab workers (rented GPU)

1. **Provider.** Choose a persistent-pod provider (RunPod, Lambda or Vast.ai) on a single RTX 4090 / L40S / A100-class GPU. mjlab with MuJoCo Warp needs CUDA 12.8 and a recent driver; check the driver version before committing. Compare hourly price against measured iterations per hour for `Beni-Velocity-Flat` on this PC's GPU.
2. **Bootstrap.**
   - A `scripts/remote/up.sh` starts Tailscale and pulls the image at the requested SHA.
   - It then runs the worker with `MONGO_URL`, store credentials, `WORKER_HOST`, and `--robots openbeni,microduck`.
   - It keeps the MuJoCo Warp kernel cache (`/root/.cache/warp`) on the pod's persistent volume, so restarts don't recompile.
3. **Routing.** The New-job form and console `create_job` gain an optional "Run on" field (any / local / a specific host). Curriculum runs can set it for the whole run or per stage.
4. **Cost guard.**
   - The worker exits after an idle timeout with no claimable work, and `up.sh` stops the pod.
   - The dashboard shows each host's uptime and estimated spend from its registered hourly rate.
5. **Verify.** Rerun Phase E steps 2–3 on the remote host while a car job trains locally. Compare its returns and fall rate with the same job run locally (same seed), and check that EVAL and EXPORT of a remote checkpoint work from either host.

### Phase R3: On-demand Hugging Face Jobs (mjlab robots)

Hugging Face Jobs run a Docker image on a chosen GPU, billed per minute while starting or running. Prices as of Nov 2026:
- `h200`: 1× H200 141 GB, 23 vCPU, $5/h, up to `h200x8` at $40/h.
- `rtx-pro-6000`: 96 GB, $2.75/h.
- `l40s`: $1.80/h.

ZeroGPU Spaces are not an option: each GPU call is capped at about 2 minutes, with a daily quota.

1. **Benchmark first.** Run the same `Beni-Velocity-Flat` TRAIN (fixed seed and iteration count) on `h200`, `rtx-pro-6000` and this PC. Raise `num_envs` (8k / 16k / 32k) where memory allows. Record iterations per hour, dollars per 1,000 iterations, and final return. Prefer `rtx-pro-6000` unless the H200 is clearly cheaper per iteration. The RTX also has full graphics, so EGL checkpoint clips render normally. The H200 is a compute GPU with minimal graphics hardware.
2. **Dispatcher.** A `hf-dispatcher` service (or a `curriculum-runner` sibling) watches for `NOT_STARTED` jobs with `run_on: "hf"`.
   - For each one it calls `huggingface_hub.run_job` with the trainer image at the job's git SHA, the flavor, and `--timeout` set from the iteration budget plus a margin (the default is 30 minutes).
   - It passes the artifact bucket volume and secrets (`MONGO_URL`, Tailscale auth key).
   - It stamps the job with `hf_job_id`, flavor and `claimed_by: hf/<job_id>`, and polls the HF job state, so a crashed or timed-out Job marks the queue job `FAILED` with the HF log tail.
3. **Inside the Job.** The container starts Tailscale in userspace mode (no extra privileges) and runs the worker with `--job <id> --exit-when-done`, so it handles exactly that job and exits. It writes artifacts to the mounted bucket and logs through the R1 log path. Cancel and pause from the dashboard map to `cancel_job` on the HF side once the worker has checkpointed.
4. **Fan-out.** Sweeps and curriculum stages queue several `run_on: "hf"` jobs. The dispatcher caps concurrent Jobs and dollars per day, and the dashboard shows running HF Jobs with their cost so far.

*Done when:* a `Beni-Velocity-Flat` TRAIN queued with `run_on: "hf"` runs on HF Jobs while a car job trains locally, and its checkpoints, clips, logs and curves show up in the dashboard. An EVAL and an EXPORT of that checkpoint must also succeed, both on this PC and as a second HF Job.

The car stays local: its Unity rendering needs a graphics-capable GPU, and its speed is limited by the 0.4 s lockstep and CPU-side physics, not by GPU size.

### Risks and open questions

- **Checkpoint size and traffic.** rsl_rl checkpoints are tens of MB, and the car's Reverb checkpoints are far larger (another reason the car stays local). Measure upload cost per job before choosing the store.
- **Mongo over the tailnet.** Change streams (Jobs tab, curriculum runner) work over a VPN, but a remote worker that loses the link mid-job must keep training and catch up on status writes, not fail the job.
- **Reproducibility across GPUs.** MuJoCo Warp results can differ across GPU models and drivers. Record `gpu_name` and the driver version on the job, and compare like with like on the leaderboard.
- **Secrets.** Store credentials and the Tailscale auth key go into the provider's secret store, never into images or job documents.
