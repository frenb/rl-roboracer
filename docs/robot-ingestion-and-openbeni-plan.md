# Standardizing Robot Ingestion and Importing OpenBeni

**Status:** Planned (not yet started). Designed against `microduck-phase0` (`roboracer` + `microduck`) and [`Mondo-Robotics/open-beni`](https://github.com/Mondo-Robotics/open-beni).

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
