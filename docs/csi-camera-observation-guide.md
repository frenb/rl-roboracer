# CSI camera → policy observation

Plan only. This document tells you how to publish **Unity CSI frames** onto
the existing ROS-TCP → gRPC plane, turn them into a **dict observation**
the SAC trainer can learn from, and reuse the same preprocess at **sim
EVAL** and later **Jetson inference**. It does **not** implement those
nodes.

Sibling docs:

- Gym visual / physics stand-in that already exists:
  [README → Sim2real](../README.md#sim2real).
- On-car bringup and leftover camera notes:
  [`jetson.md`](../jetson.md#sim-camera-mirror-the-jetracer-csi-feed-into-unity-gyms-and-the-model).
- SavedModel → Melodic `/cmd_vel` (vector policy, no pixels):
  [`tf-model-jetson-deployment-guide.md`](tf-model-jetson-deployment-guide.md).

**Do not widen `donut_no_hint`.** A 31-D SavedModel cannot load a dict
spec. Add a new `course_type=donut_camera`. Until this lands, CSI frames
are operator-only (P-view); `car_scene_data` is still the 31/32-D vector.

---

## 1. What exists today

```
Unity SceneDataPublisher
  -- ROS-TCP :10000+i -->  ros-server  unity_node.py
  -- gRPC JSON :50051+i -->  RobotApi
  --> course.scene_data_array()   # float32[31|32]
  --> TimeStep.observation
```

| Piece | State |
|---|---|
| `JetRacerCsiCamera` + 640×480 RT, 80° VFOV, mount `(0, 1.82, 1.70)`, 5° down | Implemented. P-view only. |
| `unity_node.py` `camera/overhead` | Listed. Overhead leftover from the arm gym. |
| `RobotApi.Subscribe('camera/overhead')` + `GetOverheadCameraFrame()` | Wired. Nothing in `DonutCourse` / actor / critic reads it. |
| `RobotApi.Subscribe('camera/front')` + `GetFrontCameraFrame(cmd_id)` | **Phase 3 done.** Wait-on-`header.seq`. Not called by `donut` / `donut_no_hint`. |
| `SimController` `CameraPublisher` | Commented out. Wrong viewpoint (overhead). |
| `CsiFramePublisher` | **Phase 1 done.** One 84×84 `rgb8` per `cmd_id` on `camera/front`. |
| `utility.frame_to_tensor` | Assumes **RGBA** and strips every 4th byte. Wrong for CSI `rgb8`. |
| `vision.image_to_obs` | **Phase 4 done.** float32 `[84,84,3]` in `[0,1]`. Gym: `undistort=False`. |
| Actor / critic | MLP on 31/32-D. **Phase 6:** `donut_camera` uses Conv-SAC (`camera_networks.py`). |
| Demo TFRecords / AWAC / BC | Vector-only (`FULL_OBSERVATION_SIZE = 32`). |

The trainer **never opens a ROS socket**. Frames must follow the same hops
as `car_scene_data`. The Jetson already publishes `/csi_cam_0/image_raw`
(`sensor_msgs/Image`, 640×480 @ ~20 Hz) for Foxglove. That name does
**not** exist inside `ros-server`.

---

## 2. Contract (decide once)

| Item | Choice |
|---|---|
| Sim topic | **`camera/front`**, type `niryo_moveit/Camera` (`sensor_msgs/Image frame`). Do not reuse `camera/overhead`. |
| Jetson topic | **`/csi_cam_0/image_raw`**, type `sensor_msgs/Image`. Nano never speaks `niryo_moveit`. |
| When to capture | **Once per env step**, same `cmd_id` as `car_scene_data`. Not 20 Hz. |
| Net size | **84×84×3** float32 in `[0, 1]` to start (or 160×120). Render 640×480 in Unity; downsample **before** gRPC. |
| Distortion | Leave Unity pinhole. **`cv2.undistort` on the Nano** with `config/camera_calibration/cam_640x480.yaml` so both sides look pinhole. |
| Sync | Stamp image `header.seq` = `cmd_id` (or wait for the first frame after `FORCE_APPLIED` for that id). Uncorrelated frames desync rays vs pixels. |
| Pixels on `CarSceneData` / `Sphere`? | **No.** Breaks every existing parser. |

Keep `camera/overhead` in the routing table so leftover subscribers do not
break. Never pipe it into the driving policy.

---

## 3. Target data plane

```
Unity JetRacerCsiCamera  (640×480 RT)
    CsiFramePublisher    (downsample → rgb8, header.seq = cmd_id)
    -- ROS-TCP :10000+i -->  ros-server  unity_node.py  camera/front
    -- gRPC JSON :50051+i -->  RobotApi.GetFrontCameraFrame(cmd_id)
    --> image_to_obs()           # float32 [84,84,3] in [0,1]
    --> {vector: float32[31], image: ...}
    --> tf-agents TimeStep
```

gRPC JSON (`niryo_moveit/Camera`):

```json
{
  "frame": {
    "header": {"seq": 123, "stamp": {}, "frame_id": "camera_visual"},
    "height": 84,
    "width": 84,
    "encoding": "rgb8",
    "is_bigendian": 0,
    "step": 252,
    "data": "<base64 of height*step bytes>"
  }
}
```

**Bandwidth:** 84×84×3 ≈ 21 KB raw (~28 KB base64) per step per actor.
Four actors is fine. Full 640×480 through `virtual_endpoint` (~900 KB raw,
worse as JSON base64) will stall. Compress (`jpeg` +
`sensor_msgs/CompressedImage`) only if you must send full res; then decode
with `tf.io.decode_jpeg` / OpenCV.

**Jetson (later):** the policy node does **not** subscribe to
`camera/front`. It subscribes to `/csi_cam_0/image_raw`, runs the **same**
`image_to_obs()`, and feeds the CNN. `ros-server` / ROS-TCP never run on
the Nano.

---

## 4. Phase 1 — Unity: one CSI frame per `ApplyForce`

**Status: implemented** (`CsiFramePublisher.cs`, wired from
`SceneDataPublisher` / `SimController`). `ros-server` lists
`camera/front` (Phase 2). Assess on the Unity side (below); the
destination error is gone once the ROS containers have the new table.

What landed:

1. **`CsiFramePublisher`** auto-added from `SimController.Start()`. Does
   not revive the overhead `CameraPublisher`.
2. After `WaitForEndOfFrame`, dedicated 640×480 RT → `Graphics.Blit` to
   **84×84** → `rgb8` `niryo_moveit/Camera`:
   - `header.frame_id = camera_visual`
   - `header.seq = cmd_id`
   - `step = 252`
3. `ROSConnection.Send("camera/front", msg)`.
4. One capture per **new** `cmd_id`, not the 20 Hz `car_scene_data`
   loop. Reset now calls `UpdateWorldRefs(af)` so the first frame shares
   the reset `cmd_id`.
5. Dedicated publish RT — P-view undocking the display RT does not stop
   capture. `cam.Render()` still hits `CameraViewSwitcher` overlay hides.
6. Editor Play is enough to **assess**. A promoted build is required
   before `Start-Stack` actors see this.

### How to assess Phase 1

Do this in the Unity Editor on `w-course-jetracer` (Play). No Docker
change, no `unity_node.py` change.

1. Press Play. Press **P** so the Game view is the CSI stalk (optional;
   publish runs either way).
2. Wait for a reset / first `car_scene_data`, or press **F** to force a
   dump of the current CSI frame.
3. **Console** must contain a line like:
   `[CsiFramePublisher] ok topic=camera/front cmd_id=… 84x84 encoding=rgb8 bytes=21168 seq=… dump=…`
   Fail: `csi: no JetRacerCsiCamera` (car not spawned) or silence (script
   not on `SimController`).
4. **HUD** (H if hidden) shows `csi cmd <id> 84x84 #<n>` after the first
   ok. Fail: stuck on `csi: waiting` / `csi: ready`.
5. Open the PNG. Editor writes
   `unity/CsiFrameDumps/csi_cmd<id>_<timestamp>.png` (first 3 auto, then
   F). Player writes under `Application.persistentDataPath/CsiFrameDumps/`.
   - Pass: 84×84, tarmac / hood / horizon roughly like P-view (blocky is
     expected). Not black, not a grey Default ground plane, not upside
     down.
   - Compare to a Foxglove grab of `/csi_cam_0/image_raw` — composition
     should rhyme; resolution will not.
6. Drive or let the trainer step: `#` on the HUD increments **once per
   new `cmd_id`**, not at 20 Hz. Fail: `#` climbing every 100 ms.

`unity/CsiFrameDumps/` is gitignored. Rebuild + `PromoteLatestBuild.ps1`
only when you want the standalone gym to do this.

---

## 5. Phase 2 — ROS-TCP table (`ros-server`)

**Status: implemented.** `unity_node.py` now has:

```python
'camera/front': RosPublisher('camera/front', Camera),  # niryo_moveit/Camera
```

The table is **static** and baked into the image. After editing, copy
into running containers **or** rebuild `docker_ros-server:thin` and
`docker compose … up -d` the `ros-server*` services. `compose/scale.yml`
uses the same image for `ros-server-1..3`.

No new ROS message type. Do not expect `/csi_cam_0/image_raw` inside
`ros-server`. `RobotApi` subscribes on gRPC (Phase 3); that path does
not use this table.

### How to assess Phase 2

1. Restart Unity Play (or the gym) after the ROS containers reload.
2. **Console must not** show
   `Topic/service destination 'camera/front' is not defined!`
   Fail: containers still run the old `unity_node.py` (no restart / no
   copy).
3. From a ros-server:

   ```powershell
   docker compose -f docker-compose.yml -f compose/scale.yml exec ros-server `
     bash -c 'source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && rostopic list | grep camera'
   ```

   Pass: both `/camera/front` and `/camera/overhead`.
4. Press **F** in Unity, then:

   ```powershell
   docker compose -f docker-compose.yml -f compose/scale.yml exec ros-server `
     bash -c 'source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && rostopic echo -n1 /camera/front'
   ```

   Pass: `height: 84`, `width: 84`, `encoding: rgb8`, `step: 252`,
   `frame_id: camera_visual`. Fail: hang (no publish) or 640×480 (forgot
   downsample).

---

## 6. Phase 3 — `RobotApi`: subscribe, decode, sync

**Status: implemented** (`rl_agent/api.py`). `donut` / `donut_no_hint`
do **not** call `GetFrontCameraFrame` — they stay on the 31-D vector.
Phase 5 (`donut_camera`) is the first consumer.

What landed:

1. `Initialize()` also
   `Subscribe('camera/front', 'niryo_moveit/Camera', _on_front_camera_frame)`.
2. `latest_front_camera_frame` + `front_camera_frames[cmd_id]` (last 16)
   + `front_camera_events[cmd_id]` (mirror `scene_data_events`).
   `header.seq` is the `cmd_id`.
3. `GetFrontCameraFrame(cmd_id)` / `GetFrontCameraFrameBlocking`. Call
   **after** `DoApplyForce`’s scene-data wait (Unity publishes CSI at the
   end of that same `Publish()`). `DoApplyForce` itself does **not** wait
   on the image, so current training is unchanged.
4. Timeout: `front_camera_timeouts` (2 s). Last good image, else 84×84
   `rgb8` zeros + a log line. Does not raise into the actor.
5. Do **not** use `GetOverheadCameraFrame` for driving.

First live frame also prints
`[front_camera] first frame 84x84 encoding=rgb8 seq=…` on the actor
stdout (any job that constructs a `RobotApi`).

### How to assess Phase 3

Unity Editor Play (or the gym) plus a running `ros-server`. Then:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  sh -c "cd /python_ws/src && python check_front_camera.py"
```

Press Play (or **F**) if the first `cmd_id=0` already went out. Pass:
`OK - gRPC sees 84x84 rgb8 matching Unity cmd_id.` Fail: `no camera/front`
(Unity not publishing / wrong ros-server) or zeros (timeout fallback).
Actor 0: `python check_front_camera.py ros-server-0:50051`.

---

## 7. Phase 4 — Python image libraries

**Status: implemented** (`rl_agent/vision/image_to_obs.py`). `donut` /
`donut_no_hint` still do not call it — Phase 5 packs the dict obs.

`sim-controller` already had `tensorflow==2.7` and `numpy`. Decode and
resize live in the shared module (numpy + `cv2.resize` `INTER_AREA`,
same intent as `tf.image.resize` `AREA`) so the Jetson copy does not
need TF:

```python
def image_to_obs(msg_or_uint8, encoding=None, src_hw=None, dst_hw=(84, 84),
                 undistort=False, K=None, D=None) -> np.ndarray:
    """float32 [H, W, 3] in [0, 1], RGB. rgb8 / rgba8 / bgr8."""
```

**`opencv-python-headless==4.6.0.66`** is pinned in
`docker/ros_server/python_ws/requirements.txt` for `cv2.undistort` with
`K` / `D` from `config/camera_calibration/cam_640x480.yaml`. Gym path:
`undistort=False` (Unity is pinhole). Nano: undistort at source size,
then downsample. `load_camera_calibration(path)` reads that yaml.

Do **not** add Pillow. Prefer Unity-side downsample over
`sensor_msgs/CompressedImage` for v1. No ImageNet mean/std unless
**both** sim and real use it.

`utility.frame_to_tensor` stays for the overhead leftover; do not reuse
it for CSI.

The bind-mount of `./rl_agent` is enough for the module. OpenCV is in
the image only after `pip install` (running container) or a
`sim_controller:latest` rebuild.

**Jetson:** Melodic stock nodes are **Python 2**. Do not import
`rl_agent` on the Nano. Copy `image_to_obs.py` into a **Py3** sidecar
(or TFLite runtime) that subscribes to `/csi_cam_0/image_raw`. The car
typically already has `cv2`. `niryo_moveit` msgs are **not** installed
there.

### How to assess Phase 4

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  sh -c "cd /python_ws/src && python vision/test_image_to_obs.py"
```

Pass: `N passed, 0 failed` (RGB ramp pixels, rgba8 drop-alpha, `D=0`
undistort, yaml `K`/`D`). Fail: `ModuleNotFoundError: cv2` — OpenCV is
not in this container yet:

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  pip install opencv-python-headless==4.6.0.66
```

That install dies when the container is recreated. Persist with
`docker compose … build sim-controller`.

---

## 8. Phase 5 — Observation spec

**Status: implemented.** New `course_type=donut_camera`
(`DonutCourseCamera`, subclass of `DonutCourseNoHint`). `donut` /
`donut_no_hint` TimeSteps are still a flat vector. TRAIN / EVAL on
this course are Phase 6 (Conv-SAC). DEMO / BC are still refused
until Phase 7.

```python
observation_spec = {
    "vector": BoundedArraySpec(shape=(31,), ...),   # same mins/maxs as no_hint
    "image": BoundedArraySpec(shape=(84, 84, 3), dtype=float32, 0..1),
}
```

`scene_data_array()` stays the 31-D vector. Rewards / stuck / `data_arr[7]`
are unchanged. `RobotaxiEnv` packs the dict after `DoApplyForce` (and on
reset) via `GetFrontCameraFrame` + `image_to_obs`. First pack logs
`[donut_camera] obs vector(31,) image(84, 84, 3)`.

`COURSE_OBS_KIND["donut_camera"] = "dict"`.
`apply_course_observation_size` skips `set_observation_size` for dict
courses. The dashboard Course selector includes `donut_camera`
(Phase 6).

**Do not flatten** `31 + 84*84*3` into one MLP vector.

### How to assess Phase 5

Unity Play (or the gym). Stop the trainer if it owns the same ros-server.

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  sh -c "cd /python_ws/src && python check_donut_camera.py"
```

Pass: `OK - donut_camera TimeStep is {vector: (31,), image: (84, 84, 3)}.`
`--spec-only` checks the spec without Unity. Fail: all-zero image (no CSI
/ timeout fallback) or a fight with a live TRAIN actor.

---

## 9. Phase 6 — Networks (training)

**Status: implemented.** `course_type == "donut_camera"` builds Conv-SAC
in `rl_agent/vision/camera_networks.py`. 31/32-D jobs stay MLP.

- `image` → Conv2D 32 / 64 / 64 (Nature-DQN 8/4, 4/2, 3/1), ReLU,
  flatten, Dense 256
- `vector` → Dense 64
- concat → existing `fc_layer_params` / `TanhNormalProjectionNetwork`
- Critic is `DictObsCriticNetwork` (stock DDPG `CriticNetwork` only
  uses `nest.flatten(obs)[0]`)

### TRAIN (donut_camera)

A New-job **TRAIN** with Course `donut_camera` is **not** the same
loop as `donut` / `donut_no_hint`.

- **What it learns from.** Online SAC only. The actor sees
  `{vector: (31,), image: (84, 84, 3)}` each step. Rewards and
  termination still use the 31-D vector (rays, speed, crash). There
  is no expert TFRecord, no BC pretrain, and no AWAC — those records
  are still 32-D vectors (Phase 7).
- **What to set in the form.** Job type TRAIN, Course
  `donut_camera`. Leave Demo job ID empty. Experiment-design BC /
  demo-prefill knobs are forced off. Reward design still applies
  (vector-only formulas).
- **What Unity must do.** Each actor publishes `camera/front` once
  per `cmd_id` (Editor Play, or a gym promoted after Phase 1). A
  standalone build without `CsiFramePublisher` yields zeros every
  step and the policy cannot learn from pixels.
- **What will fail.** DEMO / `BC_TRAINING_ONLY` on this course
  (refused). Loading a 31-D or 32-D checkpoint. EVAL of a camera
  SavedModel on `donut` / `donut_no_hint` (and the reverse).

The New-job modal repeats this when TRAIN + `donut_camera` is
selected. Reverb stores dict obs (~21 KB vs 124 bytes) — watch
buffer RAM.

### How to assess Phase 6

1. Restart the trainer so it loads this `robotaxi.py`.
2. New job → TRAIN → Course `donut_camera` (no demo required).
3. Trainer log: `building donut_camera Conv-SAC` and
   `SAC from scratch`. Fail: MLP shape error or
   `Front camera timed out` on every step (gym has no
   `CsiFramePublisher`).
4. EVAL a camera SavedModel on `donut_camera` only. A 31-D model on
   this course (or the reverse) must raise `EvalSpecMismatchError`.

---

## 10. Phase 7 — Collateral that breaks if you only change the spec

| Area | Why it matters |
|---|---|
| TFRecord / AWAC / BC | `collect_training_data.feature_description` is a flat `FixedLenFeature([32])`. Vision jobs need a new layout (`image` as `FixedLenFeature([84,84,3])` or JPEG bytes) **or skip expert BC** for the first TRAIN (SAC from scratch / no AWAC). |
| DEMO jobs | Heuristic drive does not need the image to steer, but BC later needs the dict recorded. `DEMO_RECORDING_COURSE` must **not** map `donut_camera` → `donut` (that drops the image). |
| `analyze_demo_metrics` | Uses `SPEED_IDX` on a 1-D vector. Point it at `obs["vector"]`. |
| Rewards / stuck / curriculum | Keep using `data_arr` / world pose. Do not index into the image. |
| `robotaxi_env._step` | `data_arr[7]` is fine **if** `data_arr` stays the vector. |
| Rollout viz | Still vector actions. No change. |
| `run_policy` / EVAL | `policy.action(time_step)` must get a dict `TimeStep`. Each actor has its own frame. |
| `write_trajectories_to_file` | Today writes a flat observation. Branch for dict or disable demo write on camera jobs. |

---

## 11. Phase 8 — Inference

**Sim EVAL (same stack):** `load_saved_model` + `run_policy` on a
`donut_camera` env. Greedy SavedModel inside `sim-controller`. No new
ROS topic. This is the first inference gate.

**Car (after the vector deploy guide):**

```
/csi_cam_0/image_raw  +  /scan (or onboard ray stand-in)  +  /odom_raw
        --> image_to_obs() + 31-D vector builder
        --> same SavedModel / TFLite
        --> geometry_msgs/Twist on /cmd_vel
```

Not `camera/front`, not gRPC, not ROS-TCP. Safety: do not fight `joy` /
`explore` on `/cmd_vel`. The Nano cannot comfortably run full Conv-SAC
in tf-agents — plan **TFLite** or a laptop sidecar first. See
[`tf-model-jetson-deployment-guide.md`](tf-model-jetson-deployment-guide.md)
for the vector half of that adapter.

---

## 12. Suggested build order

1. Unity `CsiFramePublisher` — one downsampled `rgb8` per `cmd_id`;
   screenshot vs Foxglove `/csi_cam_0/image_raw`.
2. `unity_node.py` + rebuild `ros-server`; confirm `camera/front` on the
   wire.
3. `RobotApi` subscribe + wait-on-`cmd_id`; `image_to_obs` + unit test.
4. `donut_camera` dict spec + env pack; **log shapes only** (no TRAIN).
5. CNN `preprocessing_layers`; TRAIN from scratch; skip BC/AWAC until
   records exist.
6. Dashboard course + Compat; EVAL in sim.
7. Domain randomization (exposure, noise, ±5° mount) after a policy
   drives in sim.
8. Jetson `image_to_obs` + TFLite / sidecar; then `/cmd_vel`.

---

## 13. What not to do

- Pipe `camera/overhead` into the driving policy (wrong viewpoint).
- Stream 640×480 @ 20 Hz through `virtual_endpoint`.
- Put pixels on `CarSceneData` / `Sphere.msg`.
- Flatten the image into the 31-D MLP.
- Assume `/csi_cam_0/image_raw` exists in Docker, or `niryo_moveit/Camera`
  exists on Melodic.
- Load a `donut` / `donut_no_hint` checkpoint into a `donut_camera` env
  (or the reverse).
- Run a vision policy on `/cmd_vel` while `explore_foxglove` or `joy` is
  also publishing.
- Install a Linux NVIDIA driver inside WSL “for OpenCV”.
- Ablate the raycasts by zeroing them in place (see §14).

---

## 14. Raycast ablation — `donut_camera_no_rays`

**Status: implemented.** Ablation arm for the question *"can the policy hit
`donut_camera` performance without the raycasts?"* `DonutCourseCameraNoRays`
subclasses `DonutCourseCamera` and narrows only the policy-visible vector:

```python
observation_spec = {
    "vector": BoundedArraySpec(shape=(2,), ...),    # speed, sideslip
    "image": BoundedArraySpec(shape=(84, 84, 3), dtype=float32, 0..1),
}
```

The split that makes this safe is `Course.policy_vector(data_arr)`. It is the
identity on every other course; here it returns `data_arr[:2]`.
`scene_data_array()` is deliberately **not** overridden — it stays 31-D, so
rewards, `has_failed()`, the curriculum, and `_step`'s `data_arr[7]` step cost
index it exactly as before. `RobotaxiEnv._pack_observation` puts
`course.policy_vector(vec)` on the TimeStep and nothing else changes. The first
pack logs `[donut_camera_no_rays] obs vector(2,) (of scene(31,)) image(84, 84, 3)`.

`_dict_obs` is now derived from `isinstance(course.observation_spec, dict)`
rather than a hardcoded course name, so a future dict course only has to declare
its spec.

The 29 ray slots are **dropped, not zeroed**. A 31-D vector of mostly zeros
would produce a spec identical to `donut_camera`, let checkpoints cross-load
between the two arms, and quietly invalidate the comparison. The differing
`vector` shape is what makes Compat / `EvalSpecMismatchError` fail closed.

`COURSE_OBS_KIND["donut_camera_no_rays"] = "dict"` and
`COURSE_DEFAULT_DEMO_JOB_IDS["donut_camera_no_rays"] = None`. That `None` is
load-bearing: an unmapped course falls back to the donut default and would hand
this dict-obs job the 32-wide vector demo corpus.

### Running the comparison

Arm 1 is your existing `donut_camera` TRAIN + EVAL records, untouched. For arm
2, create a TRAIN job on `donut_camera_no_rays` with the **same reward design,
experiment design, and iteration budget**, then EVAL from the Models tab (which
builds the env from the model's own `course_type`) with the same preset. Compare
goals/ep, `AverageReturn`, crash/stuck rate, and mean speed.

### How to assess

```powershell
docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller `
  sh -c "cd /python_ws/src && python check_donut_camera_no_rays.py"
```

Pass: `OK - donut_camera_no_rays TimeStep is {vector: (2,), image: (84, 84, 3)}.`
`--spec-only` covers the spec plus an offline packing check (stubbed camera, no
Unity) that asserts `donut_camera` still packs 31-D, the ablation packs 2-D, and
`donut_no_hint` still gets a plain vector. A `(31,)` vector here means the
raycasts leaked back in and the arm is invalid.
