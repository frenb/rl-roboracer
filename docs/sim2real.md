# Sim2real

How the simulator mirrors the physical JetRacer, and the observation design that lets a policy transfer.

The physical car is the
[Waveshare JetRacer Professional Version ROS AI Kit](https://www.amazon.com/JetRacer-Professional-ROS-Kit-Controllers/dp/B0BLH9Y3VN)
(dual controllers, RPLiDAR, IMX219 CSI, **Jetson Nano 4 GB** — not the 2 GB
board). On-car ROS notes live in [`jetson.md`](hardware/jetson.md). Deploying a
SavedModel onto Melodic `/cmd_vel` is still a plan
([`docs/tf-model-jetson-deployment-guide.md`](tf-model-jetson-deployment-guide.md)).
Publishing CSI pixels into the trainer (`camera/front`, a `donut_camera`
course) is still a plan
([`docs/csi-camera-observation-guide.md`](csi-camera-observation-guide.md)).

Policies that can transfer are **`donut_no_hint`** (31-D rays / speed /
sideslip — no Unity goal angle). See
[The `donut_no_hint` course](#the-donut_no_hint-course-hint-free-observation-for-simreal)
below.

## Unity gym changes (visual / physics stand-in)

These are what the play scene does today to look and collide more like
that kit. **P** switches Main Camera ↔ CSI (`CameraViewSwitcher`).

| Change | Why |
|---|---|
| `JetRacer_Physics_wcourse` prefab (mesh + rigidbody) is the `carPrefab` in every course scene | Same silhouette, wheelbase, and collision as the kit |
| Play scene `Assets/Scenes/w-course-jetracer.unity` (enabled in `EditorBuildSettings`) | Kit `Tarmac_c` tiles on layer **Road** (6) |
| `Goal` treats `JetRacer_Physics` as the car (walks parents + `attachedRigidbody`); goal meshes on the Road layer | Hits register; CSI can see the spheres |
| `JetRacerCsiCamera` on the car: 640×480, aspect 4:3, skybox clear, near **0.05 m**, far **200 m**, no HDR/MSAA | Match `gscam` **640×480** and `config/camera_calibration/cam_640x480.yaml` size |
| Yaml `fx` / `fy` stored on `JetRacerCsiIntrinsics`; **visual VFOV opened to 80°** (yaml `fx` is ~62°) | Strict `fy`/`fx` left too little tarmac ahead of the hood |
| Prefab mount **`(0, 1.82, 1.70)` m**, **~5°** pitch down | Stalk-tip pose looked **into the chassis** (hull top ≈ y 1.72, nose ≈ z 2.33) |
| `cullingMask = ~0` | Prefab mask 63 hid layer 6, so P-view showed Default grey ground instead of asphalt |
| No plumb_bob **D** warp | Same-size undistort punched black fisheye corners; undistort-on-Nano or pad-then-D is later |
| CSI hides `Sky_Dome` (finite mesh bowl, not a skybox) | Nested `Example_Track_A` + the bowl caused “bubbling” / horizon clipping |
| CSI defaults: trajectory / goals / rays **off**; **T / G / R** toggle those on CSI only | Real CSI has no debug overlays |
| P-view undocks the render texture, letterboxes 4:3, does not blit via OnGUI | Game view was “Display 1 No cameras rendering” while the RT was the only target |
| `CsiFramePublisher`: one 84×84 `rgb8` per `cmd_id` on `camera/front` | Vision Phase 1–2. HUD `csi cmd …`; PNGs in `unity/CsiFrameDumps/`; **F** dumps. `unity_node.py` lists the topic ([guide](csi-camera-observation-guide.md#4-phase-1--unity-one-csi-frame-per-applyforce)) |

Still **not** on the learning loop: CSI frames are for the operator (P-view).
`car_scene_data` is still the 31/32-D vector. Do not parent dressing under
`Track` (`TrackGenerator` destroys it). Decorative meshes named `Curb` /
`Rail` on Default become crash geometry.

## The `donut_no_hint` course (hint-free observation for sim→real)

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

### Selecting the course per job

The dashboard **New-job** form has a **Course** selector (`trainer default`,
`donut`, `donut_no_hint`, `donut_camera`, `donut_camera_no_rays`, `fly_donut`).
The choice is stored as `course_type` on the job
document and threaded through the whole pipeline:

- `do_job()` resolves `_job_course_type` (job doc → `ROBOTAXI_COURSE_TYPE`
  env → `donut`), then points the demo/BC pipeline at the matching width via
  `collect_training_data.set_observation_size(...)` and passes
  `course_type=...` to every `make_env()` (DEMO/EVAL) and to `main()` (TRAIN).
  Courses in `COURSES_WITHOUT_DEMOS` (the two camera courses and `fly_donut`)
  skip that width call and are refused DEMO / BC_TRAINING_ONLY outright, since
  no expert corpus exists at their observation shape.
- `main()` re-resolves the course (so direct callers work too) and builds the
  collect/eval envs and the actor/critic network at the right width.
- The Jobs table shows a **Course** column (legacy jobs are backfilled to
  `donut` on server startup); model records are stamped with `course_type`, and
  the Models tab's **Compat** check compares against the model's own course
  width. An EVAL launched from the Models tab inherits the model's course.

`trainer default` means "use the `ROBOTAXI_COURSE_TYPE` env var / `donut`" and
is the fallback for anything unmapped.

### One 32-wide demo corpus, reused everywhere

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

### Course-aware default demo source

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

If no source resolves at all on a **vector** course (`skip_default_demo`
with no `demo_job_ids`), the job **fails fast** with an actionable message
(stamped as `FAILED` + `eval_error` by `run_jobs_loop`) rather than silently
loading incompatible data.

`donut_camera` maps to `None` on purpose and does **not** fail: a TRAIN
job on that course is **SAC from scratch** (CNN on the 84×84 CSI image +
31-D vector). There is no expert TFRecord / BC / AWAC yet — those records
are still flat 32-D vectors. See
[CSI camera observation guide](csi-camera-observation-guide.md#train-donut_camera).
The New-job modal explains this when Job type is TRAIN and Course is
`donut_camera`. Unity must publish `camera/front` (Editor Play or a
promoted gym). DEMO and `BC_TRAINING_ONLY` on this course are refused.

### Compatibility / gotchas

- **Checkpoints are not cross-compatible.** A 32-wide (`donut`) policy can't
  resume/eval as 31-wide (`donut_no_hint`) or vice-versa — the network input
  dim differs. Start `donut_no_hint` as a fresh TRAIN.
- The per-job `set_observation_size()` mutates module-level state in
  `collect_training_data`; this is safe **only** because jobs run strictly
  sequentially in the singleton trainer. Concurrent multi-course jobs would
  need per-call plumbing instead.
- `analyze_demo_metrics.py` derives `SPEED_IDX` / `FWD_IDX` from
  `_OBS_DROP_LEADING`, so it reads correct indices for either course.
