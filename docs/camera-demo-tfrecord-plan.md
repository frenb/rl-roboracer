# Camera frames in expert TFRecords (BC and AWAC)

**Status: Phases 1–5 (write, read, metrics, DEMO, TRAIN BC/AWAC) are implemented, and Phase 6 (live check) passed on `donut_camera`.**

A `donut_camera` or `donut_camera_no_rays` DEMO job records the CSI
frame beside the policy vector. A TRAIN job on the same course reads
that corpus for BC and AWAC. `BC_TRAINING_ONLY` on those courses is
still refused.

The running SAC job `6ad99cc8f24f657c7fbdc7eb` is not a source for
this corpus. Its replay buffer holds the learning policy's own
actions, and those frames are never written to a TFRecord. Expert
rows come from a new DEMO job driven by the existing heuristic.

Sibling code:

- `rl_agent/robotaxi.py` `collect_expert_demos` / `write_trajectories_to_file`
- `rl_agent/collect_training_data.py` parse, `convert_tfrecord_to_trajectory`, `bc_pretrain_actor_net`
- `rl_agent/vision/image_to_obs.py` (uint8 rgb8 → float32 `[84,84,3]` in `[0,1]`)
- `rl_agent/vision/camera_networks.py` (the Conv actor AWAC and BC must share)
- `docs/csi-camera-observation-guide.md` §10 (Phase 7, the short version of this)
- `rl_agent/vision/camera_record.py` (Phase 1 encoder)
- `rl_agent/vision/test_camera_record.py`

---

## Phases

### Phase 1 — Write the camera row

**Implemented.** `write_trajectories_to_file` branches on a dict
observation. The policy vector goes in `observation`. The frame goes
in `image` as 21168 uint8 bytes (`vision/camera_record.py`). A flat
observation writes the same features as before, with no `image` key.
DEMO jobs are still refused. Phase 2 reads the field.

Check: `python vision/test_camera_record.py` in sim-controller. It
covers a 31-vector and a 2-vector, the byte length, and a round trip
within 1/255.

### Phase 2 — Read the camera row

**Implemented.** `collect_training_data.camera_demo_dataset(directory,
course_type)` streams one dict per step: `observation` is
`{vector, image}`, and `reward` / `discount` / `next_step_type` are
the values stored in the file. The schema comes from `course_type`
(`donut_camera` is 31, `donut_camera_no_rays` is 2). Images are not
concatenated into one array. A camera file is refused by the flat
reader, and a flat file is refused by the camera reader. A 31-wide
file is refused by the 2-wide course.

`min_accel` defaults to 0.05, the same acceleration floor the vector
loader applies. Pass `None` to keep every row. A camera TRAIN
streams it through `camera_train_datasets`.

Check: `python vision/test_camera_demo_dataset.py` in sim-controller.

### Phase 3 — Demo metrics

**Implemented.** `analyze_demo_metrics.py` reads speed from `vector[0]`
when the file has an `image` feature, for both camera widths. The
31-wide file still has the forward ray at `vector[16]`, so the
crash estimate runs. The 2-wide ablation has no ray, so that estimate
is skipped. Flat donut files still use observation[1].

Check: `python vision/test_analyze_demo_metrics.py` in sim-controller.

### Phase 4 — Accept camera DEMO jobs

**Implemented.** `donut_camera` and `donut_camera_no_rays` are no
longer in `COURSES_WITHOUT_DEMOS`. The fly courses stay there. A
camera DEMO records on its own course and, after the flush, skips
`train_agent`. `BC_TRAINING_ONLY` on a camera course is still refused.
A camera TRAIN with no demo source still starts from scratch. Phase 5
teaches TRAIN to read the new files.

The running trainer loaded the previous code. A DEMO job uses this
path after the next trainer start. Check:
`python vision/test_camera_demo_job.py` in sim-controller. A live
DEMO then writes `/tfrecords/job_<id>/` with 21168-byte images and a
vector length of 31 or 2.

### Phase 5 — BC and AWAC

**Implemented.** A camera TRAIN with a demo source keeps the
experiment design's BC steps, prefill cap, sample ratio, and
`awac_lambda`. Prefill is capped at 10,000 steps. A camera TRAIN
with no demo source still starts from scratch. A demo directory
whose vector length disagrees with the course, including a legacy
file that has no `image` feature, raises before the Unity handshake
with the path and the widths. Prefill and BC read
`camera_train_datasets`, which streams `{vector, image}` and the
stored reward. The running trainer does not have this code until
its next start.

Check: `python vision/test_camera_train_demos.py` in sim-controller.

### Phase 6 — Live check

**Done on `donut_camera`, 2026-10-23.** Checks 1 and 3–6 passed on
live jobs; check 2 is covered by the Phase 2 unit test.

- 500-step DEMO `6adad50f8948903d72304bbe`: 31-wide vectors,
  21168-byte images, one all-zero image (the first step, before the
  first frame), Goal-count v4 rewards.
- Refusals `6adad5b056c5ec72970c73cc` (31-wide demo on
  `donut_camera_no_rays`) and `6adad5b056c5ec72970c73cd` (flat donut
  corpus on `donut_camera`) failed before env spawn.
- Corpus DEMO `6adad6f4f6c617a4f77602ad`: one gym, 12,085 steps,
  19 episodes, 3 zero images, 6 repeated frames.
- TRAIN `6adae96e6e178fedfebbcdee` with experiment design "Camera
  AWAC + BC (10k camera demos, 75k buffer)": BC loss 0.53 to 0.002,
  10,000 demo steps in the protected table, AWAC at lambda 0.5.

Three fixes came out of it:

- DEMO jobs now install the job's reward design. They used to record
  the course default reward while TRAIN installed the design.
- `AwacSacAgent.actor_loss` slices the demo observation with
  `tf.nest`; the camera `{vector, image}` dict raised
  `unhashable type: 'slice'` when the Learner was built.
- `faulthandler` is registered on `SIGUSR1` in `robotaxi.py` and
  `envs.py`. One TRAIN attempt hung in the first `env.reset()` after
  BC, with no reset command reaching ros-server; the retry did not.
  `kill -USR1` on the trainer or a worker dumps its stacks to the log
  if it happens again.

Collect camera demos with one gym. With `demo_num_gyms=2`,
ros-server-1 timed out on every reset and the watchdog restarted the
stack.

---

## 1. What one row contains

The heuristic already steers from `environment.data["car"]`
(`dist_from_traj`, `speed`, `dist_from_goal`). It does not read the
image. `collect_expert_demos` already appends
`next_time_step.observation`, which on a camera course is the dict
the policy sees:

| Course | `vector` | `image` |
|---|---|---|
| `donut_camera` | `(31,)` float32, rays included | `(84, 84, 3)` float32 in `[0, 1]` |
| `donut_camera_no_rays` | `(2,)` float32, speed and sideslip | same image |

Record that dict, plus the action the heuristic sent, the env reward,
the env discount, and `next_step_type` (already written today, used
to mark the last step of an episode).

`DEMO_RECORDING_COURSE` stays unset for both camera courses. Mapping
either of them to `donut` would store the 32-wide vector and drop the
image. Each course records its own policy vector. A 31-wide camera
file must not load into the 2-wide ablation, and the other way around.
The different `vector` length is the same guard the live specs already
use.

Rewards, stuck detection, and the step-cost index stay on the 31-wide
`scene_data_array()` inside the env. They are not read back out of
the TFRecord.

## 2. On-disk layout

Keep the existing files valid. A camera row is a `tf.train.Example`
with these features:

| Feature | Type | Contents |
|---|---|---|
| `action` | float list, length 2 | force, steer. Same as today. |
| `observation` | float list, length 31 or 2 | the policy vector only |
| `image` | bytes, length 21168 | uint8 rgb8, 84×84×3, the frame `image_to_obs` already consumed |
| `reward` | float list, length 1 | the env reward for that step |
| `discount` | float list, length 1 | the env discount for that step |
| `next_step_type` | int64 list, length 1 | already written |

`image` absent means a legacy vector file. The camera parser requires
it. Do not flatten pixels into `observation`.

Store the uint8 frame, not float32 and not JPEG. 100,000 steps are
about 2.1 GB. Float32 would be about 8.5 GB and is what the network
wants in memory, not on disk. JPEG would change the pixels relative
to an online step. On read, divide by 255 and clip to `[0, 1]`, which
is what `image_to_obs` does for an already-84×84 rgb8 buffer. One
conversion, shared with the live env.

`write_trajectories_to_file` is the only writer, and Phase 1 added
the branch. A dict observation writes `observation["vector"]` into
`observation` and `uint8(round(image * 255))` into `image`
(`vision/camera_record.py`). A flat observation still calls
`.numpy().ravel()` and writes no `image` key, so existing donut
corpora are unchanged.

## 3. Read path

`feature_description` stays the flat schema for donut and
`donut_no_hint`. Camera courses get a second schema selected by
`course_type`, not by sniffing the first record.

`convert_tfrecord_to_trajectory` today fills one `(N, 32)` array and
then drops a leading column. It also replaces every reward with the
constant `1` and every discount with `0.99`, ignoring the values in
the file. BC can live with that. AWAC cannot: the critic trains on
the demo transitions, and a constant reward makes those targets
fiction. The camera converter returns a trajectory whose
`observation` is the dict `{vector, image}` and whose `reward` and
`discount` are the stored per-step values.

Do not `np.concatenate` the whole image corpus into one array the way
`robotaxi.main` concatenates vector observations. A 100,000-step
float32 corpus is about 8.5 GB before Reverb copies it. Stream with
`TFRecordDataset`, decode the bytes in a `map`, and apply the
existing acceleration filter (`action[0] >= 0.05`) as a dataset
filter. The bool-mask and `from_tensor_slices` block in `main` stays
on the vector courses. The camera course feeds BC and the Reverb
prefill from the streamed dataset.

`analyze_demo_metrics.py` indexes speed on a flat vector at
observation[1]. Phase 3 points a camera file at `vector[0]` on both
courses when the `image` feature is present. The 31-wide forward ray
stays at `vector[16]`. The 2-wide file has no ray, so the crash
estimate is skipped.

## 4. Turning the refusal off

**Implemented for DEMO.** `donut_camera` and `donut_camera_no_rays`
are out of `COURSES_WITHOUT_DEMOS`. `fly_donut` and `fly_donut_flow`
stay in that set. Their observation is the brain trace, which this
layout does not describe.

`do_job` accepts a DEMO job on either camera course. After the files
are flushed it skips `collect_training_data.train_agent`. That
function builds a 512-wide MLP against a flat dummy env. Camera BC
is the Conv actor inside the TRAIN job (`bc_pretrain_actor_net` on
the network from `camera_networks.py`). `BC_TRAINING_ONLY` on a
camera course is refused with that reason.

`main` keeps `bc_pretrain_steps`, `demo_prefill_count`,
`demo_sample_ratio`, and `awac_lambda` when a camera TRAIN names a
demo directory, and caps prefill at `CAMERA_PREFILL_CAP` (10,000).
A camera TRAIN with no demo source still starts from scratch.

A camera TRAIN must refuse a demo directory whose vector length
disagrees with the course spec (31 vs 2, or a legacy 32-wide file
with no `image` feature). Fail the job at load time with the path
and the widths. Do not slice, pad, or drop columns to make them fit.

## 5. What a TRAIN job does with the rows

Two consumers, both already present for vector courses:

**BC.** `bc_pretrain_actor_net` runs supervised steps on
`actor_net` before SAC. The camera actor is the Conv net already
built for `donut_camera` (`preprocessing_layers` on the image,
Dense 64 on the vector). `BehavioralCloningAgent` only needs
`time_step_spec` to match that dict. No new loss. The pretrain
dataset is the streamed expert set, shuffled and batched. It sees
`{vector, image}` and the expert action.

**AWAC.** `AwacSacAgent` adds a weighted behavior-cloning term on
actions drawn from the demo table. The demo table has to hold the
same dict the online table holds, or the actor graph has two
observation signatures. Prefill inserts streamed trajectories into
the protected demo table the same way vector prefill does today.
`demo_sample_ratio` then mixes those rows into the SAC batches, and
`awac_lambda > 0` turns the extra actor term on.

Reverb stores float32 images. 50,000 prefilled steps are about 4 GB
on top of the online buffer (the live camera job's 75,000-step
buffer is already about 6 GB). Keep the experiment-design prefill
cap, and start the first camera corpus at 10,000 prefilled steps
until a host has finished one BC-plus-AWAC run without swapping.

The online buffer still records the SAC policy's own frames during
TRAIN. Those rows are not copied into the expert TFRecord.

## 6. How to run it, once the code lands

DEMO job, field by field:

- Status `NOT_STARTED`, job type `DEMO`, model type `SacAgent` (unused).
- Course `donut_camera (31-D + 84x84 CSI)` or
  `donut_camera_no_rays (2-D + 84x84 CSI, raycast ablation)`.
  One course per job. The two corpora stay separate.
- Demo transitions: the step budget, same field as a vector DEMO.
- Demo job id empty. Experiment design: trainer defaults. A curriculum
  still ignores the step budget, same as today.
- Reward design: the same goal-count design the TRAIN job uses.
  The recorded reward has to be the reward AWAC's critic will see.
- Gym: a build that contains `CsiFramePublisher`
  (`real2sim2026.10.21-fences-v2` does). Both gym windows up when
  the trainer is launched with two environments.

The log should show `[front_camera] first frame 84x84 encoding=rgb8`
and then flushes under `/tfrecords/job_<id>/`. A run that prints
`Front camera timed out` on every step is writing last-good or zero
images and should be discarded.

TRAIN job on the same course:

- Demo job id: the DEMO job above. No other demo id mixed in.
- Experiment design with BC pretrain steps and `awac_lambda > 0`.
  The vector design "AWAC + BC + reward_scale1 (no curriculum)" is
  the shape to copy; its demo file is the wrong corpus, so the new
  design points at the camera DEMO job only.
- Same reward design, same gym, same iteration budget you would have
  used for the from-scratch camera run.

EVAL stays on the course the checkpoint was trained on. A 31-wide
camera policy on the 2-wide course raises `EvalSpecMismatchError`,
as it does today.

## 7. Checks before calling it done

1. A short DEMO (a few hundred steps) writes files whose `image`
   bytes are 21168 long and whose `observation` length is 31 or 2
   matching the course.
2. A round trip through the camera parser reproduces the float32
   image within 1/255 of `image_to_obs` on the same frame.
3. Loading that directory from a TRAIN job of the other camera
   course fails before any gradient step.
4. Loading a legacy 32-wide donut directory from a camera TRAIN
   fails the same way.
5. BC logs a finite loss on the Conv actor, and the first SAC
   iterations run with `awac_lambda` at the design's value rather
   than forced to 0.
6. `analyze_demo_metrics` reports speed from `vector[0]`.

## 8. Out of scope

- Mining the in-progress SAC buffer or `unity/CsiFrameDumps/*.png`
  into an expert corpus. Those frames are not paired with heuristic
  actions.
- JPEG, a new ROS message, or pixels on `CarSceneData`.
- Changing the live online-buffer layout. It already stores the dict.
- BC or AWAC for `fly_donut` / `fly_donut_flow`.
- Domain randomization and the Jetson TFLite path (guide §11–12).
