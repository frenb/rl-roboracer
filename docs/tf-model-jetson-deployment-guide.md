# TF model Jetson deployment guide

Plan only. This document tells you how to take a **TensorFlow SavedModel** trained in this repo and run it on the **WaveShare JetRacer ROS AI Kit** (Jetson Nano, ROS Melodic). It does **not** implement the bridge.

The two stacks were never designed as one system. Training already has a ROS graph, but it is a **custom gym graph** inside the `ros-server` Docker container (`niryo_moveit` messages, ROS-TCP to Unity, gRPC to the trainer). The car speaks **standard Jetson topics** (`/scan`, `/odom_raw`, `/cmd_vel`) on ROS Melodic. Those graphs do not share message types. Deployment is a **new adapter** that rebuilds the 31-D observation from Jetson sensors and writes Twist — it does **not** reuse `car_scene_data` / `sim_command` on the Nano.

---

## 1. What you are deploying

| Item | Sim (Unity + `ros-server` + `sim-controller`) | Jetson (OOTB JetRacer) |
|---|---|---|
| OS / ROS | `ros-server` = **ROS Noetic** (Python 3) in Docker; trainer is a separate Py3 GPU container | Ubuntu 18.04, **ROS Melodic**, **Python 2** for stock nodes |
| Policy format | `tf.saved_model` from `PolicySaver(GreedyPolicy)`, loaded **inside `sim-controller`**, never as a ROS node | Nothing loads SavedModels today |
| Perception on the wire | Unity publishes `niryo_moveit/CarSceneData` on `car_scene_data` | RPLiDAR **`/scan`**, wheel **`/odom_raw`**, IMU **`/imu`** |
| Command on the wire | Trainer publishes `niryo_moveit/SimCommand` on `sim_command` | **`geometry_msgs/Twist`** on **`/cmd_vel`**: `linear.x` m/s, `angular.z` steer angle (TEB uses `cmd_angle_instead_rotvel`) |
| Time | One **blocking** env step per action (`DoApplyForce` waits on `cmd_id`) | Async 10–20 Hz sensors, 20–50 Hz cmd loop |
| Localization | Unity world pose (not in the policy obs for `donut_no_hint`) | `odom` TF, optional gmapping `/map` |
| Safety | Unity crash + stuck detector | `lidar_estop`, `invert_linear`, joy/explore fighting `/cmd_vel` |

**Use `donut_no_hint` models only** (31-D). A classic `donut` (32-D) model’s first feature is **angle-to-next-Unity-goal**. That signal does not exist on the JetRacer. Loading a 32-D checkpoint on a 31-D bridge will either fail the spec check or silently drive on garbage.

Artifacts live at (host `../saved_models`, bind-mounted into **both** `ros-server` and `sim-controller` as `/saved_models`):

```
/saved_models/robotaxi/SacAgent/<version>/
/saved_models/robotaxi/SacAgent/<version>_step_<train_step>/
```

Confirm on the Models tab: `course_type=donut_no_hint`, observation shape `(31,)`, action shape `(2,)`. EVAL on that checkpoint in sim before you copy anything to the Nano.

---

## 2. Current sim data plane (Unity gym ↔ ROS container ↔ trainer ↔ model)

This is how data moves **today**. The Jetson node must **replace** this plane, not join it.

### 2.1 Three processes, two transports

`docker compose up` (repo root) starts the training stack. Unity is **not** in Compose; you launch it on the host (`.\scripts\RunClientWrapper.ps1`).

```
  Windows / host                         Docker network
  +---------------------+                +--------------------------------------+
  |  Unity gym          |  ROS-TCP       |  ros-server                          |
  |  CarController      |  :10000        |  image: docker_ros-server:thin       |
  |  SceneDataPublisher |<-------------->|  ROS Noetic + niryo_moveit           |
  |  TrajectoryRolloutViz                |  unity_node.py  (TcpServer)          |
  +---------------------+                |  virtual_endpoint gRPC :50051        |
                                         +----------------^---------------------+
                                                          | gRPC JSON
                                                          | Subscribe / Publish
                                         +----------------v---------------------+
                                         |  sim-controller                      |
                                         |  robotaxi.py  (tf-agents SAC)        |
                                         |  RobotApi  (rl_agent/api.py)         |
                                         |  SavedModel load / greedy action     |
                                         +--------------------------------------+
```

| Hop | Protocol | Port | Who speaks | Payload |
|---|---|---|---|---|
| Unity ↔ `ros-server` | **ROS-TCP** (`ros_tcp_endpoint.TcpServer`) | **10000** | `unity_node.py` routing table | Binary ROS msgs named in §2.2 |
| `sim-controller` ↔ `ros-server` | **gRPC** `virtual_endpoint.RosNode` | **50051** | `rl_agent/api.py` `RpcClient` | JSON (`rospy_message_converter`) of the same ROS msgs |
| Model ↔ trainer | In-process TensorFlow | — | `robotaxi.py` `load_saved_model` / `run_policy` | `TimeStep` `(1, 31)` → action `(1, 2)` |

The trainer **never opens a ROS socket**. Default gRPC target is `ros-server:50051` (Compose alias `ros-server-0`). Scaled actors use `ros-server-N:50051`.

`ros-server` also bind-mounts `./rl_agent` → `/python_ws/src` and `../saved_models` → `/saved_models`. The SavedModel files sit on disk for the trainer; **`ros-server` does not load them**. `start.sh` runs `roslaunch niryo_moveit part_3.launch` (plus a Python workspace server on 60062). Env: `UNITY_MACHINE_IP=host.docker.internal`, `ROBOT_TYPE=robotaxi`.

**ROS versions do not match the car.** Sim ROS is **Noetic** in Docker. The Jetson is **Melodic**. You cannot `rostopic` the gym graph onto the Nano, and the `niryo_moveit` `.msg` package is **not installed** on the Jetson.

### 2.2 ROS-TCP routing table (`unity_node.py`)

The table is **static**. A topic that is not listed never reaches Unity, even if someone `rospy.Publisher`s it.

**Unity → ROS** (`RosPublisher` — Unity writes, ROS graph reads):

| Topic | Type | Role in robotaxi training |
|---|---|---|
| `car_scene_data` | `niryo_moveit/CarSceneData` | **The observation.** One message per physics tick after a command. |
| `scene_data` | `niryo_moveit/SceneData` | Arm / leftover gym. Trainer subscribes; `DonutCourse` does **not** use it. |
| `sim_status` | `niryo_moveit/SimStatus` | Handshake: reset done / force applied. |
| `camera/overhead` | `niryo_moveit/Camera` | Optional overhead frame. **Not** in the 31-D policy. |
| `move_action/result` | `niryo_moveit/MoveActionResult` | Arm gym. Unused for donut. |
| `move_action/feedback` | `niryo_moveit/MoveActionFeedback` | Same. |

**ROS → Unity** (`RosSubscriber` — trainer writes, Unity reads):

| Topic | Type | Role |
|---|---|---|
| `sim_command` | `niryo_moveit/SimCommand` | **The action / reset.** `cmd=0` restart track, `cmd=1` apply force. |
| `policy_rollouts` | `std_msgs/String` | JSON viz for `TrajectoryRolloutViz`. Not on the learning loop. |
| `move_action/goal` | `niryo_moveit/MoveActionGoal` | Arm gym only. |

There is **no** `/scan`, `/cmd_vel`, `/odom`, or `sensor_msgs/*` on this graph.

### 2.3 Message schemas (gym package `niryo_moveit`)

Definitions live in `docker/ros_server/ROS/src/niryo_moveit/msg/`. gRPC carries the same fields as JSON objects.

**`SimCommand.msg`** (trainer → Unity):

```
int32 cmd                         # 0 = reset / restart, 1 = apply force
niryo_moveit/ApplyForce ApplyForce
```

**`ApplyForce.msg`:**

```
float64 acceleration              # policy action[0], Unity motor torque (normalized)
float64 steering_angle            # policy action[1], fraction of maxSteeringAngle
int32   cmd_id                    # trainer-issued id; echoed on the next CarSceneData
int32   num_obstacles             # reset-time track clutter
float64 corner_radius             # reset-time TrackGenerator (must be float in JSON)
float64 curvature_difficulty
int32   chicanes_north            # reset-time (must be int in JSON)
int32   chicanes_east
int32   chicanes_south
int32   chicanes_west
```

`RobotApi.DoReset` publishes `{cmd: 0, ApplyForce: {accel 0, steer 0, num_obstacles, corner_radius, ...}}` and waits on `sim_status.status == 1` (`RESTARTED`).

`RobotApi.DoApplyForce(acceleration, steering_angle)` publishes `{cmd: 1, ApplyForce: {acceleration, steering_angle, cmd_id, num_obstacles}}` and waits twice: `sim_status.status == 2` (`FORCE_APPLIED`), then a `car_scene_data` whose `last_executed_cmd_id` equals that `cmd_id`.

**`SimStatus.msg`:** `int32 status` — Unity `SimController.Status`: `STARTED=0`, `RESTARTED=1`, `FORCE_APPLIED=2`.

**`CarSceneData.msg`:**

```
niryo_moveit/Sphere car
int32 last_executed_cmd_id        # correlates this obs to the ApplyForce that produced it
```

**`Sphere.msg`** (the `car` payload). The trainer keeps the whole dict as `latest_car_scene_data`. `DonutCourse.scene_data_array()` **selects a subset** into the policy vector. Fields **not** in the 31-D obs (pose, crash, goals 1/3/4, …) are still used for reward / episode end.

| `Sphere` field | In 31-D `donut_no_hint`? | Meaning |
|---|---|---|
| `speed` | yes, index 0 | m/s |
| `goal_2` | yes, index 1 | sideslip, degrees/180 |
| `left` … `right` (29 rays) | yes, indices 2–30 | SphereCast meters; see §3.1 for order |
| `dist_from_traj` | **no** (this is classic `donut` index 0) | angle to Unity goal |
| `location_*`, `rotation_z`, `angular_velocity`, `acceleration` | no | pose / rates |
| `goal_1`, `goal_3`, `goal_4` | no | other goal hints |
| `has_reached_goal`, `has_crashed` | no (episode bookkeeping) | Unity flags |
| `cost`, `dist_from_goal`, `last_goal_reached`, `current_goal` | no | reward / debug |

Ray field names on the wire, in **observation order** after speed / sideslip:

```
left, forward_left, forward_left_left,          # -90, -30, -60
n_27_50 … n_02_50,                              # -27.5 … -2.5
forward,                                        # 0
p_02_50 … p_27_50,                              # +2.5 … +27.5
forward_right_right, forward_right, right       # +60, +30, +90
```

### 2.4 One training / EVAL step (what actually happens)

EVAL uses the **same** ROS loop as collect. The model never talks to Unity.

```
1. Policy (in sim-controller) emits [accel, steer]
2. RobotApi.DoApplyForceBlocking
      → gRPC Publish  topic=sim_command  type=niryo_moveit/SimCommand
3. virtual_endpoint converts JSON → ROS msg, publishes on the Noetic graph
4. unity_node ROS-TCP forwards SimCommand to Unity
5. Unity CarController.ApplyForce(accel, steer)
6. Unity publishes SimStatus(FORCE_APPLIED=2)
      and later CarSceneData { car: Sphere, last_executed_cmd_id }
7. unity_node ROS-TCP → ROS graph
8. virtual_endpoint Subscribe stream → JSON → RobotApi._on_car_scene_data
9. RobotApi unblocks when last_executed_cmd_id matches
10. DonutCourseNoHint.scene_data_array(latest_car_scene_data) → 31 floats
11. Next TimeStep.observation = those 31 floats
```

Reset is the same path with `cmd=0` and a wait on `sim_status == 1`.

### 2.5 What the trainer never sees (and the Jetson only has)

| Sim gym (today) | Jetson OOTB (`jetson.md`) |
|---|---|
| `car_scene_data` (`niryo_moveit/CarSceneData`) | **does not exist** |
| `sim_command` (`niryo_moveit/SimCommand`) | **does not exist** |
| `sim_status` | **does not exist** |
| — | `/scan` (`sensor_msgs/LaserScan`, `frame_id=laser_frame`) |
| — | `/odom_raw` (`nav_msgs/Odometry`) — **use this for velocity** |
| — | `/odom_combined` (`geometry_msgs/PoseWithCovarianceStamped`) — pose only, **not** Odometry |
| — | `/imu` (`sensor_msgs/Imu`) |
| — | `/cmd_vel` (`geometry_msgs/Twist`) |
| SavedModel in `sim-controller` RAM | no inference node |

### 2.6 Implication for the Jetson adapter

A Jetson policy node **replaces** steps 2–10:

```
/scan + /odom_raw  →  build the same 31-vector  →  tf.saved_model.action
                   →  map [accel, steer] to Twist  →  /policy/cmd_vel  (mux onto /cmd_vel)
```

Do **not**:

- Install `niryo_moveit` msgs on the Nano or publish `CarSceneData`.
- Point `RobotApi` at the Jetson (`ros-server:50051` is the gym only).
- Run the `ros-server` container on the Nano as the car’s ROS master.
- Expect Unity `cmd_id` handshake. The real loop is async; you pick a rate (start 10 Hz).

Hardware signs, lidar TF, and `/cmd_vel` ownership are in `jetson.md`. The rest of this guide is the **vector and Twist contracts** the adapter must honor.

---

## 3. The definition gap (what the bridge must invent)

### 3.1 Observation: 31 floats the policy expects

`DonutCourseNoHint.scene_data_array()` is the parent 32-vector **minus index 0**. Layout:

| Deploy index | Training name | Sim source (after §2.4) | Real-robot analogue |
|---|---|---|---|
| 0 | `speed` | `Sphere.speed` ← `CarController.GetSpeed()` m/s, spec `[-10, 10]` | `/odom_raw` `twist.twist.linear.x` (after `invert_linear`, this matches physical forward) |
| 1 | `goal_2` (sideslip) | `Sphere.goal_2` — heading vs velocity, **degrees / 180**, spec `[-1, 1]` | `atan2(vy, vx)` vs yaw, or `twist.angular` + body velocity; divide by 180 |
| 2–30 | 29 ray distances | `Sphere.left` … `Sphere.right` SphereCast meters, spec `[0, 1000]` | Resample **`/scan`** into the same 29 bearings |

**Ray bearings (degrees, car frame, 0 = nose, + = right), in observation order after speed/sideslip:**

```
-90, -30, -60,
-27.5, -25, -22.5, -20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5,
0,
+2.5, +5, +7.5, +10, +12.5, +15, +17.5, +20, +22.5, +25, +27.5,
+60, +30, +90
```

Notes that will bite you if ignored:

- The order is **not** monotonic. `-30` comes **before** `-60`. Right side is `+60` then `+30` then `+90`. Copy this order exactly.
- Sim rays are **SphereCast radius 0.15 m**, max **50 m** (forward **100 m**). RPLiDAR A1 is a **2D beam**, ~12 m, ~1° bins, noise, glass, and a **blind zone under the bumper**.
- Sim `laser_frame` equivalent is the car heading. On the Jetson, `lidar.launch` must keep **`base_footprint → laser_frame` yaw = 0** (scan angle 0 = nose). A 180° TF swaps left/right in the vector and the policy will steer into walls.
- **No frame stacking. No running normalization.** Feed raw meters and m/s, same as Unity.
- **No camera** in this policy. `/csi_cam_0/image_raw` and sim `camera/overhead` are unused.

### 3.2 Action: 2 floats the policy emits vs Twist the car eats

| Index | Policy | Range | Unity meaning (`ApplyForce` fields) |
|---|---|---|---|
| 0 | `acceleration` | **`[0.05, 1.0]`** | `SimCommand.ApplyForce.acceleration` — normalized motor torque (always a little forward) |
| 1 | `steering_angle` | **`[-1, 1]`** | `SimCommand.ApplyForce.steering_angle` — fraction of **`maxSteeringAngle` (45°)** on the donut prefab |

The motor board does **not** take normalized torque. `jetracer` sends mm/s + steer on the serial packet; ROS tools think in **`/cmd_vel`**.

You must choose an explicit map and **never mix it with TEB/explore/joy**:

**Recommended first map (velocity command, not “torque”):**

```
linear.x  = accel_scale * action[0]          # e.g. 0.50 * a  →  ~0.025–0.50 m/s
angular.z = steer_scale * action[1]          # e.g. 0.40 * s  →  ±0.40 rad wheel angle
```

Then clip to what you already know is safe on this car (`max_vel_x` ~0.6–0.8 m/s, estop at ~0.20 m).

`invert_linear=true` in `jetracer.launch` already flips MCU +x. The policy’s “forward” must be **ROS +linear.x** after that fix. Do not invert again in the policy node.

Steering: `+action[1]` is **left** in ROS (`+angular.z`), matching the steer-direction test. Confirm once on the stand.

**Mismatch you cannot paper over:** the policy was trained as **always-forward** (`accel ≥ 0.05`). It will not reverse. Rear-stuck recovery must stay a **separate** ROS behavior, not the TF policy.

### 3.3 Time, compute, and graph ownership

| Issue | Why it matters |
|---|---|
| Jetson Melodic is **Python 2** | Stock `jetracer` nodes stay Py2. Policy inference should be a **separate Python 3** process (venv or Docker) talking ROS via `rospy` on Py3 is painful on Melodic — prefer **ROS1 rospy on Py2 only for I/O**, or **rosbridge / a tiny C++ republisher**, or run TF in a Py3 sidecar and pass vectors over UDP/unix socket. Do **not** reuse the Noetic `ros-server` image as the car’s master. |
| Nano memory / TF 2 | Full tf-agents SavedModel is heavy. Plan: **x86 replay first**, then Nano with TF 1.15/2.x that can `tf.saved_model.load`, or convert later to TFLite (no converter in-repo). |
| `/cmd_vel` is a single pipe | **Do not** run `explore_foxglove.launch`, `joy.launch`, or Foxglove 3D “publish pose” on `/cmd_vel` while the policy is up. Bringup **includes joy** — kill `/teleop_joy` or use `jetracer.launch` + lidar only. |
| `lidar_estop` | Keep it. Policy has no hard stop. |
| No `cmd_id` on the car | Sim steps are **blocking**. The Jetson loop is **async**. You invent the rate; start at 10 Hz to match Unity `SceneDataPublisher`. |

### 3.4 TF-agents call contract

Eval already does this in `robotaxi.py` (`get_saved_model` / `run_policy`) **inside `sim-controller`**, after `RobotApi` has turned `CarSceneData` into a numpy vector. On the Jetson you call the same SavedModel API; only the vector source changes.

```
saved = tf.saved_model.load(path)
policy_step = saved.action(time_step, policy_state)
action = policy_step.action   # shape (1, 2)
```

`time_step.observation` must be **`(1, 31)` float32**. Build a dummy `TimeStep` (`FIRST`/`MID`) with `reward=0`, `discount=1`, same as tf-agents. Reuse `policy_state` across steps (often empty for this SAC greedy policy — still pass it through).

Run the same spec check as `_extract_savedmodel_specs` / `_specs_compatible` before the first hardware cmd.

---

## 4. Step-by-step deployment (when you build it)

### Phase A — Pick and freeze the checkpoint

1. Train / pick a job with **Course = `donut_no_hint`**.
2. EVAL that SavedModel in the Docker sim (`load_saved_model` path — same `ros-server` + Unity loop as collect). Record return, crash rate, mean speed.
3. Copy the **directory** (not a single file) from the host `saved_models` mount:

   ```
   saved_model.pb
   variables/
   assets/          # if present
   policy_specs.pbtxt or similar
   ```

4. Write a one-line `model.json` next to it: `course_type`, `obs_dim=31`, `action_dim=2`, git SHA, train step, eval return.

### Phase B — Offline “same vector” test (laptop, no car)

5. Dump one Unity step: 31-D obs + 2-D greedy action from EVAL logs or a short `run_policy` instrument (this vector already came through `car_scene_data` JSON).
6. Load the SavedModel on x86 and confirm **bit-close** actions (atol ~1e-4).
7. Implement **scan → 29 rays** as a pure function. Unit-test with a synthetic `LaserScan` (wall at 1 m, 0°): `forward` bin ≈ 1.0, left/right large.
8. Implement **sideslip** from a fake odom (vx>0, yaw rate 0 → ~0).
9. Implement **action → Twist** and print it; no robot yet.

### Phase C — Jetson bringup without the policy

10. SSH: `ssh jetson`. One stack only.

    ```bash
    # Motors + lidar + odom + rosbridge. NOT explore. NOT a second bringup.
    roslaunch jetracer jetracer.launch
    # second terminal, if lidar is not in jetracer.launch:
    roslaunch jetracer lidar.launch
    ```

    For a full sensor set without teleop, prefer a **new launch** later that is `jetracer + lidar + camera + description + rosbridge` **minus `joy.launch`**. Until that exists, start bringup and `rosnode kill /teleop_joy /joy_node`.

11. Hardware sign checks (already proven on this car; re-run if anything changed):

    ```bash
    rosrun jetracer motor_direction_test.py          # +cmd = physical forward
    rosrun jetracer steer_direction_test.py          # +angular.z = wheels left
    ```

12. Confirm TF: `tf_echo base_footprint laser_frame` → **yaw ≈ 0**, translation ~`(0.1, 0, 0.12)`.
13. Confirm topics: `/scan` `sensor_msgs/LaserScan`, `/odom_raw` `nav_msgs/Odometry` (not `/odom_combined` — that is `PoseWithCovarianceStamped`).
14. Confirm `invert_linear=true` still set.

### Phase D — Observation bridge on the live bag

15. Record 30 s: `rosbag record /scan /odom_raw /imu /tf`.
16. On the laptop, replay and plot the 29 ray distances vs Foxglove scan. Front bin must shrink when you walk toward the nose.
17. Compare histograms of sim rays (EVAL dump) vs real rays: real will be **shorter, noisier, more inf/nan**. Decide clip: `min(range, 12.0)`, replace `inf` with `12.0` (or 50 to mimic sim “no hit”). **Document the clip** — changing it later is a silent obs shift.

### Phase E — Policy node (to be written later)

18. Process layout (recommended). This **replaces** `car_scene_data` / `sim_command`, it does not subscribe to them:

    ```
    /scan, /odom_raw  →  obs_bridge (31-vector)  →  tf_policy  →  /policy/cmd_vel
                              ↓
                         /policy/obs  (Float32MultiArray, debug)
                         /policy/action
    ```

    A **mux** or latch: only remap `/policy/cmd_vel` → `/cmd_vel` when an arming topic is true (`/policy/enable`). Default **off**. `lidar_estop` stays on `/cmd_vel`.

19. Control rate: start at **10 Hz** (matches Unity `SceneDataPublisher`). Do not run 50 Hz until the Nano can hold TF inference.
20. First-step `TimeStep` type `FIRST`, then `MID`. Reset `FIRST` if you pause >1 s.

### Phase F — Action map calibration (wheels off the ground)

21. Hold a **constant** policy action `[0.2, 0]` (or publish it by hand). Confirm slow forward, no steer.
22. `[0.2, +0.5]` → nose left; `[0.2, -0.5]` → nose right.
23. Sweep `accel_scale` until 0.4 in policy space is a crawl you would accept indoors (~0.25–0.35 m/s).
24. Sweep `steer_scale` until ±1 is near the real max steer without oscillation.

### Phase G — On-floor tests (section 5)

Do these **in order**. Do not skip to a living room loop.

---

## 5. How to test the model

### 5.1 Software-in-the-loop (required)

- Replay a bag through `obs_bridge` + SavedModel.
- Log `/policy/action` vs time. Compare to the same bag run through a laptop TF process (same weights).
- Fail if actions diverge or NaN.

### 5.2 Stand / box test (required)

- Wheels free or car on a box.
- Enable policy 3 s, then disable.
- Watch `cmd_vel_debug`: you want **`forward` / `forward-left` / `forward-right`**, not a stuck `backward` (this policy should not emit reverse).
- E-stop: put a board 15 cm in front of the lidar; `lidar_estop` must zero `/cmd_vel` even if the policy still outputs accel.

### 5.3 Open-floor crawl (first live test)

- Empty garage, 3+ m clear ahead, person on the disable switch (or `rosnode kill` the policy).
- Duration: **10 seconds**, then mandatory disable.
- Pass: car moves **forward** along the open axis; steer is small; no spin-in-place; no reverse.
- Fail: immediate wall-seek, spin, or full-throttle. Recheck ray order, lidar yaw, `invert_linear`, action scales.

### 5.4 Obstacle garden (second live test)

- Soft obstacles (boxes, foam) at 1–3 m, left and right, not a dead-end.
- 30–60 s runs.
- Pass: policy **veers away** from the nearest red `/scan` cluster (Foxglove).
- If it veers **into** the cluster: left/right ray order or steer sign is still wrong — **stop**, do not “tune scales.”

### 5.5 Closed-loop robustness (only after 5.4)

- Same space you will demo.
- Three trials from different start poses (pointed at open space, 45° to a wall, tail near a wall).
- Metrics to write down: time-to-first-contact, mean `|linear.x|`, fraction of steps with front range < 0.4 m, whether `lidar_estop` fired.
- Tail-against-wall: policy will **not** reverse. That is expected. Keep the **forward unstick** ROS node for that case; do not ask the SAC policy to learn reverse on the Nano in v1.

### 5.6 What not to test

- Do not run **`explore_foxglove.launch`** and the TF policy together.
- Do not use Foxglove 2D tools that advertise `/scan` or `/cmd_vel`.
- Do not evaluate a **`donut` (32-D)** checkpoint on this bridge.
- Do not run the gym `ros-server` container against the Jetson ROS master, or subscribe the policy node to `car_scene_data` / `sim_command`.

---

## 6. Recommended Unity / gym changes (sim → real)

Train the **next** policies so the 31-D vector looks more like the Jetson. None of this is required to write the first bridge; it is what will make later checkpoints actually transfer.

The gym will still go Unity → ROS-TCP → `ros-server` → gRPC → trainer. These changes are about **what is inside `Sphere` and `ApplyForce`**, not about replacing that graph with `/scan` in Docker.

### 6.1 Perception: make rays look like a 2D lidar

1. **Replace SphereCast (0.15 m radius) with a thin ray or tiny radius (0.01–0.03 m)** so ranges match a lidar beam, not a “fat” probe that hits earlier.
2. **Cap max range to ~12 m** (RPLiDAR A1), not 50/100 m. Retrain; old 50 m “clear” bins become a different distribution.
3. **Publish at lidar-like angular sampling**, then **downsample to the same 29 bearings** in one shared Python function used by **both** the gym (`SceneDataPublisher` or a post-step) and the Jetson bridge. One function, two callers — or the orders will drift.
4. **Add dropouts:** random `inf` / max-range on 5–15% of rays per step (real scans have holes).
5. **Add Gaussian range noise** (σ ≈ 0.02–0.05 m) and occasional **ghost hits** at 0.2–0.5 m (false close returns).
6. **Blind zone:** zero or inflate rays that would hit the **car body / lidar mast** on the real TF (`x=0.1, z=0.12`). Sim currently does not see the chassis.
7. **No rear rays in the 29-D fan** (−90…+90 only). The real policy is **front-hemisphere blind behind**. Add 2–4 rear bins **only if** you also add them on the Jetson. Do not train on 360° if deploy is 180°.

### 6.2 Dynamics: stop training a different vehicle

8. **Match wheelbase (~0.24 m), max steer, and turning radius (~0.40 m)** to `teb_local_planner_params.yaml` / the real Ackermann, not the Waymo-ish `maxMotorTorque=300` / arcade feel.
9. **Action = something you can send on `/cmd_vel`.** Best long-term: change the gym action to **`[v, steer]`** in SI units (or the same Twist the car uses), with `v ≥ 0`. Today’s normalized torque on `sim_command` is the largest silent gap.
10. Until then, **identify a static map** (bench): record Unity `ApplyForce(a, s)` vs resulting vx, yaw rate; fit `accel_scale` / `steer_scale` from data, not guesswork.
11. **Delay the action 50–150 ms** in the env (sensor + Wi-Fi + Nano inference). The real loop is not a blocking `DoApplyForce`.
12. **IMU / odom noise** on `speed` and `goal_2`: bias, delay, and the **yaw-sign issues** we already hit (`invert_linear` must be applied in sim the same way, or sideslip will flip).
13. **Never allow reverse in the action spec** if the deploy policy will not reverse — or **do** allow reverse in sim **and** implement reverse on the ESC with the same sign tests.

### 6.3 World and curriculum: less “perfect donut”

14. **Clutter:** chairs, boxes, narrow gaps **wider than the real car + 0.16 m**. Randomize count and pose every reset (`num_obstacles` already exists on `ApplyForce`; make obstacles **lidar-visible** and **non-traversable**).
15. **Texture/material:** lidar in Unity should hit the same collision mesh the car hits. Decorative walls the SphereCast misses will not exist on the Jetson.
16. **Start-pose randomization:** against a rear wall, 45° to a wall, in a corner. The real car starts “rear stuck” often; the gym almost always starts in open track.
17. **Floor friction / motor lag** randomization (domain randomization on torque scale ±20%, steer trim ±5°).
18. **Keep `donut_no_hint`.** Do not reintroduce goal angle, map coordinates, or GPS.
19. **Optional second course:** a “living-room” gym with the **same 31-D contract** (not `simple` 9-D). Same `observation_spec` / `action_spec` so checkpoints stay compatible.
20. **Eval protocol in sim that mirrors section 5:** short open crawl, then obstacle garden, with the **same ray clip and action map** the Jetson will use (a `RealRobotObsWrapper` in Python around `RobotaxiEnv`).

### 6.4 Training hygiene

21. Export **only `GreedyPolicy`** (already done for best checkpoints). Do not deploy the stochastic collect policy.
22. Stamp every SavedModel with `course_type` and **refuse** to load 32-D on the robot node.
23. After gym ray/dynamics changes, **retrain**; do not expect an old 50 m SphereCast policy to behave.

---

## 7. Suggested build order (when you leave plan mode)

1. Shared `rays_from_laserscan(scan, yaw_offset=0) -> (29,)` + unit tests.
2. Shared `obs31(speed, sideslip, rays)` + `twist_from_action(a, s, scales)`.
3. x86 bag replay + SavedModel.
4. Jetson Py3 sidecar or Melodic-safe inference node + `/policy/enable`.
5. Box test → 10 s crawl → obstacle garden.
6. Only then: gym patches in §6 and a new `donut_no_hint` training job.

---

## 8. File index

| Role | Path |
|---|---|
| Compose: `ros-server` + `sim-controller` | `docker-compose.yml` (ports 10000, 50051; mounts `/saved_models`) |
| `ros-server` image / entry | `docker/ros_server/Dockerfile` (Noetic), `docker/ros_server/ROS/src/start.sh` |
| ROS-TCP topic table | `docker/ros_server/ROS/src/niryo_moveit/scripts/unity_node.py` |
| Gym msg schemas | `docker/ros_server/ROS/src/niryo_moveit/msg/` (`CarSceneData`, `Sphere`, `SimCommand`, `ApplyForce`, `SimStatus`) |
| gRPC JSON bridge | `protos/virtual_endpoint/proto/ros_service.proto`; server inside the `ros-server` image |
| Trainer ↔ ROS | `rl_agent/api.py` (`RobotApi`, `DoApplyForce`, `DoReset`, Subscribe/Publish) |
| Trainer, SavedModel save/load | `rl_agent/robotaxi.py` (`PolicySaver`, `load_saved_model`, `run_policy`) |
| 32-D obs / 2-D action | `rl_agent/environments/courses/donut_course.py` |
| 31-D sim→real obs | `rl_agent/environments/courses/donut_course_no_hint.py` |
| Unity rays / steer / ApplyForce | `unity/Assets/CarController.cs` |
| Unity obs publish | `unity/Assets/Scripts/SceneDataPublisher.cs` |
| Unity sim_status / command loop | `unity/Assets/Scripts/SimController.cs` |
| Jetson ROS reality | `jetson.md` |
| Motor / steer sign | `invert_linear` in `jetracer.launch`; `motor_direction_test.py`, `steer_direction_test.py` |

---

## 9. Non-goals for v1

- Camera / vision policies (ignore `camera/overhead` and `/csi_cam_0/image_raw`).
- Running SAC **training** on the Nano.
- Sharing `/cmd_vel` with `explore_lite` or the joystick.
- Assuming `/odom_combined` is `nav_msgs/Odometry` (it is not).
- Deploying a `donut` 32-D model by “padding” a fake goal angle.
- Porting `niryo_moveit` msgs, ROS-TCP, or `RobotApi` gRPC onto the Jetson.
- Running `docker_ros-server:thin` as the car’s ROS master.
