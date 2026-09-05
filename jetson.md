# Jetson (JetRacer) notes

Onboard compute for the physical car: an NVIDIA Jetson running **ROS Melodic**
(Ubuntu 18.04, Python 2). This is where the real-robot ROS graph runs — motor
base, LiDAR, camera, and the bridge the dashboard / Foxglove talk to.

## Access

```bash
ssh jetson            # alias in ~/.ssh/config -> jetson@192.168.86.210
# HostName 192.168.86.210, User jetson, key auth (id_ed25519)
```

Default credentials are `jetson` / `jetson` (key auth is set up so no password is
needed from this machine). Consider changing the password if it stays on Wi-Fi.

## Bringing everything up

Everything the car needs is started by a single master launch file:

```
~/catkin_ws/src/jetracer_ros/launch/jetracer_bringup.launch
```

> Directory vs. package name: the folder is `jetracer_ros`, but its
> `package.xml` declares `<name>jetracer</name>`, so ROS refers to it as the
> **`jetracer`** package and `$(find jetracer)` resolves to
> `~/catkin_ws/src/jetracer_ros`.

Launch it via the helper script (sources ROS, fixes device permissions, sets the
ROS master/host env, then `roslaunch`):

```bash
~/start_jetracer.sh
```

which does:

```bash
source /opt/ros/melodic/setup.bash
source /home/jetson/catkin_ws/devel/setup.bash
sudo chmod 666 /dev/ttyTHS1 /dev/ttyACM* /dev/i2c-* 2>/dev/null
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=$(hostname -I | awk '{print $1}')
roslaunch jetracer jetracer_bringup.launch
```

### Do I need to add `roscore` to the launch file?

**No.** `roslaunch` **auto-starts a ROS master (roscore)** if one isn't already
running at `ROS_MASTER_URI`, and tears it down when the launch exits. `roscore`
isn't a normal node you `<include>` or `<node>` — it's the master + parameter
server + `rosout` that roslaunch bootstraps for you. So `jetracer_bringup.launch`
should **not** try to launch roscore; adding it would just conflict with the one
roslaunch already brings up.

Run `roscore` **separately** (its own terminal / systemd unit, before the
bringup) only if you want the master to outlive the bringup, e.g.:

- you want to stop/relaunch the bringup nodes without tearing down the master;
- multiple `roslaunch` / `rosrun` invocations should share one master;
- remote machines (like this dev box) set `ROS_MASTER_URI` to the Jetson and need
  the master up independently of the bringup.

For the normal one-command startup (`~/start_jetracer.sh`), the master roslaunch
auto-starts is all you need.

## Invoking the bringup & monitoring logs

### Invoke

```bash
ssh jetson
~/start_jetracer.sh          # normal path: sources ROS, fixes perms, roslaunch
```

Or drive `roslaunch` directly (after sourcing + setting env as the script does):

```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch jetracer jetracer_bringup.launch            # normal
roslaunch --screen jetracer jetracer_bringup.launch   # force ALL nodes' output to this terminal
```

> **Run only ONE bringup at a time.** Starting `jetracer_bringup.launch` twice
> (e.g. two SSH terminals, or `start_jetracer.sh` run again while the first is
> still up) brings up duplicate nodes. Same-named nodes evict each other, and with
> `respawn="true"` the evicted node restarts instantly and the pair loops forever
> — you'll see `teleop_joy`/`joy_node` spamming
> `shutdown request: ... Reason: new node registered with same name`. Two motor
> drivers also fight over `/dev/ttyACM0` (the driver then aborts with
> `exit code -6`). `start_jetracer.sh` now **guards against this** (it `pgrep`s for
> a running bringup and refuses to start a second), but if you launch `roslaunch`
> by hand you can still hit it. If it happens, kill the extra launch:
>
> ```bash
> # list bringups, then INT the duplicate roslaunch PID(s)
> ps -ef | grep '[j]etracer_bringup.launch'
> kill -INT <pid>            # add kill -TERM <pid> if it lingers
> ```

Because the bringup runs in the foreground, an SSH drop kills it. For demos, run
it inside a persistent session so it survives disconnects and you can reattach:

```bash
tmux new -s jetracer '~/start_jetracer.sh'   # or: tmux new -s jetracer, then run inside
# detach: Ctrl-b d      reattach later: tmux attach -t jetracer
```

### Monitor logs across all the nodes

The bringup starts many nodes (motor base, EKF, LiDAR, camera, rosbridge,
joystick). A few ways to watch them:

- **All console output in one place** — start with `roslaunch --screen ...`, which
  forces every node's stdout/stderr to the launching terminal (by default only
  nodes with `output="screen"`, like `teleop_joy`, print there; the rest log to
  files). Combine with `tmux` so you can scroll back.
- **Aggregated ROS log stream (`/rosout`)** — every node's `ROS_INFO/WARN/ERROR`
  is published here regardless of `output`:

```bash
rostopic echo /rosout            # everything, all nodes
rostopic echo /rosout_agg        # de-duplicated aggregate
```

- **`rqt_console`** — GUI to filter the same messages by node / severity / text
  (needs an X display; run with `ssh -X jetson` or from a desktop session):

```bash
rqt_console
```

- **Per-node log files on disk** — roslaunch writes one `*.log` per node under the
  current run's log dir. Tail them all live:

```bash
roscd log                        # or: cd ~/.ros/log/latest
tail -f ~/.ros/log/latest/*.log
```

- **Is a node alive / what's flowing** — sanity-check individual services:

```bash
rosnode list                     # all running nodes
rosnode info /jetracer           # connections, pubs/subs for one node
rostopic hz /scan                # LiDAR is publishing? (~10 Hz)
rostopic hz /csi_cam_0/image_raw # camera (~20 Hz)
rostopic echo -n1 /odom_raw      # one odometry sample
rostopic echo /joy               # joystick input (press buttons)
```

- **Roughly per "service" (launch include) mapping** — handy when hunting which
  node is noisy: motor base → `/jetracer`, `/robot_pose_ekf`, `/odom_ekf_node`;
  LiDAR → `/rplidarNode`; camera → `/csi_cam_0` (gscam); bridge →
  `/rosbridge_websocket`; controller → `/joy_node`, `/teleop_joy`; robot model →
  `/robot_state_publisher`.

## What `jetracer_bringup.launch` starts

```xml
<launch>
  <!-- 1. Motor Base / Odometry Node -->
  <include file="$(find jetracer)/launch/jetracer.launch" />
  <!-- 2. RPLiDAR -->
  <include file="$(find rplidar_ros)/launch/rplidar.launch" />
  <!-- 3. CSI Camera -->
  <include file="$(find jetracer)/launch/csi_camera.launch" />
  <!-- 4. Rosbridge WebSocket (for Foxglove / Browser Control) -->
  <include file="$(find rosbridge_server)/launch/rosbridge_websocket.launch" />
  <!-- 5. Remote controller (joystick teleop) -->
  <include file="$(find jetracer)/launch/joy.launch" />
  <!-- 6. Robot model (URDF -> robot_state_publisher) for RViz / Foxglove -->
  <include file="$(find jetracer)/launch/description.launch" />
  <!-- 7. Static Coordinate Transforms -->
  <node pkg="tf" type="static_transform_publisher" name="base_to_laser"
        args="0.1 0.0 0.12 0.0 0.0 0.0 base_footprint laser_frame 100" />
</launch>
```

### 1. Motor base + odometry — `jetracer.launch`

- **`jetracer`** node (`pkg=jetracer`, `respawn=true`) — the C++ driver for the
  motor/servo control board over serial **`/dev/ttyACM0`**. Publishes raw
  odometry remapped to **`/odom_raw`** and subscribes to velocity commands
  (`/cmd_vel`). `publish_odom_transform=false` (the EKF owns the `odom` TF
  instead). Throttle/steering calibration lives in its params
  (`linear_correction`, `coefficient_a..d`).
- **`robot_pose_ekf`** — fuses `/odom_raw` + `/imu` into a filtered pose,
  `output_frame=odom`, `base_footprint_frame=base_footprint`, 30 Hz. Publishes
  `/odom_combined` (odom_used=true, imu_used=true, vo_used=false).
- **`odom_ekf_node`** (`odom_ekf.py`) — republishes the EKF's combined pose as a
  standard `nav_msgs/Odometry` topic.
- **static TF** `base_footprint -> base_imu_link` (IMU mount, +2 cm in Z).

### 2. RPLiDAR — `rplidar.launch` (`rplidar_ros`)

- **`rplidarNode`** on serial **`/dev/ttyACM1`** @ 115200 baud (A1/A2 class),
  `angle_compensate=true`, **`frame_id=laser_frame`**. Publishes **`/scan`**
  (`sensor_msgs/LaserScan`). The frame matches the `base_footprint -> laser_frame`
  static transform, so `/scan` lines up with the robot model in RViz/Foxglove.

### 3. CSI camera — `csi_camera.launch`

- **`gscam`** node named `csi_cam_0` using the Jetson `nvarguscamerasrc`
  GStreamer pipeline (**640×480 @ 20 fps**, `flip_method=0`). Publishes
  **`/csi_cam_0/image_raw`**; camera intrinsics from
  `$(find jetracer)/config/camera_calibration/cam_640x480.yaml`.
  Current SAC policies do **not** consume this topic. To mirror it in the Unity
  gym and train a vision policy, see **Sim camera** below.

### 4. Rosbridge — `rosbridge_websocket.launch` (`rosbridge_server`)

- WebSocket bridge (default port **9090**) exposing the ROS graph to
  **Foxglove Studio** / browser clients for teleop and visualization.

### 5. Remote controller (joystick teleop) — `joy.launch`

- **`joy_node`** (`pkg=joy`, `respawn=true`) — reads the gamepad at
  **`/dev/input/js0`** (`deadzone=0.12`, `autorepeat_rate=20` Hz). Publishes
  **`/joy`** (`sensor_msgs/Joy`).
- **`teleop_joy`** (`teleop_joy.py`, `pkg=jetracer`, `respawn=true`) — maps `/joy`
  stick input to **`/cmd_vel`**, so the same velocity topic the motor base already
  listens on is driven by the controller.

> Both nodes are `respawn="true"`, so they restart automatically if they die —
> e.g. if the gamepad drops off Wi-Fi/USB and `/dev/input/js0` disappears,
> `joy_node` keeps retrying and reconnects once the pad is back (no full relaunch
> needed).
>
> Standalone equivalent (without bringup): `roslaunch jetracer joy.launch`.

### 6. Robot model — `description.launch`

- Loads a **visual-only URDF** (`$(find jetracer)/urdf/jetracer.urdf`) onto the
  **`robot_description`** parameter and runs **`robot_state_publisher`**, which
  publishes the model's (fixed) TF frames. This is what makes a 3D car appear in
  RViz / Foxglove — see "Showing the car model in RViz / Foxglove" below.
- The URDF uses **primitive shapes only** (boxes/cylinders, no mesh files) so it
  renders in `app.foxglove.dev` (the web app can't load `package://…` meshes
  without extra hosting).
- It is **additive**: it roots at `base_footprint` (published by the EKF) and only
  adds *new* links (`base_link`, four `wheel_*`, `lidar_visual`, `camera_visual`).
  It deliberately does **not** redefine `laser_frame` / `base_imu_link`, so
  `robot_state_publisher` never fights the existing static transforms. All joints
  are fixed, so no `joint_state_publisher` is needed.
- Dimensions are approximate (body ≈ 0.30 × 0.17 × 0.05 m, wheel r=0.03 m); tweak
  `urdf/jetracer.urdf` to taste — it's cosmetic and won't affect sensing/control.

### 7. Static transform

- `base_footprint -> laser_frame` at `(x=0.1, y=0.0, z=0.12)` — LiDAR mount
  offset (10 cm forward, 12 cm up).

## Showing the car model in RViz / Foxglove

The bringup already publishes the model (`robot_description` param +
`robot_state_publisher`, via `description.launch`). To see it:

**RViz** (run on a machine with a display, `ROS_MASTER_URI` pointed at the Jetson):

1. Set **Fixed Frame** to `odom` (or `base_footprint`).
2. **Add → RobotModel**. Leave *Description Source = Parameter*, *Parameter =
   `robot_description`*. The car appears and moves with odometry.
3. Optionally **Add → TF** to see the frames, and **LaserScan** on `/scan`.

**Foxglove** (`app.foxglove.dev`, connected to `ws://<jetson-ip>:9090` via
rosbridge):

1. Open a **3D** panel.
2. In the panel settings, expand **Custom layers → Add URDF** (a.k.a. the URDF
   layer). Set **Source = Topic** and **Topic = `/robot_description`** (note the
   leading slash). `description.launch` publishes the URDF onto a **latched
   `/robot_description`** (`std_msgs/String`) topic via `publish_robot_description.py`
   specifically so the web app can load it over the ws connection. (If your build
   offers **Source = Parameter**, `robot_description` works too.)
3. Set the panel's **Follow / Display frame** to **`base_footprint`** (or `odom`),
   **not `laser`** — the `laser` frame (used by `/scan`) is not connected to the
   model's TF tree, so following it hides the car. Enable **TF** to see frames; add
   the **LaserScan** (`/scan`) and camera image as needed.

> **URDF layer shows nothing?** Check the two most common causes: (1) the Topic is
> `robot_description` without the leading slash — it must be `/robot_description`;
> (2) the display frame is `laser` (disconnected) instead of `base_footprint`/`odom`.

## SLAM, navigation & autonomous exploration

These are an **alternative** stack to `jetracer_bringup.launch` — they bring up
their own `jetracer` + `lidar` + `csi_camera`, so **do not run them at the same
time as the bringup** (duplicate drivers fight over `/dev/ttyACM0`).

| Launch (`roslaunch jetracer …`) | What it does |
|---|---|
| `slam.launch` / `gmapping.launch` | SLAM mapping only (builds `/map`) |
| `move_base.launch` | Navigation stack (global + TEB local planner) |
| `slam_nav.launch` | `jetracer` + `lidar` + `csi_camera` + `gmapping` + `move_base` |
| `explore.launch` | Frontier auto-exploration (`explore_lite`) → sends goals to move_base |
| `map_autosave.launch` | Periodically saves `/map` → `~/maps/explore_map.{pgm,yaml}` |
| `reverse_recovery.launch` | Backs the car out of dead-ends when navigating-but-not-moving |
| `lidar_estop.launch` | Front-lidar emergency stop (cancel + zero `/cmd_vel` if under 0.25 m ahead) |
| `cmd_vel_debug.launch` | Logs `forward` / `backward` / `forward-left` / `forward-right` / `backward-left` / `backward-right` from `/cmd_vel` |
| `explore_foxglove.launch` | **One-command** `slam_nav` + rosbridge + model + `explore` + `reverse_recovery` + `lidar_estop` + `cmd_vel_debug` + `map_autosave` |

### Autonomous exploration (car maps + drives itself)

`explore_lite` (installed via `ros-melodic-teb-local-planner` /
`ros-melodic-explore-lite`) watches the gmapping `/map`, finds frontiers between
known/unknown space, and repeatedly sends goals to `move_base`, so the car drives
itself to unexplored areas until the map is complete. One command:

```bash
roslaunch jetracer explore_foxglove.launch
```

This starts SLAM + navigation + rosbridge + the URDF model + exploration together.
Then in **Foxglove** (`ws://<jetson-ip>:9090`), in a 3D panel add:

- **Map** on `/map` (the occupancy grid being built),
- **LaserScan** on `/scan`,
- the **global/local costmaps** (`/move_base/global_costmap/costmap`,
  `/move_base/local_costmap/costmap`),
- frontier markers on `/explore/frontiers` (explore_lite, `visualize:=true`),
- the robot model (URDF layer, Topic `/robot_description`), display frame `map`.

### Saving the map

`explore_foxglove.launch` includes **`map_autosave.launch`**, which runs
`scripts/save_map_periodic.sh` — every 20 s it snapshots the live `/map` to
**`~/maps/explore_map.pgm`** + **`explore_map.yaml`** (overwriting). When you stop
exploring, that pair **is** your finished map. Change the cadence with
`roslaunch jetracer explore_foxglove.launch` … or run the autosave alone:
`roslaunch jetracer map_autosave.launch interval:=10`.

Save a one-off snapshot at any time (e.g. a named final map):

```bash
rosrun map_server map_saver -f ~/maps/my_map
```

Reuse a saved map later (e.g. for AMCL localization / `nav.launch`):

```bash
rosrun map_server map_server ~/maps/explore_map.yaml
```

**Notes / tuning** (`launch/explore.launch`):

- The JetRacer is **Ackermann** (min turning radius, can't rotate in place), so
  `move_base` uses the **TEB** local planner and `explore_lite` uses
  `orientation_scale=0`. In tight spaces some frontiers are unreachable —
  explore_lite blacklists them after `progress_timeout` (30 s) and moves on.
- Key params: `min_frontier_size` (0.3 m — raise to ignore small nooks),
  `progress_timeout`, `potential_scale` (favor nearer frontiers).
- **Car gets stuck / freezes on one goal:** if TEB emits all-zero `/cmd_vel` and
  spams `trajectory is not feasible` forever, it's boxed in and — because move_base
  keeps *resetting* the planner instead of aborting — explore_lite never switches
  frontiers. Mitigated by lowering **`controller_patience` 15 → 5 s** (move_base
  aborts a stuck goal sooner) in `move_base.launch` and **`progress_timeout`
  30 → 20 s** in `explore.launch`, so it abandons unreachable goals and tries a
  different frontier. A car that can't rotate in place still can't escape a true
  dead-end from planning alone — this is what the **reverse recovery** (below)
  handles automatically; only physically reposition it if even reversing can't free
  it (the map keeps auto-saving, so nothing is lost).
- **Reverse / backup self-recovery** (`scripts/reverse_recovery.py`,
  `launch/reverse_recovery.launch`, included in `explore_foxglove.launch`):
  move_base's built-in recovery behaviors are useless for an Ackermann car
  (`clearing_rotation_allowed=false`, so no in-place rotation). This node watches
  for **"navigating but not moving"** — a move_base goal is `ACTIVE` (via
  `/move_base/status`) while `/odom_raw` velocity stays below `stuck_speed`
  (0.03 m/s) for `stuck_time` (6 s). When triggered it: (1) **checks the rear lidar
  arc** on `/scan`, (2) if the rear is clear, publishes to `/move_base/cancel` so
  move_base stops fighting for `/cmd_vel` and drives a short **reverse pulse**
  (`reverse_speed` 0.20 m/s for `reverse_time` 1.5 s) with an **alternating steer**
  (`reverse_steer` 0.4 rad/s, sign flips each time) to reorient, then (3) releases so
  explore_lite picks a fresh frontier from the new pose. A `cooldown` (6 s) prevents
  thrashing. All values are ROS params in the launch file.
  - **Rear-obstacle safety (important):** the first version reversed *blind* and
    drove the car straight into the wall behind it (it only fires when the car is
    boxed in, so there is usually something close on multiple sides). It now reads
    `/scan`, computes the minimum range in a **rear sector** (`±rear_arc`, ~50°
    around `base_footprint -x` via TF — **not** raw scan angle π).
    `lidar.launch` now uses **yaw 0** so laser `+x` is the nose (the stock
    Waveshare yaw of 3.14 made scan angle 0 the tail and caused
    `FRONT BLOCKED` on a wall behind the car). If the **rear is blocked and the
    front is clear** it drives a short **forward** pulse (tail against a wall —
    move_base will not start from an in-collision pose). If the **rear is clear**
    it reverses. If both ends are blocked it logs "boxed in, needs manual
    reposition" and waits. Pulses abort the instant clearance is lost. Note: the
    ESC must accept a negative `/cmd_vel` throttle as reverse.
- **GlobalPlanner `NO PATH!` / "Failed to get a plan from potential when a legal
  potential was found. This shouldn't happen.":** the global planner *did* compute
  a valid potential field (goal is reachable) but its default **gradient-descent
  path traceback** fails to extract the path. This makes reachable frontiers look
  unreachable, so `explore_lite` blacklists good goals and stops early. Fixed by
  forcing the **grid-path traceback** in `move_base.launch`:
  `GlobalPlanner/use_grid_path=true` (plus `use_dijkstra=true`, `use_quadratic=true`,
  `old_navfn_behavior=false`, `allow_unknown=true`). Paths are slightly more jagged
  but planning is reliable.
- **Driving into objects (after motor invert):** with `+linear.x` now actually
  going forward, the old "stuck" tunings became collision tunings. Causes:
  **`max_vel_x` was 1.0 m/s** with `acc_lim_x` 2.0; **TEB `min_obstacle_dist` 0.13 m**
  / `inflation_dist` 0.10 m; the **local costmap had no inflation layer**; and
  move_base recovery **cleared both costmaps in a 3 m radius** (`Clearing both
  costmaps to unstuck robot (3.00m)`), after which the car drove through furniture
  it had just forgotten. After signs/TF were fixed, those safety cuts felt
  sluggish, so they were raised toward the original lively feel (not back to
  1.0 m/s). Current explore pace (2026-08-24): `max_vel_x` **0.85**, `acc_lim_x`
  **1.8**, `max_vel_theta` **0.55**, `weight_optimaltime` **4**,
  `weight_kinematics_forward_drive` **15** (prefer forward over reverse-wiggle),
  `min_obstacle_dist` **0.16**, `inflation_dist` **0.20**, explore
  `planner_frequency` **1.0** Hz, `gain_scale` **2.5**, `potential_scale` **1.5**,
  `min_frontier_size` **0.5** m, `progress_timeout` **10** s. Reverse recovery:
  `stuck_time` **2.5** s, `reverse_time` **2.5** s, `reverse_speed` **0.30**.
  Local costmap still has an **inflation layer**;
  **`recovery_behavior_enabled=false`**. Plus a **front lidar estop**
  (`scripts/lidar_estop.py`): uses TF so “front” is `base_footprint +x`.
  `lidar.launch` laser yaw is **0** (was 3.14, which swapped front/rear). Estop
  is a last-resort bumper at **0.12 m** (was 0.20 m, which cancelled every TEB
  plan when the nose was merely close). If anything is within **0.12 m** along
  the robot’s forward arc, it cancels the goal and publishes zero `/cmd_vel`.
- **TEB "trajectory is not feasible":** the *global* plan succeeds but the local
  planner can't find a collision-free, car-valid path — usually the car is too
  close to obstacles for its clearance, or a turn is too sharp. Prefer slowing
  down / giving it space over shrinking `min_obstacle_dist` (that is what made
  it hit things once motor direction was fixed).
- **Map size vs. Nano performance:** `gmapping.launch` bounds are **±15 m**
  (30 m × 30 m ≈ 600×600 cells). The stock config used **±50 m** (100 m × 100 m ≈
  4M cells), which made the global costmap update loop take **~7 s** on the Nano and
  the global planner time out (`move_base: Failed to get a plan`, `Map update loop
  missed its desired rate ... took 7.39 seconds`, repeated costmap clearing). If you
  need to map a bigger area, raise the bounds, but expect slower planning; coarsen
  `delta` (e.g. 0.05 → 0.10) to compensate.
- **Prerequisites installed:** `ros-melodic-teb-local-planner` (move_base local
  planner) and `ros-melodic-explore-lite` (exploration).
- No joystick is included (autonomous). If you add `joy.launch` to take manual
  control, note `teleop_joy` also publishes `/cmd_vel` and will fight move_base.

## Key topics & frames

| Thing | Value |
|---|---|
| Raw wheel odometry | `/odom_raw` |
| Fused odometry | `/odom_combined` (+ republished by `odom_ekf.py`) |
| Velocity command in | `/cmd_vel` (from teleop + dashboard) |
| Joystick input | `/joy` (from `joy_node`) |
| IMU | `/imu` |
| LiDAR scan | `/scan` (frame `laser_frame`) |
| Camera image | `/csi_cam_0/image_raw` (640×480 @ 20 fps) |
| Rosbridge WS | `ws://<jetson-ip>:9090` |
| Robot model | `robot_description` param (visual URDF, `robot_state_publisher`) |
| TF tree | `odom -> base_footprint -> {base_imu_link, laser_frame, base_link -> wheel_*/lidar_visual/camera_visual}` |

## Sim camera: mirror the JetRacer CSI feed into Unity gyms and the model

Today’s SAC policies (`donut` / `donut_no_hint`) are **1-D vectors** (31 or 32 floats). They do **not** use the CSI camera. `/csi_cam_0/image_raw` is published on the Nano for Foxglove only. In the Unity gym, overhead-camera publish is **commented out** in `SimController.cs`; `unity_node.py` still lists `camera/overhead`, and `RobotApi` already subscribes — nothing in `DonutCourse` or the actor/critic reads it.

This section is the plan to make the **gym camera look like the JetRacer CSI**, ship frames over the existing ROS-TCP → gRPC plane, and turn them into a tensor the policy can train and (later) run on. It does **not** implement the nodes.

Do this as a **new course** (e.g. `donut_camera`). Do **not** widen `donut_no_hint` in place — that would invalidate every 31-D SavedModel.

### 1. Target camera contract (copy the Nano)

Match the live Jetson feed, not a cinematic Unity cam. Values from `csi_camera.launch` + `config/camera_calibration/cam_640x480.yaml` + the URDF `camera_visual` link:

| Property | JetRacer (OOTB) | Unity gym must do |
|---|---|---|
| Topic the rest of the stack should treat as “the car camera” | **`/csi_cam_0/image_raw`** | Sim: `camera/front` (see §3). Deploy: same Nano topic. |
| ROS type on the car | `sensor_msgs/Image` | Same pixels; gym may wrap as `niryo_moveit/Camera` (one `sensor_msgs/Image frame`) because that is what ROS-TCP already knows |
| Size / rate | **640×480 @ ~20 Hz**, `gscam` + `nvarguscamerasrc`, `flip_method=0` | Render **640×480**. Do **not** stream 20 Hz over gRPC (see §3). Capture **one frame per env step** (same cadence as `car_scene_data`) |
| Encoding | typically **`rgb8`** | Unity `ReadPixels` is RGBA. Strip alpha **in Unity or** in `frame_to_tensor` (already drops every 4th byte). Publish `encoding=rgb8`, `is_bigendian=0`, `step=width*3` |
| `frame_id` | CSI / `camera_visual` | `camera_visual` (or `csi_cam_0`) so Foxglove/TF stories match |
| Mount | URDF `base_link → camera_visual` (front stalk, slightly down) | Child of the car body at the **same translation/rotation** as `jetracer.urdf` `camera_visual`. Optical axis = robot **+x** (nose), **not** an orbit/chase cam |
| Intrinsics | `cam_640x480.yaml` (`K`, `D`, `P`) | Set Unity vertical FOV from `fy`: `vfov_deg = 2 * atan(height / (2 * fy)) * 180/π`. Horizontal follows aspect 640/480. Copy `fx, fy, cx, cy` into a shared `camera_intrinsics.json` used by gym + Jetson preprocess |
| Distortion | Real IMX219 has `D` | Optional: apply the same radial model as a Unity image effect, or leave undistorted and **undistort the real image** with OpenCV `undistort` before the net so both sides look pinhole |
| Near/far clip | n/a | Near ~0.05 m (hood/lidar mast in view like the real cam). Far ~20 m |

Check the live Nano any time:

```bash
rostopic hz /csi_cam_0/image_raw          # ~20
rostopic echo -n1 /csi_cam_0/image_raw    # height/width/encoding/step
# Foxglove: Image panel on /csi_cam_0/image_raw
```

Copy `fx, fy` from the yaml on the Jetson (`$(find jetracer)/config/camera_calibration/cam_640x480.yaml`) after you `cat` it — do not guess FOV from a “driver cam” prefab.

### 2. Unity gym changes

Existing leftovers (do **not** reuse as-is):

- `SimController` had `publishedCamera` + `CameraPublisher` targeting **`camera/overhead`** — disabled. That was a **top-down** view for the arm gym / `utility.frame_to_tensor`, not the JetRacer.
- `CarController.Cam2` / `WaymoDriverCamera` is a commented chase/driver cam — wrong FOV and pose.
- `SceneDataPublisher` does not attach pixels to `CarSceneData`. Keep it that way; images stay on their own topic so `Sphere` / 31-D rays stay stable.

Build a **forward CSI stand-in**:

1. Add a Unity `Camera` on the car prefab, parented like `camera_visual`. Name it `JetRacerCsiCamera`. Disable the AudioListener.
2. `targetTexture` = a 640×480 `RenderTexture` (`ARGB32`). `allowMSAA = false`.
3. Field of view = `vfov_deg` from §1. Aspect locked 4:3.
4. Culling: render the same collision/visual meshes the lidar would “see” as surfaces (no editor gizmos, no HUD).
5. On each **applied force** (same moment `CarSceneData` is sent), blit RT → `Texture2D`, `GetPixels32`, drop A, pack `sensor_msgs/Image` (`height=480`, `width=640`, `encoding=rgb8`, `data=RGB bytes`).
6. **Downsample before publish** for training (recommended **160×120** or **84×84** RGB). 640×480×3 ≈ 900 KB raw; gRPC JSON + base64 is worse. The policy never needs full CSI resolution. Keep a `full_res` flag for Foxglove dumps only.
7. Publish on **`camera/front`** (new), not `camera/overhead`.
8. Domain randomization (after the first vision job works): exposure, noise, motion blur, slight pitch/roll of the mount (±5°), and random JPEG quality if you compress.

Optional but useful: also publish `sensor_msgs/CameraInfo` once (latched) with the yaml `K`/`D` so a later undistort node is identical in sim and on the Nano.

### 3. ROS topics and data passing (sim plane)

The trainer still never opens a ROS socket. Frames must follow the same hops as `car_scene_data`:

```
Unity JetRacerCsiCamera
    -- ROS-TCP :10000 -->  ros-server  unity_node.py
    -- gRPC JSON :50051 -->  RobotApi  (sim-controller)
    --> course.scene_data_array / image tensor
    --> tf-agents TimeStep
```

**Register the topic.** `unity_node.py`’s table is **static**. Add:

```
'camera/front': RosPublisher('camera/front', Camera),   # niryo_moveit/Camera
```

(`niryo_moveit/Camera.msg` is `sensor_msgs/Image frame`. Reuse it so you do not invent a second image type. Do **not** expect `/csi_cam_0/image_raw` to exist inside `ros-server` — that name is Melodic/Jetson only.)

**`RobotApi` (`rl_agent/api.py`):**

- Today: `Subscribe('camera/overhead', 'niryo_moveit/Camera', _on_overhead_camera_frame)` and `GetOverheadCameraFrame()`.
- Add the same pair for `camera/front` (`_on_front_camera_frame` / `GetFrontCameraFrame`).
- Tie the wait to **`last_executed_cmd_id`** the same way `DoApplyForce` waits on `car_scene_data`. Either stamp the image `header.seq` / a custom field with `cmd_id`, or treat “latest frame after `FORCE_APPLIED`” as the step image. Uncorrelated 20 Hz frames will desync rays vs pixels.

**JSON on the wire** (gRPC `virtual_endpoint`) looks like the `sensor_msgs/Image` dict:

```
{
  "frame": {
    "header": {"seq": ..., "stamp": {...}, "frame_id": "camera_visual"},
    "height": 120,
    "width": 160,
    "encoding": "rgb8",
    "is_bigendian": 0,
    "step": 480,
    "data": "<base64 of height*step bytes>"
  }
}
```

`rl_agent/utility.py` `frame_to_tensor()` already decodes that shape for the overhead cam (assumes RGBA and strips alpha). For `rgb8` from the new publisher, skip the alpha strip or branch on `encoding`.

**Bandwidth:** one 160×120×3 frame per env step is ~56 KB raw (~75 KB base64). Fine. Full 640×480 at 20 Hz through gRPC will stall actors. Compress (`jpeg` in `encoding` + `sensor_msgs/CompressedImage`) only if you must send full res; then decode in Python with OpenCV/`tf.io.decode_jpeg`.

**Jetson side (deploy, later):** the car already has `/csi_cam_0/image_raw`. The policy node does **not** subscribe to `camera/front` or `niryo_moveit/Camera`. It subscribes to **`/csi_cam_0/image_raw`**, runs the **same** `image_to_obs()` as the gym (resize, RGB, `/255`, optional undistort), and feeds the CNN. `ros-server` / ROS-TCP never run on the Nano.

### 4. Feed and process so the model can consume it

The current networks cannot eat an image:

```
ActorDistributionNetwork(observation_spec)   # spec shape (31,) or (32,)
CriticNetwork((observation_spec, action_spec), joint_fc_layer_params=...)
```

Those are MLPs on a flat float vector. A 160×120×3 array is a **different `observation_spec`**. Options:

**A. Dict observation (recommended)** — new course `donut_camera`:

```
observation = {
  "vector": float32[31],          # existing donut_no_hint rays/speed/sideslip
  "image":  float32[H, W, 3],     # e.g. 84x84x3 or 160x120x3, values in [0, 1]
}
```

`observation_spec` becomes a dict of `BoundedArraySpec`s. Build the actor/critic with **`preprocessing_layers` / `preprocessing_combiner`**:

- `image` → `Conv2D` stack (e.g. 32/64/64, ReLU) → flatten → dense 256
- `vector` → dense 64
- concat → existing `fc_layer_params` / `TanhNormalProjectionNetwork`

Stamp SavedModels with `course_type=donut_camera` and **refuse** to load them on a 31-D bridge (and vice versa).

**B. Flatten into the vector** — `31 + H*W*3` floats. Simple, terrible: no weight sharing, huge MLP, will not transfer. Do not do this.

**Shared preprocess** (one Python function, two callers: gym `RobotaxiEnv` and Jetson policy node):

```
def image_to_obs(rgb_uint8, width, height, encoding) -> float32[H,W,3]:
    # 1. decode rgb8 (or jpeg)
    # 2. optional cv2.undistort with cam_640x480.yaml
    # 3. cv2.resize to (W_net, H_net), INTER_AREA
    # 4. / 255.0
    # 5. no ImageNet mean/std unless both sim and real use it
```

Call it:

- **Sim:** `RobotApi.GetFrontCameraFrame()` after `DoApplyForce` → `image_to_obs` → pack with `DonutCourseNoHint.scene_data_array()`.
- **Jetson:** callback on `/csi_cam_0/image_raw` (latest frame, or sync to the 10 Hz control loop) → same `image_to_obs`.

`RobotaxiEnv._reset` / `_step` today do `ts.restart(np.array(data_arr))` with a 1-D array. The camera course must return a **dict** `TimeStep.observation` (or a nested spec tf-agents accepts). `collect_training_data.set_observation_size` and the TFRecord BC path are **vector-only** — vision jobs need a new record layout (or skip BC / record dicts). Plan for that before the first `donut_camera` TRAIN job.

### 5. Suggested build order

1. Unity `JetRacerCsiCamera` at URDF pose + FOV from `fy`; dump a still next to a Foxglove grab of `/csi_cam_0/image_raw`. Hood/horizon should roughly match.
2. Publish `camera/front` once per `ApplyForce`; register in `unity_node.py`; `GetFrontCameraFrame` in `RobotApi`.
3. `image_to_obs` + unit test (known RGB ramp, encoding `rgb8`).
4. New course + dict spec + CNN preprocessor. Train from scratch (no 31-D weight load).
5. Domain-randomize lighting/noise; keep rays in the vector so the car still works if the camera glitches.
6. Jetson: same `image_to_obs` on `/csi_cam_0/image_raw`. Nano cannot run a fat Conv-SAC in tf-agents comfortably — plan TFLite / a tiny CNN, or run vision on a laptop sidecar first.

### 6. What not to do

- Do not pipe `camera/overhead` into the driving policy (wrong viewpoint).
- Do not publish 640×480×20 Hz through `virtual_endpoint`.
- Do not add pixels to `CarSceneData` / `Sphere.msg` (breaks every existing parser).
- Do not assume `/csi_cam_0/image_raw` exists in the Docker gym, or `niryo_moveit/Camera` exists on Melodic.
- Do not run a vision policy on `/cmd_vel` while `explore_foxglove` or `joy` is also publishing.

## Devices

| Device | Node |
|---|---|
| `/dev/ttyACM0` | motor/servo control board (`jetracer`) |
| `/dev/ttyACM1` | RPLiDAR |
| `/dev/input/js0` | gamepad / joystick (`joy_node`) |
| CSI camera (sensor id 0) | `gscam` |

## Gotchas

- **Motor direction was inverted (fixed):** `scripts/motor_direction_test.py`
  showed a `+linear.x` command rolls the car **physically backward**, while
  `/odom_raw` reports **positive** velocity — command and encoders agree with each
  other, but both are opposite the URDF / lidar `+x` (front). The `coefficient_a..d`
  params are **steering** PWM, not throttle, so flipping them does nothing to
  direction (an earlier attempt did that by mistake; they are restored to stock).
  Velocity is sent raw to the MCU in `jetracer.cpp`. The fix is
  **`invert_linear=true`** in `jetracer.launch`: the driver negates `cmd_vel.linear.x`
  before the serial packet, and reflects odom about the robot Y axis (`x`, `vx`,
  `yaw`, `omega`) so ROS `+x` is forward **and** a physical left turn reports
  **positive** `angular.z`. (The first invert left yaw/omega unchanged; the MCU
  computes `omega = v*tan(delta)/L` from the flipped `v`, so odom said the car
  was turning the opposite way — planner then steered into objects. Confirmed by
  `steer_direction_test.py _creep:=0.08`: wheels LEFT, `/odom_raw` `angular.z`
  **negative**.) To re-test: `roslaunch jetracer jetracer.launch` then
  `rosrun jetracer motor_direction_test.py` (first pulse **forward**) and
  `rosrun jetracer steer_direction_test.py _creep:=0.08` (LEFT pulse must show
  wheels LEFT **and** odom `angular.z` **positive**). Set `invert_linear` to
  `false` to undo. Steering hardware itself is not inverted: ROS/`TEB` `+angular.z`
  points the front wheels LEFT.
- **Frame-name mismatch (fixed):** the RPLiDAR now publishes `/scan` in
  **`laser_frame`**, matching the `base_footprint -> laser_frame` static transform,
  so the scan resolves into `base_footprint`/`odom` and lines up with the model.
  (Historically it published in `laser`, which had no transform.) If you ever see
  the scan detached again, verify `frame_id=laser_frame` in `rplidar.launch`.
- **`ttyACM` enumeration is not fixed:** the motor board and LiDAR both
  enumerate as `/dev/ttyACM*`; their numbering (`ACM0` vs `ACM1`) can swap across
  reboots / replug. If odometry or the scan is dead on boot, check which device
  came up where (`dmesg | grep ttyACM`) — udev rules by USB path/serial fix this
  permanently.
- **Never run two bringups at once:** duplicate same-named nodes evict each other
  and, with `respawn="true"`, loop forever (`Reason: new node registered with same
  name`); two motor drivers also fight over `/dev/ttyACM0` and the driver aborts
  (`jetracer ... exit code -6`). `start_jetracer.sh` guards against a double start,
  but a hand-run `roslaunch` bypasses that guard. Run exactly one
  `jetracer_bringup.launch` — see "Invoking the bringup" for how to detect/kill a
  duplicate.
- **Permissions:** the serial/I2C devices need `chmod 666` (done by
  `start_jetracer.sh` with `sudo`) before the nodes can open them.
- **ROS Melodic / Python 2** on the Jetson — keep node scripts Py2-compatible.
- **Line endings on scripts:** node scripts (`scripts/*.py`) must use **LF**, not
  Windows **CRLF** — a CRLF shebang makes `roslaunch` fail with
  `/usr/bin/env: 'python\r': No such file or directory` (exit 127). If you edit a
  script from Windows, run `dos2unix` (or `sed -i 's/\\r$//'`) on the Jetson.
- Other launch files in `~/catkin_ws/src/jetracer_ros/launch/` (not part of
  bringup): `slam.launch`, `gmapping.launch`, `cartographer.launch`,
  `hector.launch`, `amcl.launch`, `nav.launch`, `move_base.launch`,
  `slam_nav.launch`, `keyboard.launch`, plus audio/ASR/TTS.
