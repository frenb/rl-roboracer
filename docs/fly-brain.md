# Fly brain: driving the car from a frozen connectome

An experiment in driving the car with fly.ai's *Drosophila* connectome — 166,700
neurons and 25,582,938 connections of **fixed, unlearned wiring**. Nothing about
the brain is trained; the only fitted part is a linear readout of its output.
The full plan, including the measurements behind each design choice, is
`docs/flybrain-driver-plan.md`.

```
31-D observation ──► RayEncoder ──► Step RPC ──► FlyBrain (166,700 neurons, CUDA)
                                                        │
   (accel, steer) ◄── ridge readout ◄── 1,314-wide descending-neuron trace
```

## Two terms this section uses

**Ridge training** (the *ridge readout*, or just *the ridge*) — the only fitted
part of the whole pipeline. The 1,314-wide trace is mapped to `(accel, steer)`
by ridge regression, i.e. L2-regularized linear least squares, fit **offline on
the expert demo corpus**. The target is what the PID expert did on each frame,
so this is behaviour cloning, not reinforcement learning: no reward is involved
and no gradient reaches the connectome, whose wiring stays frozen throughout.
Held-out R² is 0.62 for steering and 0.18 for acceleration; see *Encoder and
readout* below for why that second number is what caps the policy at 6.87.

The contrast worth holding onto: the `fly_donut` course replaces this fitted
linear map with SAC's actor, which learns the *same* readout from reward
instead. Everything upstream of the arrow — encoder, brain, trace — is
identical; only who fits the last step changes. That is the whole point of
Part 5 of the plan.

**Lane work** — the unbuilt change that would let more than one environment
train against the brain at the same time. The `fly-brain` service holds **one**
`FlyBrain` with a single voltage array behind a single lock, and `Reset` and
`Step` both mutate it. Two environments sharing it do not race and do not
error; they *interleave*, and each reads a trace shaped by the other's rays. It
shows up only as a slow, silent drift: a determinism check run while an eval
happened to be mid-job returned `max|Δtrace| = 1.6` for a fixed seed instead of
the usual `0.0`, and nothing anywhere logged a complaint.

A *lane* is one column of a batched brain — `FlyBrain` already stores voltages
as `(n_neurons, batch)`, and `Trace(aggregate="batch")` already returns one
column per fly. The work is the four things that don't exist yet: coalescing
the N concurrent `Step` calls into one batched step (you cannot advance a
single column — `step()` moves them all), a **per-lane `Reset`** for the
asynchronous episode boundaries (today's `reset()` clears every column and
reseeds globally), sending explicit zeros for lanes whose cue is inactive
rather than dropping the population, and plumbing the existing `actor_index`
through to the course, where it currently stops at log prefixing. You need
`num_envs + 1` lanes, since multi-env runs build a separate eval env.

Until that exists, **every fly job runs at `--num-envs 1`**. Note also that
batching is not free: measured 13.3 → 8.4 ms per fly going from batch 1 to 8,
so it buys about 1.6x, not 8x.

## Why it's a separate container

`flybrain` needs Python 3.10; sim-controller is pinned to Python 3.8 by
TF 2.7 / tf-agents 0.11. The two cannot share an interpreter, so the brain runs
as its own `fly-brain` service and the trainer reaches it over gRPC on `:50061`.
`rl_agent/fly_brain/client.py` is the only thing that knows this.

The contract is `protos/fly_brain/proto/fly_brain.proto`. Numeric payloads are
**raw little-endian buffers in `bytes` fields, not JSON** — a step returns 1,314
floats at 10 Hz and JSON-encoding that every step would be wasteful.

| RPC | Purpose |
|---|---|
| `Info`     | Static facts: neuron/connection counts, device, `dt`, trace width |
| `Reset`    | Clear voltages and reseed. **Required at every episode boundary** |
| `Step`     | Advance `substeps` times under an injection; return the trace |
| `Snapshot` | Current overlay activity without advancing the brain |
| `Geometry` | Static overlay positions + edges. Fetch once |
| `Cells`    | Resolve cell types or a named group to neuron indices |
| `Sectors`  | Split a population into k spatial sectors (retinotopic encoder) |

Two facts drive most of the integration. The gym publishes scene data every
0.1 s and the brain's `dt` is 0.020 s, so **one control step is five brain
steps** (`SUBSTEPS = 5`, about 13.4 ms). And the brain is **stateful** — voltages
carry across steps — so a new episode must not inherit the last one's state.

## Encoder and readout

`rl_agent/fly_brain/encoder.py` turns the 31-D observation into injections
aimed at the looming detectors (LC4, LPLC2, LPLC1, LC10a). Two quirks of the
sim are handled there: Unity leaves a ray's distance **untouched on a miss**, so
an exact 0 is read as maximum openness rather than a wall in your face, and
closing rate is clamped by car speed to reject impossible jumps.

The readout is a ridge regression fit on the expert demo corpus (step 5), living
at `/saved_models/robotaxi/FlyPyPolicy/0/readout.npz`. Held-out R² is **0.62 for
steering** and **0.18 for acceleration**. Note what that second number means in
practice: predicted accel falls below the course's own 0.05 action floor on
about half of all frames, so the car is effectively pinned near minimum throttle
and drives at 1.9–2.4 m/s against SAC's 5.4.

A measured negative result worth not repeating: a *retinotopic* encoder driving
7 sectors per eye raised the information available to the brain (R² ceiling
0.687 → 0.873) and made its output **worse** (0.620 → 0.581). Driving sectors
one at a time, within-eye pairs have cosine similarity 0.476 against 0.460
across eyes — the brain reads left-versus-right well and sector-versus-sector
essentially not at all. The flat 4-scalar encoder is the default.

## Running it

`FlyPyPolicy` (`rl_agent/fly_brain/policy.py`) is an ordinary tf-agents
`PyPolicy`, so it goes through the **normal EVAL dispatch** next to
`RandomPyPolicy` and lands a leaderboard row directly comparable to SAC. Queue
an EVAL job with `model_type: FlyPyPolicy` (it's in the dashboard's job form),
then run the trainer with `--num-envs 1` — see *lane work* above for why that
is a correctness requirement and not just a throughput choice.

The policy resets the brain and encoder on every `StepType.FIRST` and seeds the
reset with the episode index — a fixed per-episode seed, without which the
spiking noise makes runs irreproducible and incomparable against greedy SAC's
deterministic `tanh(μ)`. On first use it registers its own `models` record,
since there is no TRAIN job to have created one.

| Policy | AverageReturn | Goals/episode | Speed |
|---|---|---|---|
| `RandomPyPolicy` | ~1.0 | — | — |
| `FlyPyPolicy` | **6.87** | 8–13 | 1.9–2.4 |
| SAC `7573_step_87314` | 75.7 | 78–100 | 5.2–5.5 |

So the frozen connectome is worth roughly **7x a random policy and a tenth of
SAC**. It holds a lane for a few hundred steps rather than crashing at once.

## Training against the brain: the `fly_donut` course

Everything above runs the connectome behind a *fitted* readout. The `fly_donut`
course hands that last step to SAC instead — the 1,314-wide trace becomes the
observation and the actor learns the mapping from reward. It exists because the
ridge could not fit throttle (R² 0.18), which is what pins the car near minimum
throttle and caps it at 6.87.

`FlyDonutCourse` subclasses `DonutCourseNoHint` and leaves `scene_data_array()`
as the 31-D vector, so rewards, stuck detection, the curriculum and the per-step
stats read exactly what they read on `donut_no_hint`. Only `policy_vector()`
changes. Its observation spec is a flat `(1314,)` bounded `[0, 5.52]`, both read
from the service's `Info` at construction rather than from any literal. The
episode reset hangs off `BaseCourse.on_episode_start()`, called at the top of
`RobotaxiEnv._reset()` — the only hook that fires after the terminal observation
is packed *and* on episodes that ended in failure.

Two constraints are structural rather than temporary:

- **No demos, ever.** The trace depends on the brain's voltages and so on the
  whole episode to date, so a recorded observation cannot be reconstructed from
  a stored scene. `fly_donut` is in `COURSES_WITHOUT_DEMOS`, so `do_job` refuses
  DEMO and BC_TRAINING_ONLY. TRAIN is from scratch with no expert bootstrap,
  like the camera courses.
- **`--num-envs 1`**, for the reason under *lane work* above.

To run one: pick `fly_donut` in the dashboard's job form with job type TRAIN and
the `fly-brain` service up. The actor prints
`[fly_donut] obs vector(1314,) (of scene(31,))` once, which is the line
confirming the brain is in the observation path. Measured at ~0.23 s/iteration
(0.09 collect, 0.14 learner), putting the 87k-step budget `donut_no_hint` was
measured on at roughly 5.6 hours.

**If training destabilizes, suspect replay first.** The observation is
history-dependent, so off-policy replay is learning from features it cannot
exactly reconstruct.

## Runbook: launching a `fly_donut` TRAIN job

Worked example — 50,000 iterations on `wCourseJetRacer2026.09.20-v13`.

**1. Start the trainer at one env**, teeing to the path
`scripts/Monitor-Job.ps1` reads. Pass no `--num-envs`; the default of 1 is the
only safe value, for the reason under *lane work*.

```powershell
docker compose exec -T sim-controller sh -c "pkill -f 'robotaxi.py'; sleep 3"
docker compose exec -d sim-controller bash /python_ws/src/run_trainer.sh
```

**2. Queue the job** from the Jobs tab's **New job** form:

| Field | Value |
|---|---|
| Job type / Model type | `TRAIN` / `SacAgent` |
| Course | `fly_donut` |
| Gym | `wCourseJetRacer2026.09.20-v13` |
| Iterations | `50000` |
| Reward design | `Goal-count speed (v4, TIME_COST 0.0073)` |
| Experiment design | `AWAC + No-BC + curriculum (5-stage) + reward_scale1` |
| Demo job id / Skip first eval | blank / off |

Those two designs are the ones behind the `SacAgent/7573_step_87314` baseline,
so matching them is what lets a later comparison mean anything. AWAC is
force-disabled on a course with no demo corpus, so that half of the experiment
design is inert; the 5-stage curriculum still applies and stamps its own
geometry per stage, so leave the corner-radius and curvature boxes empty.

You do not launch Unity yourself — `do_job` POSTs the job's `gym_file_path` to
`/set_desired_gym`, and each `RunClientWrapper.ps1` supervisor hot-swaps its
binary to match.

**3. Confirm pickup.** Three lines appear within ~30 s, the last being the one
that proves the connectome is in the observation path rather than the ray
vector:

```
main: fly_donut TRAIN from scratch (skip expert TFRecords, BC pretrain, AWAC)
fly_donut: trace_len=1314 obs_max=5.52 substeps=5 device=cuda
[fly_donut] obs vector(1314,) (of scene(31,))
```

**4. Monitor** with `.\scripts\Monitor-Job.ps1 -Watch`. Judge progress by
`eval/goals_per_episode_this_eval`, not `avg_return`, which is not comparable
across jobs with different reward designs. At ~0.23 s/iteration, 50,000
iterations is roughly 3.2 hours plus eval overhead — note that the baseline
checkpoint is at step 87,314, so 50k is a real run but not yet the equal-budget
comparison.

## The overlay

`rl_agent/fly_brain/viz.py` publishes two topics; `FlyBrainViz.cs` renders them
(auto-attached by `SimController`, **B** to toggle). It stays hidden until
activity arrives, so a SAC job doesn't get a frozen brain over the track.

| Topic | Rate | Payload |
|---|---|---|
| `fly_brain_geometry` | every 10 s | 359 KB: 19,225 soma positions (edges off by default) |
| `fly_brain_activity` | 10 Hz | 25 KB: one intensity byte per neuron (~256 KB/s) |

The display subset is not the whole brain — 166,700 neurons can't be drawn at
frame rate — but it is not just the driving circuit either. It is two
populations:

- **The circuit**, 3,225 neurons: 720 sensory, 1,292 descending, 16 named
  command neurons, and 1,200 relay interneurons picked by (weight received from
  sensory) × (weight sent to descending), so the two-hop path from the looming
  detectors to the motor output is visible rather than a gap.
- **The silhouette**, a seeded uniform sample of 16,000 more
  (`FLY_CONTEXT_N`). Every neuron in the circuit is a *brain* neuron — the
  somas are all in the head — so on its own it draws a blob with no body. The
  sample is uniform rather than spatially even on purpose: neuron density
  varies enormously between neuropils, and preserving that is what makes the
  optic lobes read as dense and the ventral nerve cord as sparse.

Every numeric field is base64 of a little-endian buffer, geometry included:
**359 KB against ~680 KB** for the same arrays as JSON numbers, and Unity's
`JsonUtility` would otherwise allocate and parse ~58,000 boxed floats on each
resend instead of doing one `Convert.FromBase64String` plus a `Buffer.BlockCopy`.
Positions arrive normalized to a unit box, so Unity applies one scale factor
instead of raw MaleCNS soma coordinates (which run to ~134,000 on the z axis).

The two populations are also what the palette is built around. The circuit
keeps its per-role colours (cyan sensory, gold command, orange-red descending)
and dims toward black when quiet. The silhouette instead crosses a *two-colour*
ramp, cold slate at rest to amber when firing, so active neuropils burn out of
an otherwise dark cloud.

`FLY_SNAPSHOT_FULL` is the one that decides whether the overlay looks alive.
It sets what counts as maximum brightness, in accumulated spikes. It was
originally `1/(1-decay)` = 8.0, i.e. "fires on every substep" — which only a
few tonically-driven cells ever do, so measured over 200 frames **591 neurons
sat pinned at 255 and supplied 42% of all the light on screen**, while genuine
bursts (1–2 spikes' worth) landed near a byte of 40 and disappeared under the
1.6 display gamma. The spiking was there the whole time; it was three stops
too dim to see against a static core. At 1.8 nothing is permanently saturated
and the median neuron's variation over time is 4.5x larger.

That reads only against a dark background, and it assumes the scene gives it
one. **The overlay draws no background of its own.** If the view camera goes
back to clearing with the skybox, a resting connectome on mid-grey is close to
invisible. Note the camera's *Background* colour field has no effect while
Clear Flags is `Skybox` (`m_ClearFlags: 1`); it is the skybox the top-down
camera sees the underside of. Changing the skybox **material** instead would
also change what the car's CSI camera renders — `JetRacerCsiIntrinsics` pins
that camera to `CameraClearFlags.Skybox` — and therefore what every policy
observes, so prefer the view camera's Clear Flags.

A payload that size needs the patched ROS-TCP connector; see below.

## The connector's 64 KB message ceiling

`ROSConnection.ReadMessageContents` used to read with
`while (networkStream.DataAvailable && bytesRemaining > 0)`. `DataAvailable`
only reports what is buffered *right now*, so the loop exits the moment the
sender pauses mid-message and hands back a half-filled buffer. Under ~64 KB the
whole message is normally buffered before the first `Read` and nothing looks
wrong, which is why this survived for years; the geometry payload hit it on
every single send and arrived as truncated JSON, with the only symptom being

```
[FlyBrainViz] geometry parse failed: JSON parse error: Missing a closing quotation mark in string.
```

in `unity/Builds/instances/0/Player.log` — while activity frames, being small,
kept flowing and kept the (stale) overlay animating.

The loop now runs until the declared length is satisfied. The package arrived
as a git dependency and therefore lived in `Library/PackageCache`, which Unity
regenerates, so it is now **embedded** at
`unity/Packages/com.unity.robotics.ros-tcp-connector/` with the git entry
dropped from `manifest.json` and `packages-lock.json`. Anything publishing to
Unity over ~64 KB depends on this patch.

Geometry is resent on a slow heartbeat rather than once, because the routing
table below gives Unity no way to ask for a resend — a client that connects or
reloads its scene after the first send would otherwise draw nothing forever.

## The ROS routing table (the thing that silently breaks new topics)

`docker/ros_server/ROS/src/roboracer/scripts/unity_node.py` starts the
ROS-TCP endpoint with a dict deciding **which topics cross the ROS↔Unity bridge
and in which direction**:

- `RosPublisher(topic, Msg)` — **Unity → ROS**. Unity sends it; the endpoint
  publishes it into ROS (`car_scene_data`, `sim_status`, `camera/front`).
- `RosSubscriber(topic, Msg, tcp_server)` — **ROS → Unity**. The endpoint
  subscribes to the ROS topic itself and forwards each message down the socket
  to the Unity client (`sim_command`, `policy_rollouts`, both fly-brain topics).

```python
'fly_brain_geometry': RosSubscriber('fly_brain_geometry', String, tcp_server),
'fly_brain_activity': RosSubscriber('fly_brain_activity', String, tcp_server),
```

**This table is static.** The ROS-TCP connector embedded in Unity never sends
dynamic `RegisterSubscriber` syscommands, so `ros.Subscribe<StringMsg>(...)` in
C# only registers a callback on the Unity side. If the topic isn't listed here,
nothing on the ROS side is listening or forwarding: the publish succeeds, no
error appears anywhere, and the overlay just stays dark. This is the single most
likely reason a newly added topic does nothing.

It also explains a confusing asymmetry while debugging. `check_rollouts.py`
(which takes a topic argument) subscribes through ros-server's **gRPC** API, as
does the publisher — that leg never touches the routing table, so data flows and
verifies fine while the Unity hop is still missing:

```powershell
docker compose exec -T -w /python_ws/src sim-controller `
  python check_rollouts.py ros-server-0:50051 25 fly_brain_activity
# Received 347 message(s). ~18.0 Hz. OK - data is flowing.
```

Because only `./rl_agent` is bind-mounted into ros-server, `unity_node.py` is
baked into the image. **Editing it does nothing until you rebuild**:

```powershell
docker compose build ros-server
docker compose -f docker-compose.yml -f compose/scale.yml up -d --force-recreate `
  ros-server ros-server-1 ros-server-2 ros-server-3
```

## Knobs

All optional; the defaults are what the numbers above were measured with.

| Variable | Default | Effect |
|---|---|---|
| `FLY_BRAIN_TARGET` | `fly-brain:50061` | Where the trainer finds the brain |
| `FLY_READOUT` | `/saved_models/robotaxi/FlyPyPolicy/0/readout.npz` | Ridge readout |
| `FLY_VIZ_ENABLED` | `1` | Overlay publishing off with `0` |
| `FLY_VIZ_HZ` | `10` | Activity publish rate |
| `FLY_VIZ_GEOMETRY_S` | `10` | Geometry resend period |
| `FLY_VIZ_DISPLAY_SIZE` / `_POINT_SIZE` / `_OFFSET` / `_SPIN` | `60` / `0.09` / `2,40,130` / `0` | Where and how big the brain is drawn |
| `FLY_VIZ_AXES` | `-z,y,x` | Which connectome axis goes on which Unity axis |
| `FLY_VIZ_ROTATE` | `0` | Turns the picture in the screen plane, signed degrees |
| `FLY_VIZ_DEPTH_SCALE` | `0.12` | How much of the camera-facing axis to keep |
| `FLY_VIZ_EDGES` / `_EDGE_ALPHA` | `0` / `0.40` | Draw connections between neurons |
| `FLY_VIZ_ACT_FLOOR` | `0` | Floor under the activity byte, i.e. resting brightness |
| `FLY_SNAPSHOT_TAU` / `_FULL` | `0.15` / `1.8` | Overlay decay time, and the accumulated trace that counts as full brightness (**fly-brain**, needs a restart) |
| `FLY_CONTEXT_N` | `16000` | Silhouette neurons (on **fly-brain**, needs a restart) |

`FLY_VIZ_AXES` is the one worth explaining. These are the overlay parent's
*local* axes and that parent is rotated, so they are not the screen axes you
would guess: **Unity x is screen-vertical** (up at +x), **Unity z is
screen-horizontal** (left at +z), and Unity y is the one the camera looks down
and therefore the one thrown away. Against the connectome's own axes —
x is bilateral width (span 91k), y front-to-back (64k, the thinnest), z the
body axis (124k, brain low and nerve cord high) — `-z,y,x` stands the animal
upright with the brain on top, the way fly.ai's render has it.

The placement knobs are published inside the geometry payload rather than left
on the Unity inspector, because **a build has no inspector** — without that,
every "the brain is off screen" would cost an Editor rebuild.

Two of these defaults are less obvious than they look:

`_OFFSET` parks the brain in the empty area left of the track, under the ROS
HUD, which also puts it over the camera's flat background instead of grass.
The axes are the overlay parent's *local* ones and that parent is rotated, so
they do not read as screen directions: measured against this camera, `+z`
moves the brain left at 3.1 px/m and `+x` moves it up at 2.5 px/m (on a
1250 px-wide window). Calibrate with two known offsets rather than guessing.

`_ACT_FLOOR` exists because Unity renders a resting neuron at a hardcoded 0.18
of its role colour and 0.20 alpha, and the connectome is quiet most of the
time, so nearly the whole cloud sits at that floor and disappears into the
background. The byte is rescaled into `[floor, 255]` rather than clamped, so
a firing neuron still separates from a resting one. The honest fix is a Unity
field, but this keeps the knob on the side that doesn't need an Editor build.

## Gotchas

- **GPU contention.** TensorFlow grabs most of the card by default and the brain
  needs ~210 MB plus working space. `TF_FORCE_GPU_ALLOW_GROWTH=true` on the
  trainer, and stay at `--num-envs 1` for fly experiments.
- **Replay stores history-dependent observations.** The trace depends on the
  brain's voltages, which depend on the whole episode so far. Workable — the
  trace *is* the observation — but suspect it first if training goes unstable.
- **This is a demo, not an emulation.** Point neurons, one global parameter set,
  no dendrites, no neuromodulators, no plasticity, transmitter sign from a rough
  rule, nothing validated against recordings from real flies. Expect an
  interesting result, not a good driver.
