# How the fly brain drives the car

A walkthrough of the working system: what happens on one control step, where
behaviour cloning ends and reinforcement learning begins, and how a frozen
166,700-neuron connectome ends up steering a car in Unity.

This is the "how it works" companion to
[`flybrain-driver-plan.md`](flybrain-driver-plan.md). That document is the
decision record — twelve steps, what was measured at each, and which ideas were
falsified along the way. This one describes the result. Where a number below
has a story behind it, the plan has the story.

Every figure here was read off the running system on 2026-09-21, not copied
from the plan.

---

## The one-paragraph version

The connectome is a **reservoir**: a large recurrent network with fixed wiring
that nothing ever trains. Raycast distances from the car are translated into
drive for four visual neuron populations; the brain runs; and the resulting
activity across its 1,314 descending neurons becomes a feature vector. All the
learning happens in the **readout** that turns those 1,314 numbers into
`[acceleration, steering]`. Fitting that readout on expert demonstrations is
behaviour cloning, and it drives the car badly. Letting SAC learn it from reward
instead is the reinforcement learning, and that is what is training now.

```
Unity  ──►  31-D scene vector  ──►  encoder  ──►  fly connectome  ──►  descending trace  ──►  readout  ──►  [accel, steer]  ──►  Unity
 10 Hz       speed, sideslip,        4 cues       166,700 neurons        1,314 floats        ridge (BC)
             29 ray distances                     FROZEN                                     or SAC (RL)
                                                       │
                                                       └──►  spike snapshot  ──►  ROS  ──►  Unity overlay
```

Nothing inside the brain learns. Its weights are the MaleCNS v1.0 wiring
diagram, fixed at load. The only trainable thing in the whole picture is the
last arrow but one.

---

## What is actually running

Three containers cooperate, and the split is not architectural taste — the
connectome library needs Python 3.10 and the trainer is Python 3.8, so they
cannot share a process.

| Container | Role in this pipeline |
|---|---|
| `fly-brain` | Holds the connectome and serves `Reset` / `Step` / `Cells` / `Sectors` / `Snapshot` / `Geometry` / `Info` over gRPC on `:50061`. Needs a GPU. |
| `sim-controller` | The trainer. Runs the encoder, calls the brain once per env step, and runs SAC. |
| `ros-server` | Relays the overlay topics to Unity. |

The live service reports:

```
n_neurons = 166700     n_connections = 25582938    device = cuda
dt = 0.020 s           trace_len = 1314            trace_tau = 0.100 s
tonic = 0.14           gain = 3.0                  n_display = 19225
```

`FlyBrainClient` is the only class on the trainer side that knows any of this
exists. Everything above it — the course, the env, SAC — sees ordinary NumPy.

---

## One control step, end to end

Unity publishes scene data every 0.1 s, so one env step is 0.1 s of simulated
time. Here is what happens inside it.

### 1. Unity produces a 31-D scene vector

The course is `fly_donut`, which subclasses `DonutCourseNoHint`, and
`scene_data_array()` is **not** overridden. So the scene vector stays exactly
what every other no-hint job sees:

| Index | Content |
|---|---|
| 0 | speed |
| 1 | sideslip (angle from velocity to heading) |
| 2–30 | 29 raycast distances, in metres |
| 16 | forward clearance (the 0° ray) |

Leaving this alone is load-bearing. Rewards, stuck detection, the curriculum
and the per-step statistics all keep reading the same 31 numbers they read on
`donut_no_hint`, so results stay comparable and the reward design is untouched.
Only what the *policy* sees changes.

Two details about the rays that the encoder depends on. They are **not sorted
by angle** — indices 3/4 are −30°/−60° and 28/29 are +60°/+30°, so the two
transpositions sit inside one side each. And a ray reporting exactly `0.0`
never hit anything since the car was created, so it means "nothing out there",
not "obstacle on the bumper".

### 2. The encoder turns 29 rays into four numbers

`RayEncoder` computes two cues per side. This is the entire translation between
the car's world and the fly's, and it is four floats wide.

**Looming** is inverse time-to-contact, `−ṙ/r`, approach only:

```python
closing = (self._prev - r) / self.dt           # +ve when approaching
cap     = CLOSING_SLACK * max(abs(speed), SPEED_FLOOR)
inv_tau = np.clip(closing, 0.0, cap) / r
```

Deliberately not a proximity term — LPLC2 is a *looming* detector, so a wall
parked at 1 m and a wall rushing in from 1 m must not look alike. The clamp to
the car's own speed exists because the rays rotate with the car, so a ray
sliding off a wall edge reports a jump that is not motion; 23.5% of per-ray
steps in the demo corpus move further than the 6.7 m/s top speed allows.

**Chase** is a left-right *contrast* in openness, not a level, so a straight
road drives neither side and the asymmetry is what steers.

Both are weighted by `cos(angle)` before aggregation. Without that, the ±90°
rays — median 6.3 m against 25 m straight ahead — are the per-side minimum on
every single frame, on both sides, and the encoder reads the track wall instead
of the road. `cos(90°)` is exactly 0, which deletes them.

The four cues map onto populations whose behaviour is documented in the fly
literature and was re-measured here:

| Cue | Neuron types | Live cell count | Fly function | Racing equivalent |
|---|---|---|---|---|
| `loom_L` | LC4 + LPLC2, left | 165 | looming → escape | obstacle closing → turn away |
| `loom_R` | LC4 + LPLC2, right | 146 | | |
| `chase_L` | LC10a, left | 135 | courtship pursuit → steering | free space that way → steer toward |
| `chase_R` | LC10a, right | 140 | | |

Gains are calibrated so the 99th percentile of each cue over the demo corpus
lands at `MAX_AMOUNT = 0.8`, the "strong injection" level, so an ordinary frame
uses the dynamic range rather than clipping.

Populations whose cue is zero are dropped from the request rather than injected
with `0.0`, which keeps the common straight-road frame small on the wire.

### 3. The brain runs five substeps

`dt` is 0.020 s and a control step is 0.1 s, so `SUBSTEPS = 5`. The injection
goes in on every substep, along with a tonic photoreceptor drive of
`EYE_DRIVE = 0.45` — without it the brain is too quiet to respond.

Note what is *not* used: the photoreceptor/eye route as a signal path. Vision
does nothing at these settings, because histamine is inhibitory and real lamina
neurons are graded, which a spiking point-neuron model cannot reproduce. The
working path injects the feature-detector neurons directly.

### 4. The trace comes back, 1,314 wide

The service keeps a decaying trace over every neuron whose superclass is
`descending_neuron`: 656 left, 648 right, 10 midline. A spike adds 1.0 and the
trace decays by `exp(−dt/τ)` each substep.

The reason for a decaying trace rather than an instantaneous rate is concrete.
The named command neurons — DNa02, DNp01, DNg100 — are **one neuron per side**.
At this rate a single neuron contributes 0–3 spikes per control step, so any
hand decoder reading a rate off DNa02 is reading a near-binary signal. Reading
all 1,314 through a decay filter is what makes the signal continuous enough to
regress on.

The course derives its observation bounds from that decay rather than guessing:

```python
decay   = float(np.exp(-info.dt / info.trace_tau))   # 0.8187
ceiling = float(1.0 / (1.0 - decay))                 # 5.52
```

which is why the live job logs `fly_donut: trace_len=1314 obs_max=5.52
substeps=5 device=cuda`.

### 5. The readout turns 1,314 numbers into two

This is the only learned step, and it is where the two routes diverge. Both are
covered in the next section.

The action is `[acceleration, steering]` — **acceleration first** — bounded to
`[0.05, 1.0]` and `[−1.0, 1.0]`. Writing steering into slot 0 sends steering
commands to the throttle, which is the kind of bug that produces a
plausible-looking car that drives badly for reasons you will not find.

### Where the step is taken from

The single most important structural decision is *where* in the code the brain
gets stepped, and it is not in a policy:

```python
def policy_vector(self, data_arr):
    """Advance the brain one control step and hand back its trace."""
    _, inject = self._enc.encode(np.asarray(data_arr, np.float32), self._cells)
    trace, snap, spikes = self._client.step(
        inject, substeps=self._substeps, want_snapshot=self._viz.enabled)
    ...
    return trace
```

`policy_vector` is a `BaseCourse` hook meaning "the part of the scene the policy
is allowed to see". The env calls it in `_pack_observation`, so **the trace is
the observation**: it lands on the `TimeStep`, goes into the replay buffer, and
SAC's critic sees it. Stepping the brain inside a policy instead would keep it
out of replay entirely and there would be nothing for RL to learn against.

Two early returns in `robotaxi_env.py` had to go for this to work — both
`_pack_observation` and `_as_obs_time_step` used to short-circuit for
non-dict observations, which meant a flat-vector course had no way to change
what the policy saw. With the `BaseCourse.policy_vector` default being the
identity, routing everything through it is a no-op for `donut` and
`donut_no_hint`.

---

## The two readouts: behaviour cloning, then reinforcement learning

The same encoder and the same brain feed both. Only the box at the end differs.

### Route A — ridge regression on expert actions (behaviour cloning)

`FlyPyPolicy` is a frozen pipeline: encoder → brain → a linear readout fit
offline against the expert demo corpus. Nothing learns at run time.

It was fit by replaying 100 episodes (100,000 frames) of the existing expert
corpus through the brain and solving a standardised ridge, splitting **by
episode** rather than by row — consecutive rows are 0.1 s apart and highly
correlated, so a random row split leaks the answer across it.

Held-out R², with the controls that make it interpretable:

| Features | R² accel | R² steer |
|---|---|---|
| descending trace, 1,314 features | 0.176 | 0.620 |
| the 4 encoder cues — the brain's entire input | 0.045 | **0.687** |
| raw 31-D observation (upper reference) | **0.208** | **0.747** |

Read the middle row before the first one. **The brain loses steering
information**: handed cues worth 0.687 it returns 0.620. Without that control,
0.620 reads as a success. **Throttle is the opposite and is the interesting
result** — 0.176 against 0.045 for its own instantaneous input. The brain is
supplying something its input does not have, because it is a recurrent system
whose state integrates history, and throttle depends on where the car is in a
manoeuvre rather than on the current frame.

Driving with it works, in the sense that it is unmistakably better than chance
and unmistakably worse than SAC:

| Policy | AverageReturn | Goals/episode | Speed |
|---|---|---|---|
| RandomPyPolicy | ~1.0 | — | — |
| **FlyPyPolicy** (ridge) | **6.87** | 8.0 / 12.7 | 1.9–2.4 m/s |
| SAC on `donut_no_hint` | 75.7 | 78–100 | 5.2–5.5 m/s |

The failure mode is specific and it points directly at what to do next. The
ridge's acceleration prediction falls **below the course's own 0.05 action
floor on ~54% of frames**, so the clip to the action spec pins the car near
minimum throttle roughly half the time. It is not crashing immediately; it is
holding a lane at walking pace and eventually leaving the track. Steering
(0.620) is what keeps it on the road; throttle (0.176) is what makes it slow.

### Route B — SAC learns the readout from reward (the RL)

`fly_donut` is the same pipeline with the ridge deleted. The trace is the
observation, SAC is the readout, and the readout is learned from reward instead
of regressed onto expert actions.

This targets exactly the channel behaviour cloning could not fit. Throttle has
no hand-decodable source either — DNg100, the forward-walking neuron, measured
+0.0 to +0.1 under every stimulus and strength tested, so neither looming nor
chase drives it. Learning it from reward is the direct answer, and ridge is
linear where SAC is not, so there is headroom the R² numbers do not measure.

Concretely, the observation width goes from 31 to 1,314, which takes the
actor's first layer from about 16k weights to about 673k. Everything else in
the trainer — replay, eval, the leaderboard, TensorBoard — works unchanged.

| | Route A (BC) | Route B (RL) |
|---|---|---|
| Readout | ridge, fit offline | SAC actor, learned online |
| Linear? | yes | no |
| Learns from | expert actions | reward |
| Where the brain is stepped | inside the policy | inside `policy_vector`, so the trace reaches replay |
| Trace in the replay buffer? | no | yes |
| Job type | EVAL, `model_type: FlyPyPolicy` | TRAIN, `course_type: fly_donut` |

---

## Running a FlyPyPolicy eval

Either dialog can queue one. The Models tab is fewer clicks and fills most of
it in for you.

### From the Models tab (recommended)

1. **Models** tab, tick the row at `/saved_models/robotaxi/FlyPyPolicy/0`.
2. Click **+ Eval selected**.
3. Leave every picker on **From each model** — the row already carries
   `model_type: FlyPyPolicy` and `course_type: donut_no_hint`.
4. Set **Number of trials**, pick a **Gym**, click **Submit**.

A blue note appears in the dialog confirming the job will be queued without a
location, which is what makes it valid. If you override something into an
invalid combination the note turns amber and Submit refuses, leaving the dialog
open so you can fix it.

### From the Jobs tab

1. **Jobs** tab → **+ New job**.
2. **Job type** `EVAL` — this reveals the EVAL-only rows.
3. **Model type** `FlyPyPolicy`.
4. **Model location** — **leave empty**.
5. **Course** — **`donut_no_hint`**.
6. **Number of trials**, **Gym**, **Reward design**, then **Create job**.

Leave **Num iterations**, **Training steps** and **Episodes per trial** blank;
the first two are unread on this path and the third defaults to 5.

Prerequisites either way: the `fly-brain` service must be up, and the step-5
readout must exist at `/saved_models/robotaxi/FlyPyPolicy/0/readout.npz`. The
trainer registers a `models` row at `/saved_models/robotaxi/FlyPyPolicy/0` on
first use, so the result lands on the leaderboard next to SAC automatically.

### Why there is no location, and what happens if there is one

`run_flypolicy()` builds its policy from the frozen connectome plus the ridge
readout and never opens `location`, so a job carrying one would run happily and
write a fly-baseline score under the name of whichever checkpoint you picked.
`do_job` refuses the combination outright:

```
EVAL rejected for job <id>: FlyPyPolicy EVAL builds its policy from scratch and
cannot load a saved snapshot, but this job carries location='...'. Running it
would have scored the FlyPyPolicy baseline under that checkpoint's name.
```

The job goes to FAILED within seconds of pickup, which reads as "it completed
instantly and is broken". The reason is written to the job's `eval_error`
field, so **read that before deleting the job** — deleting it destroys the only
copy of the explanation.

The guard is not theoretical. Job `6ab07220c2736b9905844046` did exactly this
on 2026-09-20, before the guard existed, and its leaderboard row still claims
to be `SacAgent/7616_step_37296` when it is really the ridge baseline.

Until 2026-09-22 the Models tab could not queue this job at all, because
`submitEvalJobs` sent `m.location` unconditionally — so every attempt from
there hit the guard, including on the FlyPyPolicy row itself. It now drops the
location for both from-scratch baselines (`FlyPyPolicy`, `RandomPyPolicy`) and
sends it as before for everything else.

### The course must be `donut_no_hint`, not `fly_donut`

`fly_donut` is the obvious-looking choice and it is wrong. `FlyPyPolicy` does
its own encoding: it reads speed at index 0 and the 29 rays at 2..30, so it
needs the 31-D ray vector. On `fly_donut` the observation has already been
replaced by the 1,314-D trace, and `RayEncoder.cues()` raises

```
ValueError: expected the 31-D observation, got 1314
```

Even if the shapes happened to line up it would still be wrong, because the
course and the policy would each be stepping the same single brain once per
env step.

The division is: **`fly_donut` is for SAC** (the course steps the brain and the
trace becomes the observation), **`donut_no_hint` is for FlyPyPolicy** (the
policy steps the brain itself and the course knows nothing about it). Every
FlyPyPolicy eval that has ever succeeded ran on `donut_no_hint`.

Both dialogs now enforce this rather than letting you find out from a
traceback: the Models tab blocks submission, and the Jobs tab's hint line warns
as soon as the pairing goes wrong.

---

## What `fly_donut` switches off, and why

This trainer has three separate mechanisms for using expert demonstrations, and
**`fly_donut` disables all three**. Not as a tuning choice — none of them can
work at this observation width.

| Mechanism | What it normally does |
|---|---|
| BC pretrain | Fits the actor network to expert actions before SAC starts (`bc_pretrain_steps`, default 5000) |
| Demo prefill | Loads expert transitions into the replay buffer so SAC has something good to sample early (`demo_prefill_count`, `demo_min_keep`) |
| AWAC | Regularises the actor toward demo actions during training (`awac_lambda`) |

The reason none of them apply is that **the observation is not recordable**.
The trace depends on the brain's voltages, which depend on the entire episode so
far, so a stored scene cannot be replayed into the observation it produced. This
is a stronger statement than the camera courses' problem: `donut_camera` lacks a
dict TFRecord layout, which is solvable, whereas here there is no layout that
would fix it. That is why the guard is keyed on the course rather than on the
observation kind:

```python
COURSES_WITHOUT_DEMOS = frozenset({
    "donut_camera",
    "donut_camera_no_rays",
    "fly_donut",
})
```

It acts in two places. `do_job` refuses DEMO and BC_TRAINING_ONLY jobs outright
and fails them with a message. `main()` zeroes the knobs for TRAIN jobs:

```python
if _course_type in collect_training_data.COURSES_WITHOUT_DEMOS:
    bc_pretrain_steps_val = 0
    demo_prefill_count    = 0
    demo_min_keep         = 0
    demo_sample_ratio     = 0.0
    awac_lambda_val       = 0.0
```

Keying this on `COURSE_OBS_KIND` instead would let `fly_donut` load the 31-D
TFRecords against a 1314-D spec.

**A gotcha worth knowing.** Because `main()` zeroes these *after* the
experiment design has been applied, an experiment design's demo and AWAC
settings are silently inert on this course. The job training right now uses a
design named `AWAC + No-BC + reward_scale1 (no curriculum)` which sets
`awac_lambda=0.5`, `demo_prefill_count=100000` and `demo_min_keep=100000` —
none of which have any effect. The trainer says so in one line, and it is the
line to check before concluding anything about AWAC from a `fly_donut` run:

```
main: fly_donut TRAIN from scratch (skip expert TFRecords, BC pretrain, AWAC)
```

Settings unrelated to demos still apply normally — that same job's
`replay_buffer_capacity=300000` and `env_discount=1.0` are live.

One rough edge: `do_job` still resolves the `donut_no_hint` demo corpus for a
`fly_donut` job and `main()` still reads and concatenates it (143,941 expert
steps) before discarding it. Nothing is inserted — the log reads `cap=0` and
demo prefill takes 0.00 s — but the TFRecord load is wasted wall time at
startup.

---

## Episode boundaries

The brain is stateful in a way no other course's observation is, so where the
reset happens matters, and the first two obvious answers are both wrong.

`reset_after_episode()` runs from `reward_success` / `reward_failure`, which is
*before* the terminal observation is packed — so the last observation of every
episode would be a reading of an already-cleared brain. `do_reset_blocking()`
is worse: `_reset()` skips it whenever the course already triggered the Unity
reset at episode end, which is exactly the episodes that ended in failure, i.e.
most of them early in training.

The fix is `BaseCourse.on_episode_start()`, called at the top of
`RobotaxiEnv._reset()`, which always runs and runs at the right moment. It
resets the brain, resets the encoder's previous-frame memory, and advances the
seed by one episode — so runs are reproducible while the spiking noise still
varies between episodes. `FlyPyPolicy` does the same thing on `StepType.FIRST`.

---

## Watching it

The overlay exists because watching the brain is how you debug everything else.
The snapshot rides along on the step that was already being made — one extra
2 KB field rather than a second round trip — and is published on a background
thread so a slow or dead `ros-server` costs the driving loop nothing.

Two topics, deliberately split: `fly_brain_geometry` (static soma positions and
edges, sent once) and `fly_brain_activity` (per-neuron intensity as `uint8`, at
20 Hz). Both must also be listed as `RosSubscriber` entries in the ROS routing
table, which is static — an unlisted topic publishes fine and silently never
arrives in Unity.

### Controls

| Key / action | Effect |
|---|---|
| **Hover over a neuron** | Label it — cell type, side, role, what it does in the fly, and how many of that type are drawn and firing |
| **B** | Toggle the overlay |
| **Left-drag on the brain** | Turn it. The face you can see follows the cursor: drag right and it swings right, drag up and it tips up. Release and it eases back to the neutral pose in 0.15 s. |
| **`-`** / **`=`** | Shrink / grow the overlay, ×1.08 per press, as a multiple of *the track's height*. The track's gutter follows, so growing the brain narrows the track rather than covering it. |
| **`0`** | Back to ×1 — exactly as tall as the track (`overlaySizeScale`) |
| **`[`** / **`]`** | Squash / stretch the depth axis, ×1.25 per press |
| **`\`** | Return the depth to the configured default |
| **Arrow keys** | Turn the *neutral* pose itself — held, at 20°/s. This is the one that squares the overlay up with the camera. |
| **`/`** | Return the neutral pose to the configured default |
| **`;`** | Show / hide the on-screen key legend. **Hidden by default**; a one-line hint stays, naming the key back |
| **`'`** | Show / hide the colour key. **Hidden by default** |

Both reset keys go back to the component's **defaults**, not to identity. Those
defaults are tuned values rather than neutral ones, so resetting to `×1` and
`0°` would land on a pose nobody wants and mean re-climbing the ladder by hand.
The shipped defaults are `depthMultiplier = 3.81` and a neutral pose of
`-144.3°` yaw / `-9.1°` pitch, all three on `FlyBrainViz`.

The legend is drawn bottom-left whenever the overlay is up, because a build has
no inspector and these keys are otherwise undiscoverable. It shows the live
depth multiplier and yaw/pitch values, so it doubles as the readout while you
tune. `controlsFontSize` (24) sets its size, and the panel is laid out in
multiples of that, so raising it scales the box with the type instead of
clipping rows.

### The overlay camera

The brain is drawn through its own orthographic camera, on its own layer
(`FlyBrainOverlay`, layer 7), pinned to a fractional viewport rect in the left
of the window. That one rig fixes three separate problems.

**It can no longer reach a policy's observation.** `JetRacerCsiIntrinsics` sets
the CSI camera's `cullingMask` to `~0` — every layer — and that camera is what
`CsiFramePublisher` reads for `camera/front`. The overlay previously had no
layer of its own, so a debug visualisation could appear inside training input:
a silent data-corruption bug rather than a visible one. The CSI mask now
excludes the overlay layer explicitly.

**It can no longer drift off-screen.** A viewport rect is a fraction of the
window, so the overlay holds its place and its share of the screen at any
resolution, instead of being a world-space object whose framing depended on the
main camera's aspect.

**It is sized to the track.** The overlay takes its height and its centre line
from `OverheadCameraFit.TrackScreenHeight` / `TrackScreenCentreY`, so the model
is exactly as tall as the road with its top and bottom level with the track's.
The column's width then falls out of the model's own projected proportions at
that height, rather than being a separate number the model has to fit inside —
which is what keeps the shape undistorted while the height is pinned. The model
being matched is the *whole* published cloud, brain and neck connective and
nerve cord together, so the proportions are the anatomy's.

`overlayZoom` is applied to the viewport and to the camera alike. Padding only
one of them would leave the model a margin shorter than the track; applying
both cancels it, so the padding is breathing room inside the column and the
*model*, not its viewport, is what matches.

Orthographic size comes from the model's **projected extents** — what it covers
on screen, measured over the real neuron positions. The earlier version fitted a
bounding *sphere* of `_modelRadius`, which is wrong twice over. That radius is
set by the single most distant neuron along the long axis, so the camera zoomed
out far enough to fit that distance on every axis including the short one; and
it counts the depth axis, which points straight at an orthographic camera and
covers no screen at all, so stretching the depth with `]` made the model
*smaller*. Modelled at 1080p, depth ±1.5 gave 318px where the projected framing
gives 540px, and the projected framing does not move with depth at all.

Orthographic size is a *half-height*, so the half-width it implies is that times
the aspect; the code takes the larger of the two requirements. That only binds
when the column has hit `MaxGutter` and is too narrow to show the model at track
height, in which case it shrinks to fit rather than being cropped.

The width does feed back — a wider column leaves the track narrower, the fit
pulls back, and the band being matched gets shorter. It settles rather than
breathing, because each pass shrinks the correction: the track is wider than it
is tall, so a column sized from its height costs less width than it gained.
Modelled from gutters of 0.05 through 0.59 it lands on the same value within one
step.

The camera copies the scene overhead camera's **rotation** rather than picking
one. The neurons lie in the brain's local X-Z plane with depth on local Y, so
it only reads as a brain when viewed down that Y axis — which is exactly what
the overhead camera does, and why the overlay looked right as a world-space
object. A camera looking along +Z instead catches it 90° edge-on and it
collapses to a sliver. Copying the overhead rotation also keeps both views in
the same frame, so the brain's forward is the track's forward and the
−144.3° / −9.1° square-up keeps its meaning.

**The track makes room for it.** `FlyBrainViz` publishes the share of the
window's width it occupies as `TrackGutterFraction`, and `OverheadCameraFit`
narrows the overhead camera's viewport rect by that much, so the course is
rendered in the remaining space rather than underneath the overlay. Unity
derives a camera's aspect from its viewport rect, so the fit reframes for the
narrower view automatically. The value is read live, so pressing `B` to dismiss
the overlay hands the full window back to the track on the next frame. A
zero-culling-mask clear camera sits behind the gutter, since the overlay camera
clears depth only and the reserved column would otherwise keep whatever was
last in the backbuffer.

**It can no longer smear.** `viz.py` warns that depth outliers "project far off
to the side and smear the whole structure into a radial fan" under a
perspective camera, and the shipped `×3.81` was doing exactly that: at a 30 m
viewing distance the ±11.4 m depth spread makes near neurons project 2.2× larger
than far ones, which is the bright cloud and dim streak visible in screenshots.
An orthographic camera has no `w` divide, so depth changes what occludes what
and nothing else.

In `useOverlayCamera` mode the trainer's `FLY_VIZ_OFFSET` no longer applies —
the rig is parked far from the course and framed by its own camera, so there is
nothing for a world offset to mean. `FLY_VIZ_DISPLAY_SIZE` still sets the model
scale, but the camera now frames to it automatically.

The viewport also steps above the controls legend when it would collide. The
two don't scale together: the legend is IMGUI with its scale clamped at the
small end, while a viewport rect is a pure fraction, so below roughly 594 px of
window height the legend stops shrinking and a fixed `y` would overlap it. The
overlay reads the legend's *measured* height and raises itself.

### Framing the track

The overhead camera is perspective with a fixed 60° vertical FOV, so the world
height it shows is fixed and the width follows the window's aspect. On a wide
window the course sits in a band of dead black space; on a portrait one it does
not fit at all — at 1080×1920 a fixed 60 m height shows 39×69 m against a course
needing 60×34 m.

`OverheadCameraFit` pulls the camera back along its own view direction until
the road's bounding box fits both axes with a margin. It deliberately does
*not* set the rotation to anything of its own: the overhead camera is yawed in
the scene, and overwriting that with a straight-down Euler stands the course on
end. For the same reason the extents are measured along the camera's own right
and up vectors rather than world X and Z — at ±90° of yaw the course's world X
maps to the screen's *vertical*, so fitting world extents to screen axes would
fit the wrong one to the wrong axis. It measures
**road** renderers specifically rather than everything, because the ground plane
extends well past the track and fitting to it would frame mostly grass;
`TrackGenerator` names its tiles `Road` and the kit puts asphalt on the `Road`
layer, so generated and authored courses are handled the same way. It refits on
a timer and immediately on resize, since the curriculum can rebuild the track
underneath it.

This is display only. The policy's camera input comes from `JetRacerCsiCamera`,
which `CsiFramePublisher` renders to its own `RenderTexture` at a fixed
resolution, and which this never touches.

### Overlay scaling

IMGUI draws in raw pixels and has no equivalent of uGUI's `CanvasScaler`, and
every overlay in this project was authored at fixed pixel sizes. The result is
that the same panel swamps a small window and shrinks to illegibility on a
large one — the HUD, the curriculum buttons and the fly-brain panels all had
this.

`OverlayUi` fixes it the way `CanvasScaler` does: a reference height of 1080,
a scale derived from the real window height, pushed through `GUI.matrix`.
Overlays then lay out in *logical* pixels — `OverlayUi.LogicalWidth` /
`LogicalHeight` in place of `Screen.width` / `Screen.height` — and every size
in their existing code keeps working unchanged, now meaning "at 1080p".

| Window | Scale | 24 pt renders at |
|---|---|---|
| 1280×720 | 0.67 | 16 px |
| 1920×1080 | 1.00 | 24 px |
| 2560×1440 | 1.33 | 32 px |
| 3840×2160 | 2.00 | 48 px |

Height, not width: vertical space is what panels compete for, and scaling off
width would inflate them on an ultrawide window that has no more vertical room
than a 16:9 one. The scale is clamped to 0.55–3.0 so a very small window keeps
legible text at the cost of a larger share of the screen.

The ROS connection panel (`HUDPanel`) scales too, but **duplicates** the three
constants instead of calling `OverlayUi`. That is not an oversight: it lives in
the vendored `ros-tcp-connector` package behind
`Unity.Robotics.ROSTCPConnector.asmdef`, and an asmdef assembly cannot depend on
the predefined `Assembly-CSharp` where `OverlayUi` lives. If the reference
height or the clamps ever change, they have to be changed in both places, or
that panel drifts out of size with everything around it.

The curriculum panel's styles set `fontSize = 0`, meaning "use the skin font's
own size" — which is exactly what `HUDPanel` does, since it sets no size at all.
Inheriting rather than naming a number is what keeps the two matched: a
hard-coded 12 or 14 would agree only until the skin changed, with nothing to
notice when it drifted.

Three gotchas the implementation has to respect. `GUIUtility.RotateAroundPivot`
left-multiplies onto the current `GUI.matrix`, which means it treats its pivot
as a *screen-space* point. Under an overlay scale a rect's centre in logical
pixels is not its centre in screen pixels, so the HUD's steering wheel orbited
that offset instead of spinning in place — by 450 px at 0.67× and 1827 px at
2×, which is why it appeared to fly around the screen at random. It was
correct at exactly 1.0×, which is why the bug arrived with the scaling. The
wheel now composes its rotation onto the *right* of the scale, rotating in the
same space the rect is expressed in.

`GUI.matrix` is global IMGUI
state, so a scale left set leaks into every overlay that draws afterwards —
`CurriculumStageButtons` returns from the middle of its draw when the panel is
collapsed and so restores in a `finally`. And `GUI.matrix` does not apply to
coordinates sourced from outside IMGUI, so the hover label converts the mouse
position with `OverlayUi.ToLogical` before mixing it with layout, while the
hover *pick* deliberately stays in real pixels because it is comparing against
the camera viewport.

### The colour key

A second panel, bottom-right, says what the colours mean so someone watching
can read the overlay without being talked through it. It toggles separately
from the controls (`'` vs `;`), because presenting this generally wants the
colours up and the key bindings down.

| Swatch | Role | What it is, and which cue drives it |
|---|---|---|
| Cyan | **loom** | **LC4 + LPLC2**, the detectors the looming cues are injected into, side-specifically. **LPLC1** is drawn here too — see the wrinkle below |
| Violet | **chase** | **LC10a**, the detectors the chase cues are injected into |
| Grey | **relay** | Interneurons on the sensory → descending path, picked by two-hop weight |
| Amber | **command** | The named command neurons — DNp01, DNa02, MDN, pIP10, … |
| Red | **descending** | The `descending_neuron` superclass: the 1314 values the policy actually reads |
| Slate → orange | **context** | Silhouette only, not part of the circuit. Unlike the others it crosses *colour* with activity rather than dimming, which is what makes active neuropils stand out. |

Two things the swatches can't show, so the panel spells them out: brightness is
firing rate, and the left/right split is spatial rather than coloured even
though it's what carries the steering signal.

Below the swatches the panel names the cells each injected cue reaches, with
counts resolved live from the connectome:

```
CUES INJECTED
loom_L    LC4+LPLC2, left   ·  165 cells  ·  23 firing
loom_R    LC4+LPLC2, right  ·  146 cells  ·  19 firing
chase_L   LC10a, left       ·  135 cells  ·  41 firing
chase_R   LC10a, right      ·  140 cells  ·  12 firing
```

This is the cue table from earlier in this document, rendered on screen and
kept live. The cell counts are computed from the published type and side
arrays by matching the encoder's `POPULATION_CELLS`, so they agree with the
connectome rather than being transcribed — 165 / 146 / 135 / 140 against a
running `fly-brain` service. A cue row matches on type **and** side, which is
what makes `loom_L` and `loom_R` different numbers at all. Against a trainer
that publishes no type table the row still names the cells and simply omits
the counts, rather than showing a confident zero.

Each cue row is swatched in the colour its own cells are drawn in, so a row and
the dots it counts can be matched by eye.

The swatches are drawn from the same `sensoryColor` / `chaseColor` /
`commandColor` / … fields `RoleRamp` switches on, so retinting in the inspector
retints the key entry too. The role names come from `display_subset.build` and
the cue mapping from the encoder's `POPULATION_CELLS` — both Python-side, so if
either changes, this panel's text is what goes stale. One wrinkle it glosses:
`LPLC1` is in `SENSORY_TYPES` and so is drawn cyan with the loom cells, but the
encoder does not drive it.

#### Why loom and chase are two rows

Both are role 1. The connectome makes no colour distinction between them and
neither did this overlay at first — every detector was one cyan, which made the
comparison you actually want to watch, *loom firing versus chase firing*, the
one thing the overlay couldn't show. `chaseColor` splits role 1 by cue,
resolved through the same `CueOf` lookup the counts use, so it matches on type
**and** side exactly the way the injection does. Against a trainer that
publishes no type table there is nothing to split on, and the overlay degrades
to the old single cyan rather than guessing.

The specific violet, `rgb(0.60, 0.15, 1.00)`, is chosen rather than picked.
Scored in CIEDE2000 against every other colour in this palette under normal
vision plus simulated protanopia, deuteranopia and tritanopia, it stays 27
clear of its nearest neighbour in the worst case and 28 clear of the loom cyan
— comfortably above the ~10 where two colours read as *clearly* different.

The reason it is a blue-leaning violet and not the brighter magenta you might
reach for is worth knowing before someone "fixes" it: the collision risk in
this palette is not the cyan, it is `descendingColor` red. Pushing the purple
toward pink collapses the two for a tritanope — `rgb(1.00, 0.30, 0.95)` scores
**0.6** against that red, where anything under 2 is indistinguishable. The red
channel is the margin. Raise it and this stops working for some viewers while
still looking fine to you.

### Hovering a neuron

Point at any neuron and a label appears next to the cursor with the same
information the cue table above carries, resolved for whatever is under the
pointer:

```
LC4 · left
lobula columnar neuron, type 4
looming · speed of an expanding edge
sensory  ·  drives DNp01 +25 spikes/s here
cue loom_L  ·  LC4 + LPLC2, left
71 cells drawn  ·  23 firing now
```

Read top to bottom that is: **what it is**, **what it detects**, what it does in
*this* connectome, which cue population it belongs to, and how many of it there
are. Every line is optional, so an unnamed relay or a context cell still gets a
short honest label rather than blank rows.

The separation between lines two and three is deliberate. Line two is a finding
from the *Drosophila* literature; line three is what this repo measured. Mixing
them, as an earlier version did, makes it impossible to tell which claims you
could check by reading a paper and which you could check by re-running step 2.

The population line names the group rather than just the cue, because
`LC4 + LPLC2, left` is what the encoder actually injects into — so a hovered LC4
says what it is bundled with. It resolves through the same `CueOf` lookup the
colour key uses, matching on type *and* side exactly the way the injection does,
rather than a second table that could drift out of agreement with it.

The counts are live in both senses. "Cells drawn" is resolved from the
connectome at runtime rather than hardcoded — it is the same quantity as the cue
table's *live cell count* column, and the LC4-left figure should read 71 against
step 1's `LC4 (L71/R55)`. "Firing now" is recounted four times a second from the
current activity frame, thresholded at `hoverFiringThreshold` (0.25) of full
scale. That threshold reads the **raw** activity byte, before `intensityGamma`,
so changing the display curve cannot quietly move the tally.

The headline is drawn in the colour that neuron is drawn in, tying the label to
the thing it describes.

**Only types the repo has actually measured get a function line.** LC4, LPLC2
and LC10a carry step 2's measured pathways; DNp01 and DNa02 are their measured
targets; DNg100 is described as flat because step 2's strength sweep measured it
at +0.0–0.1 under every stimulus; LPLC1 is labelled as undriven because it is in
`SENSORY_TYPES` but absent from the encoder's `POPULATION_CELLS`. Everything
else — MDN, pIP10, DNp10, DNg13, most relay interneurons, most of the 1,314
descending neurons, and the whole context sample — falls through to its **role**
description. Plausible-sounding functions for those would make the label
untrustworthy exactly where nobody can check it.

**The tuning lines, and what backs each one.** `LC` is lobula columnar, `LPLC`
is lobula plate / lobula columnar (dendrites in both neuropils), and `DN` is a
descending neuron with its Namiki et al. 2018 group letter and number.

| Type | Shown as | Source |
|---|---|---|
| LC4 | looming · speed of an expanding edge | Encodes the angular *velocity* of a looming edge; with LPLC2, one of two functionally distinct giant-fibre inputs (von Reyn et al. 2017; Ache et al. 2019) |
| LPLC2 | looming · outward motion = collision course | Ultra-selective for outward radial motion by opponency — the optic-flow signature of a direct collision course — and encodes angular *size* (Klapoetke et al. 2017) |
| LC10a | chasing · tracks a small moving object | Small-object tracking, required for visually guided courtship pursuit, steering via DNa02 (Ribeiro et al. 2018; Sten et al. 2021) |
| DNp01 | looming target · giant fibre, escape take-off | The giant fibre, one cell per side, driving short-mode escape |
| DNa02 | chasing target · turns toward the target | Drives ipsilateral turning during walking |
| LPLC1 | looming family · no cue injected here | In `SENSORY_TYPES`, absent from `POPULATION_CELLS` — a repo fact, not a tuning claim |
| DNg100 | *(no tuning line)* | — |

DNg100 is the one to note. The plan document calls it "forward walking", and the
label used to repeat that, but the literature does not establish it — so the
hover now shows only what step 2 measured, that it is flat under every cue and
strength tested. The pattern is the same one the function lines follow: where
the repo cannot source a claim, it says less rather than guessing.

This needs cell types on the wire, which the geometry payload now carries as a
table of distinct names plus a `uint16` index per neuron. The index is 16-bit
because a real display subset has around 400 distinct types, comfortably past a
byte: the context sample is drawn from the whole brain and brings a long tail
with it. That costs about 54 KB on a payload sent once per job. `side` was
already being published and simply wasn't parsed; the label needs it, since
"LC4" alone doesn't say which eye. Against a trainer that predates the type
table the label still works and falls back to role-level text.

Picking projects every neuron into screen space rather than raycasting, since
the overlay is billboarded quads with no colliders — the same reason the drag's
hit test is a screen-space circle. Three things keep that affordable: the full
pass only runs when the cursor is already inside the overlay's bounding circle,
it is skipped entirely while dragging, and the firing recount is on a timer
rather than per frame. Circuit neurons beat context ones within the pick radius
even when a context point is nearer, or 16,000 points of silhouette would
swallow every hover.

Both bottom corners are now in use. If the two panels would collide on a narrow
screen, the colour key lifts itself above the controls rather than either panel
shrinking; `DrawControlsLegend` reports the size it actually drew and the colour
key is handed that, so the two never have to agree on a layout formula.

Both panels **measure their own columns** with `GUIStyle.CalcSize` rather than
assuming an em-width per character. The first cut guessed, and got it wrong in
the way that is easy to miss: `GUI.skin.label` word-wraps by default, so a name
slightly wider than its column took a second line, the row had height for one,
and the glyphs were cut through the middle — "descending" lost its descender and
the rows looked squashed. All three styles now have `wordWrap = false`, which
both fixes the cropping and makes `CalcSize` meaningful (with wrapping on it
reports a tall box rather than a line width). Row height is measured off a
string with an ascender and a descender for the same reason.

### Squaring it up with the camera

The arrows and the drag do different jobs and it matters which you reach for.
The **drag springs back**, so it is a look-around and cannot hold a correction.
The **arrows move the neutral pose**, and the drag springs back *to* that — so
if the overlay sits a few degrees off flush to the top-down camera, the arrows
are the fix and the correction survives releasing the mouse.

This can't be done from the Python side. `FLY_VIZ_ROTATE` turns the picture
*within* the screen plane, which cannot correct a tilt, and `FLY_VIZ_AXES` only
permutes and flips whole axes. The tilt comes from the rotation of the overlay's
parent transform, so squaring up is a Unity-side rotation or nothing.

Each arrow release logs the pair:

```
[FlyBrainViz] orientation yaw 7.25 pitch -1.5 - to keep it, set
baseYawDegrees / basePitchDegrees to these before the next build
```

Those are `public` fields on `FlyBrainViz`, so a value found by eye becomes the
default in the next build. Both rotations are about the *camera's* axes, with
the same sign convention as the drag: positive yaw turns the brain right,
positive pitch tips its near face up.

The depth controls exist for a related reason: `FLY_VIZ_DEPTH_SCALE` defaults to
0.12, squashing the camera-facing axis to a slab so the top-down perspective
camera does not smear the structure into a radial fan. Rotating a slab shows
you a sheet, so the depth keys are what make the rotation worth having.

**The depth keys are not an approximation of the env var — they are the same
arithmetic.** `viz.py` clips to ±1 *before* multiplying:

```python
pos[:, 1] = np.clip(pos[:, 1], -1.0, 1.0) * self.cfg["depth_scale"]
```

The clip bounds are literal and do not depend on `depth_scale`, so the scale is
a pure final multiplier and applying another one in Unity lands on the same
positions `FLY_VIZ_DEPTH_SCALE × depthMultiplier` would have produced. Nothing
is lost, and you can find a value by eye on a running job and then make it
permanent.

The legend and the log both report the absolute value, so there is nothing to
work out by hand:

```
[FlyBrainViz] depth x3.81 - to keep it, set FLY_VIZ_DEPTH_SCALE=0.4572
(it is 0.12 now), or depthMultiplier=3.81 on FlyBrainViz
```

Two places it can live, and they are not equivalent. `FLY_VIZ_DEPTH_SCALE` on
the trainer applies to every client and every future job and needs no rebuild;
`depthMultiplier` on `FlyBrainViz` is Unity-side and ships with the binary.
Prefer the env var.

The current default takes the Unity route (`depthMultiplier = 3.81`). The
equivalent is `FLY_VIZ_DEPTH_SCALE=0.4572` with the multiplier back at 1 —
identical positions, and changeable without a rebuild.

Reporting the absolute value needs `depthScale` in the geometry payload, which
the trainer sends as **provenance only** — the flattening is already baked into
`pos`, and Unity must never apply it a second time. A trainer that predates the
field sends nothing, `JsonUtility` zero-fills it, and the overlay falls back to
showing the bare multiplier.

Doing it in Unity is also the only way to change it on a **running** job.
`get_config()` reads the environment in `FlyBrainViz.__init__`, once per job,
and `_geometry_json` is packed once and cached for that instance's lifetime, so
the depth scale is already baked into the positions on the wire. Changing the
env var needs a trainer restart, which ends the job you were watching.

Three implementation notes, all consequences of how the overlay is drawn. Both
rotations are about the *camera's* right and up axes rather than world ones,
because the sim camera looks straight down and a rotation about Unity's y would
spin the brain in the screen plane — the same reason `FLY_VIZ_SPIN` ships at 0.
The drag's hit test is a screen-space circle around the overlay's centre, sized
from `displaySize`, rather than a raycast: the brain is billboarded quads and
lines with no collider to hit. And the neuron mesh's bounds scale with the depth
multiplier, since they are set from the extent rather than recalculated, and a
stretched cloud would otherwise be frustum-culled as a whole.

### Sizing the overlay

`overlaySizeScale` is a multiple of **the track's height**, not a size in the
abstract: at the default of 1 the model is exactly as tall as the road. `-` and
`=` step it by ×1.08 and `0` returns to the default, so the keys are a
deliberate departure from the thing the overlay is meant to line up with.

The framing is measured over the real neuron positions rather than a bounding
sphere or an axis-aligned box, because the cloud is neither and both
approximations round in the direction of drawing it too small. The measurement
is a full pass over every neuron, so it runs only when the answer can have
changed — a new pose, or new positions from geometry or a `[` / `]` press. At
the shipped spin of 0 that means it is idle.

The **drag rotation is excluded** from it. Folding the drag in would rescale the
model while it is being turned, so it would swell and shrink under the cursor
instead of just rotating; the settled pose is what gets framed, and the extra
reach of a turned one is covered by `overlayZoom`'s padding.

The camera is also offset onto the centre of what the model covers, using
min/max rather than max-absolute, so a cloud whose centre of mass sits off its
origin is centred on what it covers instead of being framed around empty space
on the opposite side.

**The legends deliberately do not push the overlay around.** An earlier version
kept the overlay clear of the controls panel, which sounds tidier but made that
panel's height the thing that decided how big the brain could get, capping the
size outright and leaving `-` and `=` with nothing to do. The panels are
translucent and draw on top; letting them overlap costs a corner of the overlay
and buys back the entire column.

Both panels are **hidden by default** for the same reason — they sit over the
brain, which is the thing being looked at. The one-line hint in the corner
(`; fly-brain keys   ' colours`) is what keeps them discoverable in a build,
which has no inspector.

The overlay may never claim more than `FlyBrainViz.MaxGutter` (0.6) of the
window's width, and `OverheadCameraFit` clamps the track's gutter to the same
constant — if the two disagreed, the overlay would claim a strip the track was
still drawing into.

Hiding both panels leaves a one-line hint naming the keys that bring them back,
because otherwise the only route back from `;` or `'` is already knowing the
answer. For the same reason the colour-key row reads "show colour key" when the
key is hidden rather than asserting "hide".

Keys are checked against the rest of the scene: `B` `[` `]` `\` `/` `;` `-` `=`
`0` and the arrows are the overlay's, while `H` (hud), `T` (rollout fan), `C` (curriculum
stages), `P` (car camera) and `F` (CSI dump) belong to other scripts. The
legend's last line lists those too, so one panel answers "what are the keys" —
which also means it is what goes stale if one of them is renamed.

The curriculum panel starts **collapsed**, showing only `(C to show stages)`.
Its buttons are a manual override for pushing track geometry around without a
trainer attached; they are not something to watch during a run, and expanded
they put five buttons in the corner of every screenshot.

To check the whole cross-container path without touching a running job:

```
docker compose exec -w /python_ws/src sim-controller python -m fly_brain.client
```

Beware that this smoke test calls `Reset` and `Step`. Run it while a job is
live and you will corrupt that job's brain state (see below). `Info` and
`Cells` are read-only and safe.

In the trainer log, the lines that confirm the pipeline is wired correctly are:

```
fly_donut: trace_len=1314 obs_max=5.52 substeps=5 device=cuda
[fly_donut] obs vector(1314,) (of scene(31,))
```

The second is the one to look for: a 1314-wide observation derived from a
31-wide scene is the whole integration in one line.

---

## Constraints that bite

**The service holds exactly one brain.** `FlyBrainService.__init__` creates one
`self.brain` behind one lock, and both `Reset` and `Step` mutate it. Two
environments sharing it do not *race* — the lock prevents that — they
**interleave**, and each reads a trace shaped by the other's rays, silently and
with no error. So `fly_donut` runs at `--num-envs 1`.

This is not theoretical. A determinism check run from a shell while a
`FlyPyPolicy` eval happened to be mid-job reported `max|trace − trace2| = 1.6`
for the same seed, against `0.0` when the brain is idle. Nothing errored on
either side; the first step after a `Reset` still matched exactly and the
divergence grew with step count. A slow drift with no signal, which silently
contaminated a leaderboard row.

One lane is exactly enough at `--num-envs 1` and not one more: `main()` only
builds a separate eval env when `num_envs > 1`, so below that the collect loop
and the periodic evals share one course instance and one brain client and never
overlap. Batching is the fix and the groundwork exists (`FlyBrain(batch=N)`,
`Trace(aggregate="batch")`), but measure first — batching buys about 1.6× at
batch 8, not 8×, because the synaptic input is a sparse matmul over 166,700
neurons that scales close to linearly in the batch dimension.

**The brain is non-deterministic.** The same rays give a different trace twice.
Seeding is per episode, which is why comparisons are reproducible at all. When
comparing against SAC, remember greedy SAC eval is a deterministic `tanh(μ)`
while the fly readout is a point estimate with spiking noise underneath, so the
two variances are not comparable by default.

**Replay stores history-dependent observations.** The trace depends on the
brain's voltages, which depend on the whole episode so far, so off-policy replay
learns from features it cannot exactly reconstruct. This is workable — the trace
*is* the observation, so there is no mismatch between what was stored and what
the critic sees — but if training goes unstable, suspect this first.

**GPU contention.** TensorFlow grabs most of the card by default and the brain
wants ~210 MB plus working space. `TF_FORCE_GPU_ALLOW_GROWTH=true` on the
trainer, or the brain service fails to allocate.

**Throughput is not the brain's fault.** Five substeps cost about 13.4 ms per
control step. That fits a 10 Hz sim-time budget easily, and although
`Time.timeScale` of 3–5 pushes the real control rate to 30–50 Hz, the measured
`fly_donut` run sits at 0.23 s/iteration with 0.09 s in collect — so the Unity
round trip, not the brain, sets the pace at one env.

---

## Where things live

| File | What it is |
|---|---|
| `docker/fly_brain/fly_brain_server.py` | The gRPC service. Owns the one `FlyBrain` and the descending trace. |
| `protos/fly_brain/proto/fly_brain.proto` | The wire contract. |
| `rl_agent/fly_brain/client.py` | Trainer-side client. `SUBSTEPS = 5`, `EYE_DRIVE = 0.45`. Also the smoke test. |
| `rl_agent/fly_brain/encoder.py` | `RayEncoder` (the 4-cue encoder in use) and `RetinotopicEncoder` (the measured negative result). |
| `rl_agent/fly_brain/step5_readout.py` | Fits the ridge readout and reports held-out R² against its controls. |
| `rl_agent/fly_brain/policy.py` | `FlyPyPolicy` — the BC route, as a tf-agents `PyPolicy`. |
| `rl_agent/environments/courses/fly_donut_course.py` | The RL route. The trace becomes the observation here. |
| `rl_agent/fly_brain/viz.py` | Publishes geometry and activity to ROS for the overlay. |
| `unity/Assets/Scripts/FlyBrainViz.cs` | Renders it. |
| `rl_agent/environments/robotaxi_env.py` | `_pack_observation` / `_as_obs_time_step` — where `policy_vector` is applied. |
| `rl_agent/collect_training_data.py` | `COURSES_WITHOUT_DEMOS`, `COURSE_OBS_KIND`. |

---

## Where this has got to

Measured so far: the ridge readout drives at roughly 7× a random policy and
about a tenth of SAC. SAC on the trace is training now — job
`6ab0bc6ac2736b990584404a`, 100,000 iterations on `fly_donut`.

Two things are still open, and the second is the one that determines whether
any of this means anything.

**Compare honestly.** `fly_donut` against `donut_no_hint` at an equal step
budget, on `eval/goals_per_episode_this_eval` rather than `avg_return` —
comparing returns across jobs with different reward designs is meaningless.

**Run the scrambled-wiring control.** Rebuild the brain with degree-preserving
shuffled weights, keeping neuron count, connection count and encoder identical,
and train again at the same budget. Without it you cannot claim the fly
connectome did anything, because a 166,700-unit *random* recurrent network is a
perfectly good reservoir on its own. fly.ai's own equivalent control came out
ambiguous. The actual result of this project is four numbers at one budget:
`donut_no_hint` SAC, the ridge readout, the real connectome under SAC, and the
scrambled connectome under SAC.

A closing caveat that the upstream project states plainly and that should
travel with any result from here: this is a demo, not an emulation. Point
neurons, one global parameter set, no dendrites, no neuromodulators, no
plasticity, transmitter sign from a rough rule, and nothing validated against
recordings from real flies. Expect an interesting result, not a good driver.
