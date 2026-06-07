# Design: Trajectory Rollout Visualization in the Unity Client

**Status:** Implemented (Phase 1, open-loop). Sections 3–9 below are the original
design; the **"As-built" section** immediately below is the authoritative
description of what actually shipped and where it diverged from the draft.
**Scope:** Render candidate future trajectories in the Unity client, sampled
from the SAC actor policy, so an operator can *see* what the policy "intends"
and how much spread/uncertainty there is in its action choices.

---

## As-built implementation (current)

End-to-end pipeline as actually shipped. Key files: `rl_agent/rollout_viz.py`,
`unity/Assets/Scripts/TrajectoryRolloutViz.cs`,
`docker/ros_server/ROS/src/niryo_moveit/scripts/unity_node.py`.

### Sampling — background thread, not in-loop (diverges from §5)

A **dedicated 20 Hz daemon thread** (`RolloutViz`, a module-level singleton in
`rollout_viz.py`) does the sampling, **decoupled from the training loop**. The
train/eval loops merely feed it the latest `(policy, observation)` each
iteration via `update_context()` (the old `maybe_publish()` is now a thin shim
for that). This was necessary because the loop runs at only ~3 Hz and **stalls
for 20–30 s during periodic eval/checkpoint cycles** — in-loop publishing made
the fan update slowly and vanish during eval. The thread re-samples + publishes
at a steady ~20 Hz regardless of phase (verified ~19.9 Hz, max gap ~70 ms).

Each tick tiles the **current observation** into a batch of `K*H` and samples
the policy once → `K` sequences of `H` actions (`accel`=action[0],
`steer`=action[1]). Still strictly **open-loop**: all `H` steps are i.i.d. draws
from the *same* current-observation distribution.

### Per-trajectory probability weights (new vs the draft)

Beyond the spread, each trajectory gets a **relative probability**: sum the
per-step `log_prob` over the horizon (open-loop joint likelihood), then
**softmax across the K sequences** → `weights[]`.

Getting `log_prob` required unwrapping the policy: the trainer hands a
`PyTFEagerPolicy` (numpy wrapper) which has no `distribution()` /
`_actor_network`; we unwrap to the underlying TF policy (`._policy`) and call
`distribution(ts_tf)` with TF tensors. The `'loc'` serialization bug only
affects the **reloaded SavedModel** eval policy — on it (and any failure) we
fall back to **uniform weights**. So weighting is meaningful during **training
(stochastic collect policy)** and degenerate during **eval (greedy)**.

### How the probabilities are computed (the math)

**The actor head is a tanh-squashed diagonal Gaussian — NOT a softmax.** SAC's
policy is continuous over the 2-D action `(accel, steer)`:

1. The actor network outputs a per-dimension mean and (log) std, defining a
   diagonal Gaussian over a *pre-squash* latent `u ~ N(μ(o), σ(o))`.
2. The action is the **tanh squash** `a = tanh(u)` (bounded to (−1, 1), then
   scaled to the action range), which is what keeps `accel`/`steer` in range.

The density of a tanh-squashed Gaussian needs the change-of-variables
(Jacobian) correction for the squash. For one action vector at observation `o`:

```
log π(a | o) = Σ_i [ log N(u_i; μ_i(o), σ_i(o))  −  log(1 − tanh²(u_i)) ]
               \_______ base Gaussian _______/   \__ tanh Jacobian __/
   with  u = atanh(a)
```

We don't implement this by hand — tf-agents' `dist.log_prob(a)` returns the
fully corrected log-density (summed over the action dims). We sample and score
in one shot: `a = dist.sample()`, `logp = dist.log_prob(a)`.

**Per-trajectory likelihood (open-loop).** A trajectory `k` is `H` actions
`a_{k,0..H-1}`, all drawn from the *same* current-observation distribution
`π(·|o₀)`. Its joint log-likelihood is the sum over the horizon:

```
L_k = Σ_{h=0}^{H-1} log π(a_{k,h} | o₀)
```

Because every step is scored against the same `π(·|o₀)`, `L_k` is largest for
sequences whose actions sit near the distribution's mode (≈ the greedy action
repeated) and smaller for atypical/outlier sequences. It is **not** a
closed-loop path probability (that would re-query `π` at each predicted pose).

**Where softmax actually enters — across trajectories, for the viz.** To turn
the `K` log-likelihoods into comparable, sum-to-one weights we softmax over the
`K` sequences (max-subtracted for stability):

```
w_k = exp(L_k − max_j L_j) / Σ_j exp(L_j − max_j L_j)
```

So the only softmax in the system is this **visualization weighting over the K
candidate paths** — the policy head itself is the continuous tanh-normal above.
Unity then min-max normalizes `w_k` across the K paths and maps it to opacity +
width (most likely = most opaque/widest).

### Publish path — own gRPC channel (diverges from §4 RobotApi route)

The thread owns a **synchronous gRPC channel straight to `ros-server-0:50051`**
(actor-0's server) and publishes a `std_msgs/String` JSON blob on
`policy_rollouts`. Using its own channel (not the env-subprocess `RobotApi`)
avoids racing the training thread's `env.step()` on the same pipes.

### Payload — FLAT arrays (diverges from §4 nested schema)

`JsonUtility` can't parse nested lists, so the payload uses flat, k-major
parallel arrays plus the style knobs:

```jsonc
{
  "stamp": 1733300000.1, "step": 26771, "dt": 0.1, "horizon": 25, "k": 8,
  "accel": [ /* length k*H, accel[k*H + h] */ ],
  "steer": [ /* length k*H */ ],
  "weights": [ /* length k, softmax over joint log-prob */ ],
  // render-style knobs (so Unity is tunable WITHOUT a rebuild):
  "lineWidth": 1.5, "probFalloff": 1.0,
  "minAlpha": 0.30, "maxAlpha": 0.95, "minWidth": 0.40, "maxWidth": 1.6
}
```

### ros-server bridge fix (not anticipated in the draft)

This connector uses a **static routing table**, not dynamic Unity subscriptions
(the endpoint logged zero `RegisterSubscriber` events). A topic only reaches
Unity if listed in `unity_node.py`. We added
`'policy_rollouts': RosSubscriber('policy_rollouts', String, tcp_server)` and
rebuilt the `docker_ros-server:thin` image. The message stays `std_msgs/String`,
so adding JSON fields needs no further ros-server change.

### Rendering — 20 Hz redraw from the car FRONT (diverges from §6 on-message)

`TrajectoryRolloutViz` (auto-attached by `SimController`) caches the latest
payload and **redraws at 20 Hz from the car's *current* front-axle pose** (not
just on message arrival), so the fan stays glued to the moving car. For each
sequence it forward-simulates the bicycle model (§7) from the car's live
speed/heading. Per-line style is driven by the probability weight (min-max
normalized across K):

- **opacity** decays **exponentially** with lower probability, floored at
  `minAlpha`; **width** scales similarly; single **blue** hue.
- Style params come from the payload when present, else Unity inspector
  defaults — so opacity/width are tunable from Python env vars, no rebuild.

### Toggles & related viz (new)

Because Active Input Handling is **Input System only**, toggles use
`Keyboard.current` (legacy `Input.GetKeyDown` throws). Keys: **T** = rollout
fan, **R** = in-build perception ray lines (real `LineRenderer`s replacing
build-invisible `Debug.DrawRay`), **H** = the steering/speed/force HUD.

### Env vars (current)

`ROLLOUT_VIZ_ENABLED` (gate), `ROLLOUT_VIZ_K` (8), `ROLLOUT_VIZ_HORIZON` (25),
`ROLLOUT_VIZ_DT` (0.1), `ROLLOUT_VIZ_HZ` (20), `ROLLOUT_VIZ_ACTOR_INDEX` (0),
`ROLLOUT_VIZ_ROS_ADDR` (`ros-server-0:50051`), and the style knobs
`ROLLOUT_VIZ_LINE_WIDTH`, `ROLLOUT_VIZ_PROB_FALLOFF`, `ROLLOUT_VIZ_MIN_ALPHA`,
`ROLLOUT_VIZ_MAX_ALPHA`, `ROLLOUT_VIZ_MIN_WIDTH`, `ROLLOUT_VIZ_MAX_WIDTH`.
(`ROLLOUT_VIZ_EVERY_N_STEPS` is obsolete — the background thread is rate-based.)

---

## 1. Goal

While a job is training (or being evaluated), draw **K candidate trajectories**
in the live Unity client originating at the car's current pose. Each trajectory
is produced by sampling the actor policy's action head and forward-simulating
the car's motion. The visualization communicates, at a glance:

- the policy's **immediate intent** (where it's about to go), and
- the **spread** of action sequences consistent with the current policy
  distribution (a proxy for behavioral uncertainty / multimodality).

### Chosen Phase-1 configuration (confirmed)

| Decision | Choice |
|---|---|
| Rollout fidelity | **Open-loop** (no policy re-query along the path) |
| What each trajectory is | **An H-step action sequence**, sampled from the *current* policy distribution |
| Coloring | **Uniform** (just show the spread) |
| Deliverable | This design doc first; implement after review |

> Terminology note: the SAC actor head is a **continuous tanh-squashed
> Gaussian** over `(accel, steer)`, *not* a softmax. "Sampling the head" means
> drawing continuous 2-D action vectors from that distribution.

---

## 2. The core constraint that shapes the design

A faithful rollout needs, at every step, the **observation** at the predicted
future state in order to re-query the policy. The observation here is built
primarily from **raycasts against the track**, which only the Unity client can
compute. Python has no model of the future raycasts.

**Open-loop sidesteps this entirely:** we compute the policy's action
distribution **once**, at the current observation `o₀`, then sample action
*sequences* from that single distribution and forward-simulate them with a
kinematic model. We never need a future observation because we never re-query
the policy mid-rollout.

This makes Phase 1 a **heuristic spread visualization**, not a true
state-conditioned rollout. A true closed-loop rollout (re-querying the policy at
each predicted pose, with Unity supplying the hypothetical-pose raycasts) is
explicitly **Phase 2** (Section 8).

### Why this runs in the trainer process

The actor's `.distribution()` works on the **live, in-memory** policy but
**not** on a loaded `SavedModel` (the `policy_saver` "Could not serialize
policy.distribution()" warning — the tanh-normal can't be reconstructed from the
SavedModel). Therefore the sampling+publish hook must live in the **training /
eval process** that holds the live policy, not in a separate SavedModel-loading
service.

---

## 3. Architecture & data flow

```
 ┌─────────────────────────── trainer process (Python) ───────────────────────────┐
 │  collect/eval actor loop                                                        │
 │    every N env steps:                                                           │
 │      dist = policy.distribution(time_step)        # live tanh-normal            │
 │      for k in range(K):                                                         │
 │          seq_k = [dist.sample() for _ in range(H)]   # H iid 2-D samples        │
 │      publish /policy_rollouts  { chosen, samples_seq[K][H], horizon, dt, stamp }│
 └───────────────────────────────────┬─────────────────────────────────────────────┘
                                      │  ROS topic (Python -> Unity), JSON payload
                                      ▼
 ┌─────────────────────────── Unity client (C#) ──────────────────────────────────┐
 │  TrajectoryRolloutViz (MonoBehaviour)                                           │
 │    on message:                                                                  │
 │      pose0, v0 = car current transform + speed                                  │
 │      for k in range(K):                                                         │
 │          waypoints = forward_sim_bicycle(pose0, v0, samples_seq[k], dt)         │
 │          lineRenderer[k].SetPositions(waypoints)                                │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

**Direction:** Python **publishes**, Unity **subscribes** — the same direction
already used by `sim_command` (Python -> `SimController.onCommand`). So the
plumbing pattern already exists in the codebase.

---

## 4. Component 1 — ROS topic + message

**Topic:** `policy_rollouts`

**Transport for v1:** JSON payload (matches how the existing `_do_sim_command`
path already serializes dicts to JSON over `RpcClient.Publish`). A typed
`.msg` (like `ApplyForce.msg`) is the "more correct" long-term option but adds
message-converter + regen friction; JSON keeps Phase 1 light. Decision to be
confirmed at implementation time.

**Payload schema:**

```jsonc
{
  "stamp": 1733300000.123,      // publisher wall-clock (for staleness/fade)
  "step": 26771,                // env step index (debug)
  "dt": 0.1,                    // seconds per rollout step (matches sim cadence)
  "horizon": 15,                // H steps per trajectory
  "chosen": { "accel": 0.42, "steer": -0.13 },   // the action actually taken
  "samples": [                  // K sequences, each H actions
    [ {"accel": 0.5, "steer": -0.2}, ... H items ],
    ...
    K items
  ]
}
```

Bandwidth: `K·H` action pairs. At K=8, H=15 that's 120 pairs (~a few KB JSON) at
the throttled cadence (Section 5) — negligible.

---

## 5. Component 2 — Python publisher hook

**Where:** a small observer/callback invoked from the actor stepping loop. Two
candidate insertion points:

- The `collect_actor` loop in `main()` (training-time viz), and/or
- `run_policy`'s `eval_actor` loop (eval-time viz).

Wrap in a helper `maybe_publish_rollouts(policy, time_step, api, step)` so both
sites share one implementation.

**Cadence:** throttle to every `ROLLOUT_VIZ_EVERY_N_STEPS` (default ~5–10) and/or
a min wall-clock interval, so we don't flood ROS at the per-step rate. Gate the
whole feature behind an env var `ROLLOUT_VIZ_ENABLED` (default off) so normal
training pays zero cost.

**Sampling (open-loop, H-step sequences):**

```python
dist = policy.distribution(time_step).action      # tanh-normal over [accel,steer]
samples = []
for _ in range(K):
    seq = [dist.sample().numpy().reshape(-1).tolist() for _ in range(H)]
    samples.append(seq)
chosen = dist.mode_or_sample(...)  # or reuse the action already chosen this step
```

- Each of the K sequences is **H i.i.d. draws** from the *same* current
  distribution `π(·|o₀)`. This yields varied curves (each step a different
  sample) while staying strictly open-loop.
- Clamp/også note: the env maps `action[0]=accel`, `action[1]=steer`; keep that
  ordering in the payload so Unity interprets them correctly.
- Publish via the existing `RobotApi` publish path (add a
  `PublishRollouts(payload)` method mirroring how `_do_sim_command` publishes).

**Multi-actor note:** under `--num-envs N`, only publish for **actor 0** (or a
configurable index) so we visualize one car, matching the single Unity client
that renders. Each env subprocess holds its own policy clone; gate on the
actor index.

---

## 6. Component 3 — Unity subscriber + renderer

New MonoBehaviour `TrajectoryRolloutViz` (in `unity/Assets/Scripts/`):

1. **Subscribe** to `policy_rollouts` via `ROSConnection` (same pattern as
   `SimController`'s `ros.Subscribe<SimCommand>(...)`).
2. On each message, capture the car's **current pose** (`transform.position`,
   `transform.eulerAngles.y`) and **current speed** (`CarController.GetSpeed()`).
3. For each of the K sampled sequences, **forward-simulate** the bicycle model
   (Section 7) to produce `H+1` world-space waypoints (at `trackY`).
4. Render each trajectory with a pooled `LineRenderer` (K renderers reused
   frame-to-frame; uniform color, slight per-line width). Draw the `chosen`
   action's path in a distinct style (e.g. thicker) for reference.
5. **Fade/stale handling:** if no message arrives within T seconds, hide the
   lines (avoids a frozen "ghost" fan when the trainer pauses/eval ends).

Performance: K LineRenderers of H segments, refreshed at the throttled cadence —
trivial. Reuse the renderer pool; never instantiate per message.

---

## 7. Component 4 — Kinematic (bicycle) forward model + calibration

State `(x, z, yaw, v)` on the ground plane (Unity `y` is up). Per step, given
action `(accel, steer)` and `dt`:

```
steerAngle = steer * maxSteeringAngle          # degrees -> radians
v      = clamp(v + kAccel * accel * dt, 0, vMax)
yawDot = (v / wheelbase) * tan(steerAngle)
yaw    = yaw + yawDot * dt
x      = x + v * sin(yaw) * dt                 # Unity yaw is CW from +z
z      = z + v * cos(yaw) * dt
```

**Parameters to calibrate against `CarController`:**

| Param | Source / method |
|---|---|
| `wheelbase` | Distance between front/rear axle transforms (`axleInfos`) |
| `maxSteeringAngle` | Read directly from `CarController.maxSteeringAngle` |
| `vMax` | Observed top speed, or derived from `maxMotorTorque` + drag |
| `kAccel` | Fit: drive constant `accel`, measure `dv/dt`; tune until predicted vs actual path match |

**Calibration procedure:** drive the real car with a few constant
`(accel, steer)` inputs, log the actual trajectory (already available via scene
data), and tune `kAccel` / `vMax` until the model's predicted path overlays the
real one for ~1–2 s. Exact match isn't required — this is a visualization, not a
controller — but it should be visually faithful over the horizon.

> The WheelCollider physics in `CarController` is more complex than a kinematic
> bicycle (slip, torque curves). The bicycle model is a deliberate, cheap
> approximation; Section 8 (closed-loop) would instead use Unity's *actual*
> stepped physics for exactness if needed.

---

## 8. Out of scope for Phase 1 (future phases)

- **Phase 2 — closed-loop rollouts:** refactor `CarController`'s raycast /
  observation construction to evaluate an **arbitrary hypothetical pose**, add a
  **batched policy-query RPC**, and have Unity roll out K×H with real per-step
  observations re-queried from the policy. True state-conditioned trajectories.
- **Phase 3 — polish:** dashboard/Inspector toggle, coloring by **predicted
  return** (needs the critic) or by log-prob, configurable K/H/cadence from the
  job document, trajectory fade animation.

---

## 9. Task breakdown (Phase 1)

1. **ROS plumbing**
   - [ ] Add `PublishRollouts(payload)` to `RobotApi` (`rl_agent/api.py`),
         publishing JSON to topic `policy_rollouts`.
   - [ ] (Optional) define a typed `PolicyRollouts.msg` instead of JSON.
2. **Python publisher**
   - [ ] `maybe_publish_rollouts(policy, time_step, api, step)` helper in
         `robotaxi.py`: throttle, sample K×H from `policy.distribution()`,
         publish.
   - [ ] Call it from the `collect_actor` loop (and/or `run_policy`'s eval loop).
   - [ ] Env-var gating: `ROLLOUT_VIZ_ENABLED`, `ROLLOUT_VIZ_EVERY_N_STEPS`,
         `ROLLOUT_VIZ_K`, `ROLLOUT_VIZ_HORIZON`, `ROLLOUT_VIZ_DT`.
   - [ ] Multi-actor: publish only for the configured actor index.
3. **Unity renderer**
   - [ ] `TrajectoryRolloutViz.cs`: subscribe to `policy_rollouts`, bicycle
         forward-sim, pooled `LineRenderer` rendering, stale-fade.
   - [ ] Wire car pose + speed access (`CarController.GetSpeed()`).
   - [ ] Inspector knobs: line width, color, max trajectories, fade timeout.
4. **Calibration**
   - [ ] Read `wheelbase` / `maxSteeringAngle` from the car; tune `kAccel`/`vMax`.
   - [ ] Visual A/B: predicted vs actual path over ~1–2 s.
5. **Validation**
   - [ ] Confirm payload on the wire (Python publishes, topic carries K×H).
   - [ ] Confirm K fanned lines render at the car and update at the cadence.
   - [ ] Confirm zero overhead when `ROLLOUT_VIZ_ENABLED` is unset.

---

## 10. Risks & notes

- **Heuristic, not a true rollout (Phase 1).** H i.i.d. samples from the current
  distribution shows action *spread*, not the policy's actual closed-loop path.
  Label it as such in any UI affordance so it isn't misread as a prediction.
- **Bicycle vs WheelCollider mismatch.** Predicted paths will diverge from real
  motion under hard accel/steer or wheel slip; acceptable for a spread viz over
  a short horizon.
- **Distribution access.** Must use the live policy (`.distribution()` works);
  never a loaded SavedModel (serialization limitation).
- **Bandwidth/throttle.** Keep the publish cadence well below per-step to avoid
  ROS churn; default off via env var so training is unaffected.
- **Multi-actor.** Only one Unity client renders; publish for one actor index.
