# Driving the sim car with a fruit fly connectome

Twelve steps in a fixed order, in five parts. Each part ends with something you
can look at, so you never build two things before finding out the first one
works. Parts 1 and 2 need no Unity at all, and Part 2 is a cheap go/no-go on
the whole idea.

The idea in one line: [fly.ai](https://github.com/alextitonis/fly.ai)'s
`flybrain` is a frozen 166,700-neuron *reservoir*, and our SAC trainer becomes
the *readout* that fly.ai normally fits with ridge regression.

```
31-D obs  ->  encoder  ->  fly connectome (FROZEN)  ->  descending-neuron trace  ->  readout  ->  [accel, steer]
              (Part 2)      166,700 neurons                  1,314 outputs         ridge (Part 2)
                                    |                                              SAC   (Part 5)
                                    +-> spike snapshot -> Unity overlay (Part 4)
```

Nothing inside the brain ever trains. Its weights are the MaleCNS v1.0 wiring
diagram, fixed at load time. The only learned thing is what we read off the
1,314 descending neurons, which is exactly how `flybrain/reservoir.py` is meant
to be used.

Sibling docs: [`trajectory-rollout-viz.md`](trajectory-rollout-viz.md) is the
overlay this plan copies, and
[`csi-camera-observation-guide.md`](csi-camera-observation-guide.md) is the
precedent for adding a course with a new observation shape.

---

## Verified interface facts

These were read out of the code, not assumed. Getting any of them wrong
produces a plausible-looking car that drives badly for reasons you won't find.

**Action spec** (`donut_course.py:136`) is `shape=(2,)` ordered
**`[acceleration, steering]`** — acceleration first:

```
minimum = [0.05, -1.0]      maximum = [1.0, 1.0]
```

`robotaxi.py` independently confirms `action[:, 0]` is "the force channel".
Writing steering into slot 0 sends steering commands to the throttle.

**Observation layout** for `donut_no_hint` (31-D) is speed, sideslip, then the
29 rays:

| Index | Content |
|---|---|
| 0 | speed |
| 1 | sideslip (`goal_2`, angle from velocity to heading) |
| 2–30 | 29 ray distances |
| 16 | forward clearance (the 0° ray) |

The 32-D base `donut` course has one extra leading column
(`dist_from_traj`, the goal-angle hint), so every index above shifts by one.
If you port index constants from anything written against the 32-D vector,
shift them down.

**The rays are contiguous but NOT sorted by angle.** In index order:

```
idx:    2    3    4     5     6  ...  15   16   17  ...  27   28   29   30
deg:  -90  -30  -60  -27.5  -25  ... -2.5    0  2.5  ...  27.5  60   30   90
```

Two transpositions: indices 3/4 and 28/29. A plain left/right split survives
this, because both swaps are within one side — left is 2–15, forward is 16,
right is 17–30. It stops being safe the moment the encoder weights a ray by its
angle, which a good encoder should. Angles come from
`CarController.SetUpDirectionToAngle()`.

**Control rate is 10 Hz in simulated time.** `SceneDataPublisher` publishes
every 0.1 s, so one env step is 0.1 s of sim time and the brain should run
**K = 0.1 / dt = 5 substeps** per control step at the default `dt=0.020`. (Note
the source comment on that line says "20Hz" and is wrong; `0.1f` is 10 Hz.)
Wall-clock rate is faster because `Time.timeScale` is 3–5, which matters for
throughput but not for how many brain steps a control step deserves.

---

## Why this mapping is a good fit

The fly's two documented, *working* visual pathways happen to be the two things
a racing policy needs:

| Fly pathway | Neuron types | Racing equivalent |
|---|---|---|
| Looming → escape | LC4, LPLC2 → DNp01 | obstacle close and closing → turn away |
| Courtship pursuit → steering | LC10a → DNa02 | free space is that way → steer toward |

Both are side-specific (left stimulation moves left outputs only), which is
what makes steering decodable at all. fly.ai measured both: left LC4+LPLC2
raised left DNp01 by 17–25 spikes/s, and left LC10a raised left DNa02 by
1.4–3.7 spikes/s, with the other side unchanged.

**The named command neurons are single cells.** Step 1 found DNa02, DNp01 and
DNg100 are exactly **one neuron per side** (MDN is two). At 50 Hz a single
neuron contributes 0–3 spikes per control step, so any hand decoder reading an
instantaneous rate off DNa02 is reading a near-binary signal. Use the decaying
`Trace` rather than a per-step rate, and treat this as another reason the
learned readout over all **1,314** descending neurons (L 656 / R 648 / M 10) is
the route more likely to work than hand decoding.

**Do not use the photoreceptor/eye route.** fly.ai's own findings 1 and 2 are
that vision does nothing with Fly64's settings, and that the photoreceptor
signal dies at the first relay because histamine is inhibitory and real lamina
neurons are graded, which a spiking point-neuron model can't reproduce. The
working path injects the feature-detector neurons directly.

**Target `donut_no_hint`, not `donut`.** The 32-D course's leading column is
the goal-angle hint, which would do the steering for you and teach you nothing
about the wiring.

---

## Part 1 — Get the brain running and prove it responds

Desktop only. No Unity, no training.

### Step 1 — Stand up a `fly-brain` container *(desktop)*

**Do this.** Add a `fly-brain` service to `docker-compose.yml` on a Python 3.11
CUDA base. Install `flybrain[gpu]`, run `flybrain download` into a persistent
`$FLY_DATA` volume, then `flybrain info`. Time 100 `brain.step()` calls.

**Why it matters.** `flybrain` needs Python 3.10+ and `sim-controller` is
Python 3.8, so it cannot live in the trainer. Timing it now tells you whether
the plan is affordable before you write integration code.

**You are done when.** `flybrain info` reports CUDA working, the build reports
exactly **166,700 neurons and 25,582,938 connections**, and a step costs
roughly 1–4 ms. If you see 9–15 ms you are on CPU and everything downstream gets
ten times slower.

> **DONE — measured 2026-09-19.** Counts match exactly. `device='cuda'`,
> **2.67 ms/step** (374 steps/s), so K=5 costs **13.4 ms per env step**. That is
> slower than fly.ai's 1.4 ms on an RTX 4060, most likely because a single
> unbatched sparse multiply is launch-latency bound and WSL2 GPU passthrough
> adds overhead — batching across actors should amortize it.
>
> Useful things step 1 established for later steps:
> - `tonic`, `gain`, `decay`, `noise_amp`, `noise_hz`, `refractory_steps` are
>   plain settable attributes on the brain, so step 2 sets them directly.
> - `positions`, `indices`, `indptr`, `weights`, `cell_type`, `side` and
>   `superclass` are all exposed — step 7's overlay export needs no extra work.
> - `flybrain` exports `Trace`, `Readout`, `run`, `fit_ridge`, `fit_logistic`,
>   `ENCODER` and `FeatureDetectors` at the top level.
> - Every neuron type this plan needs resolves: LC4 (L71/R55), LPLC2 (L94/R91),
>   LPLC1 (L68/R66), LC10a (L135/R140).

### Step 2 — Reproduce fly.ai's one solid result *(desktop)*

**Do this.** Copy the settings out of `inject.py` (tonic 0.14, gain 3.0 — read
them from the file rather than trusting this doc). Stimulate
`brain.cells(["LC4","LPLC2"], side="L")` and check left DNp01 rises while right
DNp01 does not. Repeat with `LC10a` → DNa02.

**Why it matters.** This is the only part of the fly.ai stack with a measured,
side-specific, reproducible effect, and the entire encoder rests on it. If it
doesn't reproduce, your parameters are wrong and everything after this is noise
you'll spend a week misinterpreting.

**You are done when.** Left DNp01 gains roughly 17–25 spikes/s over baseline
with the right side flat, across a few noise seeds.

> **DONE — measured 2026-09-19** (`docker/fly_brain/step2_inject.py`). All four
> side-specificity checks pass, bilaterally:
>
> | Stimulus (×0.8) | Same side | Other side |
> |---|---|---|
> | loom_L → DNp01 | **+25.2** | +0.1 |
> | loom_R → DNp01 | **+24.8** | +0.4 |
> | chase_L → DNa02 | **+3.9** | −0.1 |
> | chase_R → DNa02 | **+3.0** | −0.2 |
>
> fly.ai's published "+17 to +25" turns out to be the strength sweep, not
> seed noise: ×0.3 gives +17.0 and ×0.8 gives +25.2. Same for chase, where
> ×0.3 gives +1.8 and ×0.8 gives +3.9 against their +1.4–3.7.
>
> **Right-side stimulation works too**, which inject.py never tested — it only
> ever drives the left. Our encoder depends on both, so this was worth checking.
>
> Three things that change later steps:
> - **The pip release already defaults to `tonic=0.14, gain=3.0`.** No setting
>   required. Running Fly64's `0.18 / 1.5` for contrast roughly halves every
>   effect (DNp01 +12.1 instead of +25.2) and triples the resting rate to
>   3.2 Hz, so the defaults are the regime you want.
> - **`brain.side` is identical to inject.py's feather-derived sides**, so
>   `flybrain download` is sufficient and the 1.1 GB `flybrain build` is not
>   needed.
> - **`brain.groups` ships curated motor groups**: `steer_L/R`, `forward_L/R`,
>   `escape_L/R`, `backward_L/R`, `punch_L/R`, `kick_L/R`. Prefer these over
>   hand-picking cell types in steps 6 and 10.

### Step 3 — Wrap it in a small service *(desktop)* — DONE

**Do this.** Expose the calls over gRPC, to match the rest of the stack.

**Why it matters.** The brain is *stateful* — voltages carry across steps — so
episodes must reset it, and each actor needs its own copy. `FlyBrain(batch=N)`
shares one sparse multiply across actors at about 1.2 ms per fly per step.

**You are done when.** A test client can reset, step 100 times, and get a trace
of the right shape.

#### What was built

| Piece | Path |
| --- | --- |
| Contract | `protos/fly_brain/proto/fly_brain.proto` |
| Server | `docker/fly_brain/fly_brain_server.py` |
| Overlay subset | `docker/fly_brain/display_subset.py` |
| Codegen | `docker/fly_brain/gen_protos.sh` |
| Trainer client | `rl_agent/fly_brain/client.py` |

Six RPCs, not four: `Info` and `Cells` were added. `Cells` resolves cell types
or a named group to neuron indices, which the step-4 encoder needs and only the
brain side can answer.

The container's `CMD` is the server, so bringing it up is enough:

```powershell
docker compose up -d fly-brain
docker compose exec -w /python_ws/src sim-controller python -m fly_brain.client
```

#### Verified facts

- **Latency is a non-issue.** Warm, 100 steps of `k=5` from the trainer:
  **9.6 ms per control step (104 Hz)** against a 100 ms budget. The brain alone
  is 9.0 ms in-process, so gRPC costs about **0.6 ms** — comfortably inside the
  "well under a millisecond" this step asked for, with 10x headroom. Overlay
  bookkeeping is free (0.03 ms), so snapshots can ride along on every step.
- **A newly *created* container costs about 30 ms/step extra for the first
  ~100 steps.** CuPy compiles kernels on first use, so the first run after
  `up --force-recreate` or a rebuild sits at ~40 ms/step (25 Hz). The cache
  lives in the container filesystem, so a plain `restart` keeps it and comes
  back at ~10 ms. Still inside budget either way, but the smoke test does a
  warmup pass before timing, and a training run should too.
- **It is exactly reproducible.** Same seed and injections twice gives
  `max|trace - trace2| = 0.0`.
- **The trace is 1,314 wide, not 1,304.** `brain.cells(["descending_neuron"])`
  returns 1,314. Other places in this doc saying 1,304 are off by ten; size the
  readout from `Info.trace_len`, never from a literal.
- **The display subset is 2,034 neurons and 8,000 edges**, which step 7 can
  refine without touching the contract.

#### Two things that constrain later steps

**Injection amounts are per-population, not per-neuron.** `FlyBrain._amount`
reshapes any array to `(1, batch)` and reads it as one value *per fly*, so
`stimulate(idx, array_of_len_k)` fails with "Out shape is mismatched". The
contract therefore takes a repeated `Injection{idx, scalar amount}`. A step-4
encoder wanting graded drive emits several Injections, one per sector — which
is how the four scalars in step 4 were going to work anyway.

**The gRPC toolchain is pinned and must stay that way.** sim-controller is
Python 3.8 with grpcio 1.51.1 / protobuf 3.20.1, and protobuf 4.x codegen will
not load on a 3.x runtime. `grpcio-tools` is therefore pinned to **1.48.2**, the
last release whose protobuf floor is below 4.0 — 1.51.1 requires protobuf
>=4.21.6 and fails to resolve. Regenerate only via `gen_protos.sh` inside the
fly-brain container, then copy `docker/fly_brain/gen/fly_brain/proto/*` to
`rl_agent/fly_brain/proto/`. Codegen uses `-I /protos` so the emitted import is
`from fly_brain.proto import ...`, matching `virtual_endpoint`.

---

## Part 2 — Find out whether there's any signal, with no Unity

This part is the cheap go/no-go. If the connectome carries nothing useful for
driving, you learn it here in an afternoon instead of after building a bridge,
an overlay and a training course.

### Step 4 — Write the encoder: 29 rays into fly neurons *(desktop)*

**Do this.** From the ray block (indices 2–30, angles above), compute four
scalars per control step: left looming, right looming, left chase, right chase.
Looming on a side is large when the nearest obstacle on that side is close *and*
closing — keep the previous frame's ranges to get the closing rate. Chase on a
side is large when that side holds the most open space. Inject looming into
`["LC4","LPLC2"]` and chase into `["LC10a"]` on the matching side.

**Why it matters.** This is the whole translation between our world and the
fly's, and the only part with no reference implementation to copy. Closing rate
matters because LPLC2 is a *looming* detector: a static wall at 1 m and a wall
rushing at you from 1 m should not look alike.

**You are done when.** Synthetic scans move the expected side — a wall
approaching on the left raises left DNp01, open space on the right raises right
DNa02.

> **DONE — measured 2026-09-19.** Encoder in `rl_agent/fly_brain/encoder.py`,
> acceptance test in `rl_agent/fly_brain/step4_check.py`:
>
> ```powershell
> docker compose exec -w /python_ws/src sim-controller python -m fly_brain.step4_check
> ```
>
> All four scenarios pass, side-specifically. Values are the settled trace
> delta against a symmetric baseline; at `tau=0.1` a trace of 3.0 is about
> 30 spikes/s, consistent with step 2's +25.2.
>
> | Scenario | Same side | Other side |
> |---|---|---|
> | wall closing left → DNp01 | **+3.020** | −0.003 |
> | wall closing right → DNp01 | **+2.983** | +0.078 |
> | open space left → DNa02 | **+0.444** | −0.004 |
> | open space right → DNa02 | **+0.333** | −0.043 |
>
> **The strongest result is that the chase contrast already correlates −0.808
> with the expert's steering** across all 500,001 corpus rows, before the brain
> is involved at all. The encoder is demonstrably not discarding the steering
> signal, so a weak step-5 R² would indict the brain or the readout, not this.
> Looming asymmetry correlates only +0.008, as expected on a course whose only
> obstacles are static walls.
>
> Four measured facts that forced the design:
>
> - **Rays must be weighted by `cos(angle)`.** The ±90° rays sit at a median
>   6.3 m and the ±60° rays at 7.4 m, against 25 m straight ahead — they are
>   pinned to the track wall on both sides, every frame. An unweighted per-side
>   minimum is therefore the wall and carries no information. `cos` is exactly
>   0 at ±90° and 0.5 at ±60°. This is the angle weighting the ray-order note
>   above warns about, so the non-monotonic order genuinely matters now.
> - **A raw frame-to-frame ray difference is not a closing rate.** 23.5% of
>   per-ray steps move more than 1.0 m while the 6.7 m/s top speed allows only
>   0.67 m. The rays rotate with the car, so a ray sliding off a wall edge
>   reports a discontinuity that is not motion. Since every obstacle is static,
>   clamping closing to `1.5 × speed` fixes it: looming's p99 fell from 11.8 to
>   0.77 and its max from 3038 to 17.3. This is why the encoder needs the speed
>   channel and cannot work from the rays alone.
> - **A ray reading exactly 0 means "never hit anything", not "obstacle at zero
>   range".** `CarController.DrawRay` leaves `distToClosestObjects[d]` untouched
>   on a miss and the array starts zeroed, so a miss reports a *stale* value.
>   0.035% of readings are exact zeros; the encoder reads them as maximally
>   open. The staleness is also part of why the raw difference is unusable.
> - **Gains are calibrated, not guessed.** `LOOM_GAIN=1.04` and
>   `CHASE_GAIN=1.44` put each cue's corpus p99 at the 0.8 injection strength
>   step 2 found strong, so ordinary frames use the dynamic range rather than
>   clipping.
>
> **One API trap, now designed out.** `cues()` is the only stateful call.
> Asking for the cues and then the injection separately advances the frame
> twice, and the second call sees no change in range, which silently zeroes
> looming while leaving stateless chase working — a failure that looks like a
> dead pathway. `encode(obs, cell_idx)` is the single entry point that does
> both from one advance.
>
> **For step 5:** the expert actions are small and off-centre — accel p50 0.012
> and p99 0.200 against a `[0.05, 1]` spec, steer p50 0.041 within `[-1, 1]`.
> R² is scale-invariant so this does not distort it, but do not expect the
> readout to need the full action range.

### Step 5 — Fit a readout on the existing demo corpus *(desktop)*

**Do this.** Read the expert-demo corpus for job `64168c1b58d4d8ccdb76e721`
(the baked-in default for both `donut` and `donut_no_hint`) through
`collect_training_data.convert_tfrecord_to_trajectory`, which already drops the
leading column for you. Replay each observation through the frozen brain,
collect the descending trace, and `Readout.fit(activity, actions, kind="ridge")`
against the recorded expert `[accel, steer]`. Report held-out R² per channel.

**Why it matters.** This is fly-brain behaviour cloning with zero new data
collection, no Unity, no live loop and no RL — and it answers the only question
that matters before you build anything else: does the trace contain enough about
the scene to recover an expert action? Do it before Part 3, not after.

**You are done when.** You have held-out R² for steering and acceleration. A
steering R² meaningfully above zero means the encoder works and the rest of the
plan is worth building. Near zero means go back to step 4 and retune the
encoder gains — fly.ai found looming often fails to propagate depending on
tonic and gain, so that's the knob, not the readout.

> **DONE — measured 2026-09-19.** `rl_agent/fly_brain/step5_readout.py`,
> 100 episodes (100,000 rows), split by episode with 20 held out, ridge with λ
> chosen on a validation slice of the training episodes only. The 18-minute
> replay is cached at `/tmp/fly_step5_trace_100ep.npz`, so refits are instant.
>
> | Features | R² accel | R² steer |
> |---|---|---|
> | descending trace, 1,314 features | 0.176 | 0.620 |
> | the 4 encoder cues — the brain's whole input | 0.045 | **0.687** |
> | chase asymmetry alone, 1 feature | 0.000 | 0.660 |
> | raw 31-D observation (upper reference) | **0.208** | **0.747** |
>
> **Steering passes the stated bar and the brain still is not earning its
> keep.** R² 0.620 is far above zero, but the four scalars fed *into* the brain
> score 0.687 and a single hand-computed scalar scores 0.660. As a steering
> channel the connectome is lossy: 0.687 in, 0.620 out. The plan's criterion
> was necessary, not sufficient — without the input control the 0.620 reads as
> a success.
>
> **Throttle is the opposite, and it vindicates step 6.** The trace scores
> 0.176 against 0.045 for its own instantaneous input, recovering most of the
> raw observation's 0.208. The brain is supplying something its input does not
> have: it is a recurrent system whose state integrates history, and throttle
> depends on where the car is in a manoeuvre rather than on the current frame.
> Step 6 already concluded that throttle has no hand-decodable source and must
> come from the learned readout — that is now measured, not assumed.
>
> **The bottleneck is the encoder, not the connectome.** Losses compound in a
> clear order: 0.747 raw → 0.687 after the encoder compresses 29 rays into 4
> scalars (−0.060) → 0.620 after the brain (−0.067). No readout can recover
> what the encoder discarded before the brain ever ran, so retuning gains (the
> knob this step suggests) cannot close the steering gap. Giving the brain a
> *retinotopic* input — mapping rays across many LC populations by angle rather
> than aggregating to four numbers — is the change with room to pay off, since
> it is the only one that raises the 0.687 ceiling.
>
> **What this does and does not license.** Part 3 is worth building: there is a
> usable readout, throttle genuinely benefits, and step 6's eval row will be
> measured against SAC on identical geometry. But do not expect the fly policy
> to beat a raw-observation policy at steering, and treat any such result as a
> bug. Note also that ridge is linear; SAC in Part 5 is not, so the nonlinear
> readout has headroom this number does not measure.
>
> #### The retinotopic encoder: a measured negative result
>
> The conclusion above — "the bottleneck is the encoder" — was tested directly
> and **turned out to be wrong**. `RetinotopicEncoder` splits the rays into 7
> angular sectors per side and drives a matching slice of each LC population,
> instead of collapsing each side to one number. Run it with
> `FLY_ENCODER=retino`.
>
> | Encoder | Cue dim | Ceiling: R² steer on the cues | After the brain |
> |---|---|---|---|
> | flat, 4 scalars | 4 | 0.687 | **0.620** |
> | retinotopic, absolute openness | 28 | 0.869 | 0.488 |
> | retinotopic, mirrored contrast | 28 | **0.873** | 0.581 |
>
> **A much better input produced a worse output.** Raising the encoder ceiling
> from 0.687 to 0.873 — past the raw observation's 0.747, because the cues are
> a nonlinear transform carrying temporal memory the raw single frame lacks —
> moved the brain's output *down*, from 0.620 to 0.581. Keep the flat encoder;
> `ENCODER` defaults to it.
>
> Two diagnostics explain it, and rule out the obvious guess:
>
> - **Not saturation.** Spikes per step (72.9k vs 73.4k), trace mean (0.323 vs
>   0.339) and trace variance across frames (0.228 vs 0.234) are all
>   indistinguishable between the two encoders. Scaling the drive down 10x
>   changes nothing either.
> - **The brain cannot express within-eye sector structure.** Driving one
>   sector at a time and comparing the descending responses, within-eye pairs
>   have mean cosine similarity **0.476** against **0.460** across eyes — the
>   response to any single sector is largely common-mode, and two sectors of
>   the same eye are no more alike than one from each eye. The connectome reads
>   left-versus-right well and sector-versus-sector essentially not at all.
>
> That also explains the absolute-openness variant being worst: absolute
> openness sits near 0.6 on both sides, so the left/right contrast the brain
> *can* read is buried in common mode. Mirroring the sectors restored most of
> the loss (0.488 → 0.581) without beating the flat encoder, which already
> delivers that contrast at full amplitude.
>
> **Scope of the conclusion.** This falsifies retinotopy *for the partition
> available here*, which sorts each population along the principal axis of its
> soma positions. The prebuilt connectome exposes only soma coordinates —
> `positions`, `cell_type`, `side`, `superclass`, `groups` — and no receptive
> field, so a true retinotopic map cannot be built from it. A second, weaker
> confound remains: splitting the contrast over 7 sectors drives ~20 cells per
> channel where the flat encoder drives all ~135, so lower per-channel drive
> may contribute alongside the separability result.
>
> **The real conclusion is that the brain, not the encoder, is the binding
> constraint for steering.** Handed a 0.873 input it returns 0.581; handed a
> 0.687 input it returns 0.620. More input information does not become more
> output information, so effort is better spent on the nonlinear readout in
> Part 5 than on further encoder work.

---

## Part 3 — Drive the car, with no RL yet

### Step 6 — Run a `FlyPyPolicy` through the existing EVAL path *(sim)*

**Do this.** Wrap encoder → brain → readout in a `PyPolicy` with the course's
own action spec, and run it through the normal EVAL dispatch the way
`RandomPyPolicy` already does (`robotaxi.py` has a `model_type ==
"RandomPyPolicy"` branch and writes it a Models-tab row). Use the step-5 readout
if it worked, otherwise a hand decoder: steering from the left/right `steer_L` /
`steer_R` group difference. Run with `--num-envs 1`.

**Throttle has no hand-decodable source — use the learned readout for it.**
Step 2 measured DNg100 (forward walking) at +0.0 to +0.1 under every stimulus
and strength tested, so the "throttle from DNg100" idea does not survive
contact with the data. Neither looming nor chase drives it. Either hold
acceleration at a constant while you validate steering, or take throttle from
the step-5 ridge readout over all descending neurons, which does not depend on
any one neuron carrying the signal.

**Why it matters.** This buys the entire measurement apparatus for free —
AverageReturn, goals per episode, crashes per 1k steps, and a leaderboard row
directly comparable to SAC on identical geometry. That is a far better
experiment than "does it drive", and it costs less than a bespoke harness.

**You are done when.** You have an eval row for the fly policy next to your SAC
baseline. **Expect it to drive poorly**, especially on the hand decoder — fly.ai
tried exactly this for their Wiz character and reported three calibration
attempts that all failed their pre-set criteria. A weak number here is
information; Part 5 is the path that doesn't assume the textbook mapping.

**Before recording any number you'd want to compare:** greedy SAC eval takes
`tanh(μ)` from a distribution, while the fly readout is a point estimate with
spiking noise underneath. Fix the noise seed per episode, or average the trace
over the substeps, or the two variances aren't comparable.

> **DONE.** `rl_agent/fly_brain/policy.py` wraps encoder → brain → step-5 ridge
> readout as a `FlyPyPolicy`; `robotaxi.py` gained a `run_flypolicy()` and a
> `model_type == "FlyPyPolicy"` EVAL branch alongside the RandomPyPolicy one.
> The policy resets the brain and the encoder on every `StepType.FIRST` and
> seeds the reset with the episode index, which is the per-episode fixed seed
> the note above asks for. Its readout lives at
> `/saved_models/robotaxi/FlyPyPolicy/0/readout.npz`, and the first run
> registers its own `models` record because there is no TRAIN job to have
> created one.
>
> **It drives. Badly, but unmistakably better than chance.** 2 trials x 3
> episodes on `donut_no_hint`:
>
> | Policy | AverageReturn | Goals/episode | Episode length | Speed |
> |---|---|---|---|---|
> | RandomPyPolicy | ~1.0 | — | — | — |
> | **FlyPyPolicy** | **6.87** | 8.0 / 12.7 | 297 / 394 | 1.93 / 2.43 |
> | SAC `7573_step_87314` | 75.7 | 78–100 | 1105–1383 | 5.2–5.5 |
>
> So the frozen connectome is worth about **7x a random policy and about a
> tenth of SAC**. It holds a lane well enough to collect 8–13 goals before
> crashing, rather than the ~1 goal random scores. The plan predicted "expect
> it to drive poorly" and that is what happened, but the failure mode is
> informative: the car is not crashing at once, it is driving slowly and
> eventually leaving the track.
>
> **Throttle is the weaker of the two channels, as step 5 predicted.** The
> readout's accel prediction (R² 0.176) sits below the course's own 0.05
> action floor on ~54% of frames, so the clip to the action spec is doing real
> work and the car is effectively pinned near minimum throttle half the time —
> hence 1.9–2.4 m/s against SAC's 5.4. Steering (R² 0.620) is what is keeping
> it on the track.
>
> **Caveat on comparability.** This ran on the current `unity/Builds/latest`
> build, not the `wCourseJetRacer2026.09.06-v10` build the SAC row was measured
> on. Same course type and same reward, but the geometry is not guaranteed
> identical, so treat the SAC column as a scale reference rather than a
> controlled head-to-head until it is re-run on one build.

---

## Part 4 — The overlay

Build this before training, because watching the brain is how you'll debug
everything in Part 5.

### Step 7 — Export the display subset once *(desktop)*

**Do this.** Pick a few thousand neurons: both sides of LC4, LPLC2, LPLC1 and
LC10a, all 1,314 descending neurons, the named command neurons (DNa02, DNp01,
DNg100, MDN), and a sample of the strongest interneurons between them. Take soma
positions from `brain.npz`. Keep only the strongest few thousand edges *among
that subset*. `flybrain export --web` already does most of this for the browser
build — start there rather than writing your own.

**Why it matters.** 166,700 neurons and 25.6 million edges can't be drawn at
frame rate and nobody could read them anyway. A curated few thousand showing
sensory inputs, motor outputs and the paths between is both drawable and
legible.

**You are done when.** You have one JSON file of positions and edge pairs,
comfortably under a few MB.

> **DONE.** `docker/fly_brain/display_subset.py`, served by the `Geometry` RPC
> rather than written to a file — the overlay needs it over the wire anyway and
> a file would be a second thing to keep in sync. **3,225 neurons and 8,000
> edges**, in 166 KB on the wire: 720 sensory (LC4/LPLC2/LPLC1/LC10a), 1,292
> descending, 16 named command neurons, and 1,197 relay interneurons.
>
> Two corrections came out of actually rendering it:
>
> - **Six descending neurons carry no soma position in `brain.npz`.** They are
>   0.3% of the subset and they made *every* drawn position NaN, because the
>   centre and extent used to normalize are means and maxima over all of them.
>   They are now dropped in `build()`, before the slot map and edge list, so
>   `n_display` means "what can actually be drawn" and no consumer has to
>   defend itself.
> - **The first version had no interneurons at all.** `idx` was the union of
>   sensory, command and readout, so the "interneuron" role existed in the code
>   and matched nothing: the overlay drew inputs and outputs with empty space
>   between them, and step 9's "its path to DNp01" had no path to light up.
>   `_relay_interneurons()` now scores candidates by (weight received from
>   sensory) x (weight sent to descending) and keeps the top 1,200. That buys
>   585 sensory→relay edges and 940 relay→descending edges, so the two-hop path
>   is visible. Candidates are restricted to actual targets of the sensory
>   populations, which keeps this a few thousand row scans instead of a pass
>   over all 25.6M connections — startup stayed at 2.0 s.

### Step 8 — Publish geometry once, activity continuously *(sim)*

**Do this.** Add two topics: `fly_brain_geometry` (the step 7 file, published
once when Unity connects) and `fly_brain_activity` (per-neuron intensity as
base64 `uint8`, at 20 Hz). **Both must be added to the static routing table in
`docker/ros_server/ROS/src/niryo_moveit/scripts/unity_node.py` as
`RosSubscriber` entries** or Unity will never receive them.

**Why it matters.** That routing table is static — the embedded ROS-TCP
connector doesn't register subscribers dynamically — and it's the single most
likely reason a new topic silently does nothing. Splitting static geometry from
per-frame activity keeps the stream small: 5,000 neurons as `uint8` is 5 KB a
frame, about 130 KB/s after base64. Sending float JSON every frame, or resending
geometry, recreates the saturation we already hit with the camera feed.

**You are done when.** `rl_agent/check_rollouts.py`, pointed at the new topic,
shows geometry arriving once and activity arriving steadily at 20 Hz.

> **DONE.** `rl_agent/fly_brain/viz.py`. Both topics are registered in
> `unity_node.py`. Verified with the now topic-aware `check_rollouts.py`:
>
> ```
> python check_rollouts.py ros-server-0:50051 25 fly_brain_activity
>   Received 347 message(s). ~18.0 Hz. OK - data is flowing.
> python check_rollouts.py ros-server-0:50051 25 fly_brain_geometry
>   Received 2 message(s). OK - data is flowing.
> ```
>
> Three decisions worth keeping:
>
> - **Every numeric field is base64 of a little-endian buffer, geometry
>   included** — not just the activity bytes the plan called for. Measured at
>   the final 3,225-neuron subset: **184 KB against 348 KB** for the same arrays
>   as JSON numbers, and Unity's `JsonUtility` would otherwise allocate and
>   parse ~33,000 boxed floats on each resend; `Convert.FromBase64String` plus a
>   `Buffer.BlockCopy` is one allocation.
> - **Geometry is resent every 10 s, not once.** The routing table is static
>   and gives Unity no way to ask for a resend, so a client that connects (or
>   reloads its scene) after the first send would otherwise never draw
>   anything. At 184 KB that averages ~18 KB/s against the activity stream's
>   ~86 KB/s (4.3 KB per frame at 20 Hz).
> - **Positions are normalized to a unit box before publishing**, so the Unity
>   side is one scale factor instead of raw MaleCNS soma coordinates (which run
>   to ~84,000 on the x axis).
>
> The publish is a blocking gRPC round trip and `FlyPyPolicy` calls it from
> inside its action loop, so it goes through a background thread holding only
> the newest frame. A dead ros-server costs the driving loop nothing and stale
> frames are dropped rather than queued. The snapshot rides along on the `Step`
> the policy already makes (`want_snapshot`), so the overlay costs no extra
> round trip. Off with `FLY_VIZ_ENABLED=0`.

### Step 9 — Render it in Unity *(sim)*

**Do this.** Write `unity/Assets/Scripts/FlyBrainViz.cs` modeled directly on
`TrajectoryRolloutViz.cs` — self-subscribing `MonoBehaviour`, lazily grabbing
`ROSConnection.instance`, JSON-parsing a `StringMsg`. Build one `Mesh` with
`MeshTopology.Lines` for edges and points for neurons on the geometry message,
then update only vertex colors per activity frame. `sshfighter/`'s dashboard is
the reference for *what* to show; `TrajectoryRolloutViz` is the reference for
*how* to wire and draw it here.

**Why it matters.** `TrajectoryRolloutViz` already solved subscribing, stale
payload handling and pooled rendering in this codebase. Uploading the mesh once
and animating only colors keeps a few thousand neurons free at frame rate.

**You are done when.** Driving the car lights up the correct side: approach a
wall on the left and the left looming cluster and its path to DNp01 flare.

> **WRITTEN, NOT YET SEEN.** `unity/Assets/Scripts/FlyBrainViz.cs`, auto-
> attached by `SimController` alongside `HudOverlay`/`TrajectoryRolloutViz`, so
> there is no scene setup to remember. Toggle with the B key; it stays hidden
> until activity actually arrives, so a SAC job doesn't get a frozen brain
> floating over the track.
>
> Two meshes share one `Sprites/Default` material (the same always-available,
> vertex-colour-aware, `Cull Off` shader `TrajectoryRolloutViz` settled on):
> a `MeshTopology.Lines` mesh whose vertices *are* the neurons, so each edge
> interpolates between its endpoints' colours and lights from the firing end;
> and a quad per neuron, billboarded toward the camera in `LateUpdate`, because
> `MeshTopology.Points` renders single pixels that are unreadable at this
> scale. Activity frames rewrite only `mesh.colors` — the vertex buffers are
> uploaded once. Role sets the hue (sensory cyan, interneuron grey, command
> amber, descending red) and intensity sets brightness and alpha, so the
> structure stays legible while the brain is quiet.
>
> **This cannot be verified from here.** There is no headless Unity build in
> this repo — `PromoteLatestBuild.ps1` promotes a folder the Editor produced.
> Verifying needs: build from the Unity Editor into `unity/Builds/<name>/`, run
> `scripts/PromoteLatestBuild.ps1`, then `scripts/Start-Clients.ps1`, and run a
> `FlyPyPolicy` EVAL job. The ros-server image also has to be rebuilt for the
> routing-table change (`docker compose build ros-server`), since
> `unity_node.py` is baked in rather than bind-mounted.

---

## Part 5 — Add the reinforcement learning

Steps 5 and 6 fit a readout on expert actions, which is behaviour cloning. This
part is the actual RL: SAC learns the readout from reward.

### Step 10 — Add a `fly_donut` course *(sim)*

**Do this.** Copy `donut_course_no_hint.py` to a `fly_donut` course whose
`observation_spec` is the descending-neuron trace instead of the 31-D vector.
Register it in `robotaxi_env.py`, add its width to `COURSE_OBSERVATION_SIZES` in
`collect_training_data.py`, and call the brain service from the observation
path. Reset the brain whenever the episode resets.

**Why it matters.** Everything else in the trainer — SAC, replay, eval, the
leaderboard — then works unchanged, exactly as it did when `donut_camera`
introduced a new observation shape. Rewards keep using the full
`scene_data_array()`, so the reward design is untouched and results stay
comparable.

**You are done when.** A short job trains without shape errors and TensorBoard
shows a moving `avg_return`.

### Step 11 — Train it and compare honestly *(sim)*

**Do this.** Train `fly_donut` and compare against `donut_no_hint` on
`eval/goals_per_episode_this_eval` at an equal step budget.

**Why it matters.** Use the reward-invariant metric, not `avg_return` — we
already found that comparing returns across jobs with different reward designs
is meaningless. `donut_no_hint` reached 86.94 avg return and is the bar.

**You are done when.** You have goals-per-episode for both at the same budget.

### Step 12 — Run the scrambled-wiring control *(sim)*

**Do this.** Rebuild the brain with degree-preserving shuffled weights, keeping
neuron count, connection count and encoder identical. Train again at the same
budget.

**Why it matters.** Without this you cannot claim the fly connectome did
anything — a 166,700-unit random recurrent network is a perfectly good reservoir
on its own, and that's precisely the open question in fly.ai's own roadmap.
Their talking-flies control came out ambiguous: scrambled wiring carried nothing
at 20 ms and as much as the real brain at 2 ms.

**You are done when.** You have four numbers — `donut_no_hint` SAC, the step-5
ridge readout, the real connectome under SAC, and the scrambled connectome under
SAC — at one budget. That comparison is the actual result of this project.

---

## Things that will bite

**GPU contention.** TensorFlow grabs most of the card by default and the brain
wants ~210 MB plus working space. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` on the
trainer before the first joint run, or the brain service will fail to allocate.
Run fly experiments at `--num-envs 1` until you've measured throughput.

**Throughput.** Measured: K=5 substeps at 2.67 ms is **13.4 ms per control
step**, about 20 minutes of pure brain compute over an 87k-step run. That fits
a 10 Hz sim-time budget easily, but `Time.timeScale` (3 in `SimController`, 5 in
`BootStrap`) pushes the real control rate to 30–50 Hz, i.e. a 20–33 ms
wall-clock budget. 13.4 ms fits with less headroom than is comfortable, so
batch across actors (`FlyBrain(batch=N)`, ~1.2 ms per fly per step) before
raising `--num-envs`.

**Noise makes the brain non-deterministic.** The same rays give a different
trace twice. Seed per episode on reset so runs are reproducible, and see the
eval-variance note in step 6 before recording comparisons.

**Replay stores history-dependent observations.** The trace depends on the
brain's voltages, which depend on the whole episode so far, so off-policy replay
learns from features it can't exactly reconstruct. Workable — the trace *is* the
observation — but if training is unstable, suspect this first.

**fly.ai is explicit that this is a demo, not an emulation.** Point neurons, one
global parameter set, no dendrites, no neuromodulators, no plasticity,
transmitter sign from a rough rule, nothing validated against recordings from
real flies. Expect an interesting result, not a good driver.
