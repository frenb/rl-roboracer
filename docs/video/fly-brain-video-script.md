# Fly brain video script

Nine slides, roughly 20–30 seconds of narration each, about three and a half minutes in total. Each slide is 1920×1080: the on-screen text sits in the left third, the image in the right two thirds.

Every slide comes in two forms in [`video/fly-brain/`](video/fly-brain/): `slide-NN-*` is the finished frame with the text laid out, and `image-NN-*` is the right-hand 1280×1080 image on its own, for laying out the text yourself. Both are SVG and PNG.

On the six step slides the current panel sits in the middle, with its neighbours faded at the edges, so a slow pan from one slide to the next reads as moving along the loop.

## Slide 00: A real fly's brain, driving a race car

![A real fly's brain, driving a race car](video/fly-brain/slide-00-overview.png)

**On screen**

- Label: THE FLY BRAIN DRIVER
- Headline: A real fly's brain, driving a race car
- Every 0.1 s, the car's sensors are translated into the signals a fly's eyes would send.
- A frozen map of 166,700 real neurons processes them.
- Only a small readout at the end learns to turn brain activity into driving.

**Narration**

> What happens if you let a real fly's brain drive a race car? In this project, a complete map of a fruit fly's brain, 166,700 neurons wired exactly as they are in the fly, sits in the control loop of a simulated car. Here is one trip around that loop. It happens ten times every second.

*56 words, about 22 s.*

## Slide 01: The car reports what it sees

![The car reports what it sees](video/fly-brain/slide-01-unity.png)

**On screen**

- Label: STEP 1 OF 6 · UNITY
- Headline: The car reports what it sees
- The Unity gym sends 31 numbers every tenth of a second.
- Speed, sideslip, and 29 distance rays fanned out ahead.
- The same view our conventional driving agents learn from.

**Narration**

> It starts in Unity. Every tenth of a second the simulated car reports what it sees: its speed, how much it is sliding sideways, and 29 distance rays fanned out in front of it. That is the same 31 numbers our conventional driving agents learn from.

*46 words, about 18 s.*

## Slide 02: Translating rays into fly vision

![Translating rays into fly vision](video/fly-brain/slide-02-encoder.png)

**On screen**

- Label: STEP 2 OF 6 · ENCODER
- Headline: Translating rays into fly vision
- Looming: how fast a wall is closing in.
- Chase: which side is more open.
- New: optic flow, how fast the walls stream past each eye. It gives the brain its speed and its place in the lane.

**Narration**

> A fly's brain does not understand distance rays, so an encoder translates them into the signals a fly's eyes produce. Looming: a wall is rushing toward me. Chase: there is open space over there. And new in the latest version, optic flow: how fast the walls stream past each eye. That tells the fly how fast it is going and which wall is closer, the same cue bees use to fly down the middle of a corridor.

*77 words, about 31 s.*

## Slide 03: A frozen fly brain in the loop

![A frozen fly brain in the loop](video/fly-brain/slide-03-connectome.png)

**On screen**

- Label: STEP 3 OF 6 · CONNECTOME
- Headline: A frozen fly brain in the loop
- The MaleCNS v1.0 connectome: 166,700 neurons and 25.6 million connections.
- The cues stimulate the fly's own looming, chase and motion detectors.
- Nothing inside it learns. The wiring is exactly as mapped.

**Narration**

> Those cues are injected into the brain's own visual detectors: the looming cells, the chase cells, and the motion detectors. From there the signal spreads through the full connectome, 25.6 million connections, simulated on the GPU. And here is the key point: this brain is frozen. Nothing in it learns. The wiring is exactly as it was mapped from a real fly.

*62 words, about 25 s.*

## Slide 04: Listening to the brain's outputs

![Listening to the brain's outputs](video/fly-brain/slide-04-trace.png)

**On screen**

- Label: STEP 4 OF 6 · DESCENDING TRACE
- Headline: Listening to the brain's outputs
- 1,314 descending neurons carry the brain's commands to the body.
- Each spike leaves a fading trace, turning sparse spikes into a smooth signal.
- These 1,314 numbers are all the driver ever sees.

**Narration**

> At the other end, 1,314 descending neurons, the cells that carry commands from a fly's brain to its body, start to fire. Each spike leaves a fading trace, which turns sparse, noisy spikes into a smooth signal. Those 1,314 numbers are all the driver ever gets to see.

*48 words, about 19 s.*

## Slide 05: The only part that learns

![The only part that learns](video/fly-brain/slide-05-readout.png)

**On screen**

- Label: STEP 5 OF 6 · READOUT
- Headline: The only part that learns
- Copying an expert steers well, but drives far too slowly.
- Reinforcement learning learns from reward; AWAC + BC also learns from demos.
- With optic flow,* SAC's return rose from 16.8 to 40.8. AWAC + BC reaches 58.2.
- Footnote: * Optic flow: two extra cues, flow_L and flow_R, measure how fast the walls stream past each side of the car and drive the T4a + T5a motion detectors. They carry the car's speed and lane position. Adding them raised the ridge readout's held-out R² from 0.62 to 0.67 for steering and 0.18 to 0.25 for acceleration, and SAC's return from 16.8 to 40.8. Returns are means of fresh 10-trial evals, not training-time bests.

**Narration**

> So the only thing that learns is the readout, a small network that turns brain activity into driving. Copying an expert steers well but drives far too timidly. Reinforcement learning learns from reward instead. Once the brain could sense its own speed through optic flow, its score more than doubled, from 17 to 41, and adding expert demonstrations with AWAC plus behaviour cloning took it to 58.

*65 words, about 26 s.*

## Slide 06: Back to the car

![Back to the car](video/fly-brain/slide-06-action.png)

**On screen**

- Label: STEP 6 OF 6 · ACTION
- Headline: Back to the car
- Two numbers come out: acceleration and steering.
- Unity applies them, the car moves, and the loop repeats ten times a second.
- In Unity every neuron lights up as it fires, so you can watch the fly brain drive.

**Narration**

> The readout produces just two numbers, acceleration and steering. Unity applies them, the car moves, the rays change, and the loop starts again. In Unity you can watch it happen: every neuron lights up as it fires, so you can see the fly's brain react as the car takes a corner.

*51 words, about 20 s.*

## Slide 07: One step through the fly brain

![One step through the fly brain](video/fly-brain/slide-07-control-loop.png)

**On screen**

- Label: UNDER THE HOOD · CONTROL LOOP
- Headline: One step through the fly brain
- The trainer turns the scene into six cues and sends them to the fly-brain service.
- The brain runs 100 ms of spiking on the GPU and returns its 1,314-number trace.*
- The readout picks an action,† and every step becomes training data for AWAC + BC.
- Footer: Numbered steps match the six-step walkthrough.
- Footnote: † Readout: the policy network, 1,314 trace values in, two 512-unit layers, two numbers out: [accel, steer]. "Picks an action" means one forward pass per step: in training it samples near its best guess to explore, in evaluation it takes the best guess. AWAC + BC is what trains it.
- Footnote: * Trace: one number per descending neuron, the 1,314 cells that carry the brain's commands to the body. Each spike adds 1 and the value fades with a 0.1 s time constant, so it reads as recent firing rate: 0 when silent, about 5.5 when firing every 20 ms.

**Narration**

> Here is the same loop as the services actually run it. Unity streams the scene through its ROS server to the trainer. The trainer encodes six cues and calls the fly-brain service, which spikes 166,700 neurons for 100 milliseconds on the GPU and hands back the trace. The readout picks an action, it goes back to Unity, and the whole step lands in the replay buffer for AWAC to learn from.

*69 words, about 27 s.*

## Slide 08: How well does it drive?

![How well does it drive?](video/fly-brain/slide-08-results.png)

**On screen**

- Label: RESULTS
- Headline: How well does it drive?
- Random actions score about 1. Copying an expert through the brain, about 7.
- Learning from reward with optic flow reaches 40.8.
- Adding expert demos with AWAC + BC lifts it to 58.2.
- That is 77% of a conventional agent that sees the raw sensors directly.
- Footer: Next: the same pipeline on a real JetRacer robot car.

**Narration**

> So how well does it drive? Random actions score about one. Copying an expert through the brain, about seven. Learning from reward with optic flow reaches about 41, and adding expert demonstrations on top, with AWAC plus behaviour cloning, lifts it to 58. That is 77 percent of a conventional agent that sees the raw sensors directly, even though every decision has to pass through a frozen fly brain first. Next stop: a real robot car.

*73 words, about 29 s.*

### Alternate versions: goals and speed

Same five models and the same 10-trial evals, measured two other ways. Use either in place of slide 08, or after it. Random actions are left out because their evals did not record goals or speed.

![How far does it get?](video/fly-brain/slide-08-results-goals.png)

- Label: RESULTS · GOALS
- Headline: How far does it get?
- Goals per episode: how many checkpoints the car reaches, out of 100.
- Copying an expert reaches about 10. SAC with optic flow reaches 53.
- AWAC + BC reaches 65, three quarters of the no-brain agent's 88.

> Another way to score it is how far the car gets: how many of the track's checkpoints it reaches before the episode ends, out of a hundred. Copying an expert gets about ten. SAC with optic flow gets 53. AWAC plus behaviour cloning gets 65, three quarters of what the no-brain agent manages.

![How fast does it drive?](video/fly-brain/slide-08-results-speed.png)

- Label: RESULTS · SPEED
- Headline: How fast does it drive?
- SAC with optic flow survives by crawling at 1.6 m/s, no faster than the copied expert.
- AWAC + BC drives at 3.7 m/s, more than twice as fast.
- That is 69% of the no-brain agent's 5.4 m/s.

> Speed tells a different story. SAC with optic flow stays on the track by crawling, at 1.6 metres a second, no faster than simply copying the expert. With expert demonstrations, AWAC plus behaviour cloning drives at 3.7, more than twice as fast, and about 70 percent of the no-brain agent's pace.

## Where the numbers come from

| Figure | Source |
|---|---|
| 166,700 neurons, 25.6 M connections, 5 × 20 ms substeps | The fly-brain service, `docker/fly_brain/`; see [fly-brain-driver-how-it-works.md](fly-brain-driver-how-it-works.md) |
| Six cues, optic flow into T4a + T5a | `FlowEncoder` in `rl_agent/fly_brain/encoder.py` (the `fly_donut_flow` course) |
| Ridge R² steering 0.67, acceleration 0.25 (was 0.62 / 0.18) | Step-5 replay of 100 expert episodes, `FlyDonutFlowCourse` docstring |
| Slide 05 returns: SAC 16.8 → 40.8 with optic flow, AWAC + BC 58.2 | The same 10-trial evals as slide 08. The earlier 36.9 / 60.7 were training-time bests and were replaced on 2026-09-28. |
| Slide 07: gRPC Step with 5 substeps and eye drive 0.45, overlay at 10 Hz on a background thread | `protos/fly_brain/proto/fly_brain.proto`, `FlyDonutCourse.policy_vector` in `rl_agent/environments/courses/fly_donut_course.py`, `rl_agent/fly_brain/viz.py` |
| Slide 08: AWAC + BC 58.2, SAC 40.8 (fly_donut_flow), SAC 16.8 (fly_donut) | Fresh 10-trial greedy evals, read from the trainer log 2026-09-28. AWAC + BC is `SacAgent/7657_step_26502` from TRAIN job `6ab861dfe8ad86fce0df9564`, EVAL job `6ab96623a249d535d4d976b4` (trials 24.9–77.3; best in-training eval 84.7). SAC flow is EVAL job `6ab83af3a249d535d4d976b3`. fly_donut is the midpoint of its two evals, 15.6–18.0. |
| Slide 08 goals and speed: copied expert 10.4 / 1.6 m/s, SAC fly_donut 20.0 / 2.7, SAC fly_donut_flow 53.1 / 1.6, AWAC + BC 64.8 / 3.7, no-brain SAC 88.1 / 5.4 | `db.leaderboard_scores` per-trial `avg_goals_per_episode` and `avg_speeds`, read 2026-09-28. Entries: `SacAgent/7657_step_26502` 2026-09-27, `SacAgent/7653_step_90967` 2026-09-26, `SacAgent/7573_step_87314` 2026-09-07 (the 75.7 eval). Copied expert averages the two 10-trial `FlyPyPolicy` evals of 2026-09-20/21; fly_donut averages the two 10-trial `SacAgent/7639_step_49163` evals of 2026-09-22/23. |
| FlyPyPolicy 6.87, no-brain SAC 75.7, random ~1 | [fly-brain-driver-how-it-works.md](fly-brain-driver-how-it-works.md), figures from 2026-09-21 |

The optic-flow cells (T4a + T5a) are drawn in blue on these slides. The Unity overlay does not give them their own colour yet.
