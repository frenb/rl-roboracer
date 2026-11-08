---
name: sim-real-observation-gap
description: Measures how far the JetRacer's real sensor inputs (speed, sideslip, 29 lidar rays) are from what the donut_no_hint policy saw in simulation, ranks the gaps, and says which to fix in the car-side bridge and which need a sim change and retraining. Use when checking sim-to-real transfer, before or after a real-car test, when the real car behaves differently from sim, or when the user mentions observation gap, domain gap or sim2real.
---

# Sim-to-real observation gap

Compares real 31-input vectors against the simulated steps a job logged, using `scripts/compare_obs.py`. Background: `docs/tf-model-jetson-deployment-guide.md` (section 3.1 input contract, section 6 sim changes).

## Workflow

```
- [ ] 1. Get real vectors
- [ ] 2. Pick the simulated baseline
- [ ] 3. Run the comparison
- [ ] 4. Diagnose the gaps
- [ ] 5. Report
```

**1. Get real vectors.** The script needs an N x 31 CSV (or .npy) of vectors exactly as the car-side bridge would feed the policy: after any clip, fill or scale, in policy order (speed, sideslip, then the 29 rays from -90 round to +90 as in the guide's bearing list).
- If `deploy/jetracer/` has the bridge and replay script from the `deploy-policy-to-jetracer` skill, replay a recording through it and write the vectors to CSV.
- No bridge yet: stop and offer the `deploy-policy-to-jetracer` skill (steps 4-6). Don't write a second scan-to-rays translator here; two copies drift apart.
- The car must have moved during the recording, or speed and sideslip are all zero and tell you nothing. Ask the user to push it by hand or drive it slowly with the joystick. Ask before connecting to the car at all.
- Record where the policy will drive. A living room is not a donut track: walls 1 m away will read as a gap even with a perfect bridge. Best is the physical track mat, compared against its real2sim gym.

**2. Pick the simulated baseline.** Use evals of the model being deployed: `get_model` for its location, then `list_jobs` with `job_type: "EVAL"` and pick one with that `location`. Eval steps come from the greedy policy, which is what the car will run. Otherwise use the training job (`job_id` on the model). Prefer a job on the same track as the recording.

**3. Run the comparison** (needs `uv` and the stack's MongoDB on `127.0.0.1:27017`):

```bash
uv run .cursor/skills/sim-real-observation-gap/scripts/compare_obs.py --job <job id> --real real_obs.csv --json gap.json
```

Columns: `med r/s` real/sim median ratio; `outside` share of real values beyond the sim 1st-99th percentile; `nonfin` missing; `pinned` share stuck at the real maximum; `KS` distribution shift (0 same, 1 disjoint). Rows are worst first.

**4. Diagnose.** Match patterns to causes:

| Pattern | Likely cause | Fix |
|---|---|---|
| Rays on one side short, the other long, in a symmetric space; or the car would steer toward obstacles | Bearing order wrong or lidar yaw not 0 | **Stop.** Recheck the bearing list and `tf_echo base_footprint laser_frame`. Never tune scales to hide this. |
| `med r/s` roughly the same constant (not 1) across all rays | Scale error: the x12 crutch is wrong for this space | Bridge: proper normalisation (rays by wheelbase). Long-term: true-scale sim (guide 6.2.8). |
| Front rays `pinned` high | Real lidar caps near 12 m; sim rays reach 50 m (100 m forward) | Bridge: document the clip. Sim: cap rays at 12 m and retrain (6.1.2). |
| `nonfin` high | Lidar dropouts (the deployment plan measured 41.9% of RPLIDAR A1 beams missing) | Bridge: fill with the clip value, never max range ("clear road"). Sim: dropouts and noise (6.1.4-5), or a validity mask. |
| Near rays have a real p1 near 0 on every step | Chassis or mast hits (blind zone) | Sim: model the blind zone (6.1.6). Bridge: mask those bearings. |
| Speed ratio off, or negative while moving forward | Odometry units or `invert_linear` | Bridge: use `/odom_raw` `twist.linear.x`; check `invert_linear=true`. |
| Sideslip shifted or mirrored | Yaw sign or degrees/radians | Bridge: atan2 of velocity vs heading, divided by 180, sign as in sim (6.2.12). |
| Gaps everywhere but small KS (under 0.2) | Normal real-world noise | Note it; not a priority. |

Before blaming the bridge, check the space matches the track (step 1). Rays shorter everywhere in a cluttered room is the room, not a bug.

**5. Report.** Lead with one sentence: is the gap small enough to try the policy, and if not the single biggest problem. Then:

```markdown
**Data:** real steps and where recorded; sim baseline job and steps.
**Gaps, worst first:** for each (group several rays into one finding): what differs, numbers (median ratio, missing, pinned, KS), likely cause.
**Fix in the bridge (no retraining):** ...
**Fix in the sim (needs a gym change and retraining):** with the guide section 6 item.
**Blockers:** anything from the Stop row; the car must not drive until fixed.
```
