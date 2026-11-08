---
name: deploy-policy-to-jetracer
description: Gets a trained donut_no_hint policy driving the physical JetRacer by following the repo's 10-step deployment plan, with each step's check as a hard gate and the user's explicit go-ahead before anything that can move the car. Use when the user asks to deploy, run or test a policy on the real car or JetRacer.
disable-model-invocation: true
---

# Deploy a policy to the JetRacer

The plan is `docs/jetracer-deployment-plan.md`; read the step there before doing it. Contracts (bearing order, action map, topics) are in `docs/tf-model-jetson-deployment-guide.md` sections 3-5; car access and bringup in `docs/hardware/jetson.md`. This skill adds the order, the gates and the safety rules.

## Safety rules (never relax)

- **Ask before every step marked (car).** Say exactly what will happen ("wheels will spin for 3 seconds on the box"), then wait for a yes. Never chain physical steps.
- Never publish to `/cmd_vel` directly or start the policy node with `/policy/enable` on. The user, not the agent, turns the switch on, unless they explicitly ask the agent to for that one step.
- Keep `lidar_estop` running. Kill the joystick teleop (`/teleop_joy`, `/joy_node`) before the policy node; never run `explore_foxglove.launch` alongside it.
- **If the car ever steers toward an obstacle: stop at once.** Bearing order or steering sign is wrong. Go back to steps 5 and 8; never adjust scales to compensate.
- A failed check means stop and fix that step. Don't skip ahead.

## Workflow

Copy and tick off:

```
Part 1, desktop only
- [ ] 0. Pick the checkpoint
- [ ] 1. Export weights to .npz
- [ ] 2. numpy policy
- [ ] 3. Parity with TensorFlow (<= 1e-4)
Part 2, car on, never moving
- [ ] Health check
- [ ] 4. Record 30 s (car, stationary)
- [ ] 5. Lidar scan -> 31 inputs
- [ ] 6. Replay through the whole chain
- [ ] Gap check (sim-real-observation-gap skill)
Part 3, moving
- [ ] 7. Policy node behind a default-off switch (car)
- [ ] 8. Wheels off the ground (car on a box)
- [ ] 9. Emergency stop wins (car on a box)
- [ ] 10. Ten seconds on the floor (open floor)
```

**0. Pick the checkpoint.** Use the roboracer-console MCP tools: `list_models` with `course_type: "donut_no_hint"` and `sort: "best_score"`, then `get_model` for its evals. It must be `donut_no_hint` (31 inputs; refuse any 32-input `donut` model) and a greedy SavedModel, and it should have evals: suggest `eval_models` if not. The plan's default is `/saved_models/robotaxi/SacAgent/7573_step_87314`.

**Code location.** Put deployment code in `deploy/jetracer/`, one module per job, so the gap skill and the car share it:
- `export_npz.py` (step 1), `policy_np.py` (2), `test_parity.py` (3)
- `scan_to_obs.py` (5): the single scan-to-31-inputs function; must also run on the car's Python 2
- `replay_bag.py` (6): bag -> printed actions and a `real_obs.csv` for the gap skill
- `policy_node.py` (7)

**Steps 1-6:** do what the plan says; its "You are done when" line is the gate. Extra checks:
- Step 3: if parity fails, try the other half of the four outputs as the plan says, and record which pair is right in `policy_np.py`.
- Step 5: copy the bearing order from the guide's list exactly; unit-test a wall 1 m dead ahead (forward input near 12 with the x12 crutch) and a wall to the left (left-side inputs short, right long).
- Step 6: also write `real_obs.csv`, then run the `sim-real-observation-gap` skill. Any Stop-row finding blocks Part 3.

**Health check before Part 2:** `ssh jetson`, start with `~/start_jetracer.sh`, confirm `/scan` and `/odom_raw` publish (`rostopic hz`), `tf_echo base_footprint laser_frame` shows yaw near 0, and `invert_linear=true`. Report go/no-go.

**Steps 7-10:** one at a time, each after an explicit yes. Before step 8, give the user the exact commands to turn the switch on and off. After each step, write down what the car did. Step 8 failures: forward reversed, so check `invert_linear` (never negate in the policy); steering mirrored, so check the steering sign.

## After step 10

Stop on purpose, as the plan says. Write up what the car did in its ten seconds and map it to the plan's "Stop here on purpose" list:
- crawls timidly: replace the x12 crutch with proper normalisation
- drives into dark furniture: missing lidar returns; a validity mask
- roughly reasonable: calibrate the action map (guide Phase F)

Then suggest re-running the gap skill on a recording from the floor run.
