# Getting the existing policy to drive the JetRacer

Ten steps in a fixed order. No choices to make, nothing running in parallel.
Do one, confirm it worked, then move on to the next.

This gets the checkpoint you **already have** driving the car, using no
training whatsoever. Improving *how* the car drives — retraining with
corrected scale, simulated lidar dropouts, a better action space — is
deliberately left out. Those choices get much easier once the car has moved
once, and you will have real data instead of predictions.

Sibling docs:

- SavedModel → Melodic `/cmd_vel` background and the sim/real contract gap:
  [`tf-model-jetson-deployment-guide.md`](tf-model-jetson-deployment-guide.md).
- On-car bringup, topics, and `lidar_estop`:
  [`jetson.md`](../jetson.md).
- Why the camera courses are not the starting point:
  [`csi-camera-observation-guide.md`](csi-camera-observation-guide.md).

The policy being deployed is `donut_no_hint`, checkpoint
`/saved_models/robotaxi/SacAgent/7573_step_87314`: a 31-D observation
(speed, sideslip, 29 rays) and two dense 512 layers. Every input it takes has
a real-world analogue, which is why it and not `SimpleCourse` is the target.

---

## Part 1 — Turn the policy into plain arithmetic

Your desktop only. No car, no Jetson, nothing that can move or break.

### Step 1 — Copy the policy's numbers into a plain file *(desktop)*

**Do this.** Open `/saved_models/robotaxi/SacAgent/7573_step_87314` with
`tf.train.load_checkpoint` and save the six arrays it contains into a single
`.npz` file.

**Why it matters.** A trained neural network sounds complicated, but yours is
six tables of numbers totalling 1.12 MB. Once they sit in an ordinary file,
nothing downstream needs TensorFlow — which is the thing that would otherwise
be painful to install on the Nano.

**You are done when.** The `.npz` holds six arrays with shapes `(31,512)`,
`(512,)`, `(512,512)`, `(512,)`, `(512,4)` and `(4,)`. Those have already been
confirmed as what the checkpoint contains.

### Step 2 — Write the twenty lines that run the policy *(desktop)*

**Do this.** Write a function that takes 31 numbers in and gives 2 numbers out:
multiply by the first table, zero out negatives, multiply by the second table,
zero out negatives, multiply by the third table, then keep the first two
results and squash them into the steering and throttle ranges.

**Why it matters.** This is the entire policy. Three multiplications. Writing
it yourself means you can read it, debug it, and run it anywhere Python and
numpy exist — including inside the car's Python 2 ROS code.

**You are done when.** You can feed it 31 made-up numbers and get back two
numbers, one throttle between 0.05 and 1, one steering between -1 and 1.

### Step 3 — Prove your version matches the real one *(desktop)*

**Do this.** Load the original policy with TensorFlow, feed the same 31 numbers
to both it and your twenty-line version, and compare the two answers.

**Why it matters.** If these two ever disagree, every later test is
meaningless — you would be debugging the car when the bug is in your
arithmetic. It also settles one open question: the policy produces four
numbers and the first two are believed to be the ones you want, but this test
proves it rather than assuming.

**You are done when.** Both give the same answer to within 1e-4. If they
differ, the four outputs are ordered differently than assumed — try the last
two instead, and re-test.

---

## Part 2 — Feed it real sensor data

The car is switched on but never moves. Still all analysis, no driving.

### Step 4 — Record thirty seconds of real sensor data *(car, stationary)*

**Do this.** With the car sitting still in the room you plan to drive in, run
`rosbag record` on `/scan`, `/odom_raw`, `/imu` and `/tf` for thirty seconds.

**Why it matters.** You need real data to develop against, and a recording is
repeatable — you can replay the same thirty seconds a hundred times while
fixing code, instead of re-running the car.

**You are done when.** You have a bag file, and `rosbag info` lists `/scan` and
`/odom_raw` inside it.

### Step 5 — Write the translator from lidar to policy input *(desktop)*

**Do this.** Write a function that takes one lidar scan and returns 29
distances at the specific angles the policy expects. The lidar gives you 720
measurements around a full circle; the policy wants 29 in front. Then multiply
every distance by 12.

**Why it matters.** Two separate things. First, somebody has to translate
between what the sensor gives and what the policy expects, and the angle list
is unusual — it is not in order, so it must be copied exactly from the
documented list. Second, the simulated car is roughly twelve times larger than
your real car, so real-world distances must be scaled up to resemble what the
policy was trained on. That multiply-by-12 is a temporary crutch to let us
test; it gets replaced properly later.

**You are done when.** Feeding it a made-up scan with a wall one metre
straight ahead gives a forward distance near 12 (one metre, scaled). Put a real
box a metre in front of the car and confirm the same.

### Step 6 — Replay the recording through the whole chain *(desktop)*

**Do this.** Play the bag from step 4 through the translator from step 5 and
the policy from step 2, and print the throttle and steering it would have
commanded. Nothing is connected to the motors.

**Why it matters.** This is the first time you see the policy react to your
actual room. It is completely safe, and it catches the embarrassing problems —
all-zero inputs, distances in the wrong units, steering pinned to one side —
while the car cannot move.

**You are done when.** The printed commands look reasonable: throttle modestly
positive, steering small and varying rather than stuck at an extreme.

---

## Part 3 — Let it move

Escalating carefully. Do not skip ahead, and keep a hand on the stop.

### Step 7 — Publish commands, but behind an off switch *(car)*

**Do this.** Write the ROS node that runs steps 5 and 2 in a loop at 10 Hz and
publishes to `/policy/cmd_vel`. Only forward it to the real `/cmd_vel` while a
separate `/policy/enable` topic says yes. Default it to off. Leave
`lidar_estop` in place.

**Why it matters.** The car has exactly one steering-and-throttle input, and
several things want to write to it. An explicit, default-off switch means the
policy can never surprise you, and you can stop it without killing processes.

**You are done when.** The node runs and publishes to `/policy/cmd_vel`, and
the wheels stay still because the switch is off. Also confirm the joystick
teleop node is not running — bringup starts it by default and it fights for the
same topic.

### Step 8 — Wheels off the ground *(car on a box)*

**Do this.** Put the car on a box so the wheels spin freely. Turn the switch on
for about three seconds, then off.

**Why it matters.** The first time the policy drives the motors, you want the
car unable to go anywhere. This is where you find out that forward is backward,
or that steering is mirrored.

**You are done when.** Wheels turn forward, not backward, and steering moves in
the direction the picture suggests. If forward is reversed, check
`invert_linear` rather than negating anything in the policy.

### Step 9 — Check the emergency stop still wins *(car on a box)*

**Do this.** With the switch on and the wheels spinning, hold a board about
15 cm in front of the lidar.

**Why it matters.** The policy has no concept of an emergency stop.
`lidar_estop` is the safety layer, and you need to know it overrides the policy
before the car is on the floor — not after.

**You are done when.** The wheels stop, even though the policy is still
commanding throttle.

### Step 10 — Ten seconds on the floor *(open floor)*

**Do this.** Clear at least three metres ahead. One person holds the stop.
Switch on for ten seconds, then off regardless of what happens.

**Why it matters.** This is the answer to the real question: does any of this
work at all? Ten seconds is long enough to learn a great deal and short enough
that nothing gets broken.

**You are done when.** The car moves forward along the open space without
spinning, without lunging at a wall, and without reversing. Anything else is
information, not failure — write down exactly what it did.

---

## Stop here on purpose

Do not plan past step 10 yet. How the car behaves in those ten seconds decides
what comes next:

- If it crawls timidly, the 12x scale crutch needs replacing with proper
  normalisation (rays by wheelbase, speed by `sqrt(g * wheelbase)`).
- If it drives into dark furniture, the lidar's missing returns are the
  priority — the real RPLIDAR A1 drops 41.9% of beams, and encoding those as
  max range tells the policy "clear road". A validity mask is the fix.
- If it drives roughly reasonably, the action mapping is worth calibrating
  before anything else.

## One thing to watch throughout

If the car ever steers **towards** the nearest obstacle rather than away, stop
immediately and do not adjust any scaling numbers. That symptom means the 29
angles are in the wrong order or the steering sign is flipped, and tuning will
only hide it. Recheck the angle list from step 5 and the steering direction
from step 8.
