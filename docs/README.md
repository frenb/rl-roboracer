# Documentation

Start with the [project README](../README.md). Guides describe what exists; design notes explain decisions in built systems; plans describe work that is partly or not yet built, and each says its status at the top.

## Guides

| Doc | What it covers |
|---|---|
| [setup.md](setup.md) | Manual development setup: host software, Docker images, building a Unity gym |
| [operations.md](operations.md) | Starting, stopping and restarting the stack; multi-actor runs; the service map |
| [training.md](training.md) | Greedy evaluation, AWAC, the curriculum, stuck detection, the demonstration corpus, pause and resume |
| [sim2real.md](sim2real.md) | How the simulator mirrors the JetRacer, and the hint-free `donut_no_hint` observation |
| [dashboard.md](dashboard.md) | Dashboard architecture and known limitations |
| [madscientist.md](madscientist.md) | The autonomous experiment orchestrator: researcher, judge, approval, outcomes |
| [fly-brain.md](fly-brain.md) | Driving the car with a frozen fruit fly connectome |
| [fly-brain-driver-how-it-works.md](fly-brain-driver-how-it-works.md) | One control step of the fly-brain driver, end to end |
| [hardware/jetson.md](hardware/jetson.md) | The physical car: Jetson bringup, ROS topics, SLAM, gotchas |
| [rltest-clean-test.md](rltest-clean-test.md) | Testing the installer in a clean Windows account |

## Design notes (built)

| Doc | What it covers |
|---|---|
| [double-critic-implementation.md](double-critic-implementation.md) | Where SAC's twin critics come from in tf-agents |
| [trajectory-rollout-viz.md](trajectory-rollout-viz.md) | Drawing the policy's candidate paths in Unity |
| [camera-demo-tfrecord-plan.md](camera-demo-tfrecord-plan.md) | Camera frames in expert demonstrations (BC and AWAC) |
| [experiment-designs-tab-plan.md](experiment-designs-tab-plan.md) | Making the Experiments tab express every trainer field |
| [fly-brain-job-lifecycle.md](fly-brain-job-lifecycle.md) | Every step from encoder idea to a trained fly-brain policy |
| [flybrain-driver-plan.md](flybrain-driver-plan.md) | The step-by-step plan the fly-brain driver was built from |

## Plans (partly or not yet built)

| Doc | What it covers |
|---|---|
| [jetracer-deployment-plan.md](jetracer-deployment-plan.md) | Ten steps to get a trained policy driving the real car |
| [tf-model-jetson-deployment-guide.md](tf-model-jetson-deployment-guide.md) | The sim/real input and action contracts, testing, and sim changes for transfer |
| [csi-camera-observation-guide.md](csi-camera-observation-guide.md) | Putting the CSI camera on the learning loop |
| [real2sim-fence-colliders-plan.md](real2sim-fence-colliders-plan.md) | Collidable fences on the real2sim track model |
| [madscientist-autonomous-codegen.md](madscientist-autonomous-codegen.md) | Letting the Mad Scientist write the code its experiments need |
| [proposal-spawn-multiprocessing.md](proposal-spawn-multiprocessing.md) | Switching parallel environment workers from fork to spawn |
| [shared-mongodb-plan.md](shared-mongodb-plan.md) | Sharing the database with a second developer |
| [install-script-plan.md](install-script-plan.md) | The Windows one-script installer and how to test it |
| [install-script-plan-macos.md](install-script-plan-macos.md) | A macOS installer |

## Drafts

| Doc | What it covers |
|---|---|
| [drafts/rl-roboracer-preprint.md](drafts/rl-roboracer-preprint.md) | Working draft: reward design and demonstration-guided RL ablations |

## Media

[`images/`](images/) holds the diagrams used here; [`video/`](video/) holds the scripts and slides for the platform and fly-brain videos.
