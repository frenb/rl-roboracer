#!/usr/bin/env python
"""Analyze a DEMO-collection job's tfrecords: speed, episodes, goals, crashes.

DEMO jobs (collect_expert_demos) do NOT write TensorBoard/course metrics and
their collection stdout is transient (it lives in /tmp/trainer.log, which gets
overwritten by the next trainer launch). This tool recovers per-run driving
statistics after the fact, straight from the stored tfrecords.

What is MEASURED vs. RECONSTRUCTED
----------------------------------
* SPEED is measured directly. On a flat donut record, observation[1] is
  the signed car speed in m/s (DonutCourse: obs[0]=angle-to-goal,
  obs[1]=speed, obs[17]=forward raycast clearance). On a camera record
  the stored vector is the no-hint policy vector, so speed is vector[0]
  for both donut_camera (31) and donut_camera_no_rays (2). The forward
  ray is vector[16] on the 31-wide file and is absent on the 2-wide file.
* EPISODES / GOALS-PER-EPISODE / CRASHES are RECONSTRUCTED from the per-step
  sequence, because the tfrecords store single-step transitions with synthetic
  step_types (no episode boundaries) - see
  collect_training_data.read_files_from_directory. Rows are in temporal order
  within each stage file, so:
    - an episode boundary is detected as a respawn: speed dips below
      SPEED_STOP after having driven above SPEED_MOVING (with min_force>=0.2 the
      demo car only sits at ~0 m/s on a fresh spawn);
    - goals/episode is inferred from the per-stage goal budget divided by the
      reconstructed episode count (the collector advances a stage after
      GOAL_BUDGET goals, and each episode is truncated at GOAL_CAP goals, so a
      crash-free stage lands on exactly ceil(GOAL_BUDGET/GOAL_CAP) episodes);
    - a crash-ended episode is flagged when the forward clearance at the
      episode's last step is in the bottom decile (car against a wall), as
      opposed to a clean cap/budget cutoff on open road.
  Treat the reconstructed figures as estimates, not ground truth.

Filenames are `<stage>_g<gym>_<subbatch>trajectories.tfrecord`, so the leading
zero-padded number is the curriculum stage (see
robotaxi._flush_trajectories_to_disk).

USAGE (must run inside the sim-controller container so collect_training_data
and the tfrecords volume are importable/visible):

    docker compose -f docker-compose.yml -f compose/scale.yml \
        exec -T sim-controller bash -lc \
        "cd /python_ws/src && PYTHONPATH=/python_ws/src \
         python -u analyze_demo_metrics.py <JOB_ID_OR_DIR>"

Examples:
    python analyze_demo_metrics.py 6a63d44dc9d3ac4a1af57a0a
    python analyze_demo_metrics.py /tfrecords/job_6a63d44dc9d3ac4a1af57a0a
    python analyze_demo_metrics.py 6a63d44d --goal-budget 100 --goal-cap 30
"""
import argparse
import glob
import os
import shutil
import tempfile

import numpy as np

import collect_training_data

# Observation-vector indices (DonutCourse.observation_spec). These shift down
# by _OBS_DROP_LEADING when analyzing 'donut_no_hint' demos (ROBOTAXI_
# OBSERVATION_SIZE=31), whose observation drops the leading dist_from_traj
# element - keep them derived from the same constant the parse schema uses so
# a mismatched env var can't silently make this read the wrong columns.
_OBS_DROP_LEADING = getattr(collect_training_data, "_OBS_DROP_LEADING", 0)
SPEED_IDX = 1 - _OBS_DROP_LEADING   # signed car speed, m/s, flat donut vector
FWD_IDX = 17 - _OBS_DROP_LEADING    # forward raycast clearance, flat donut vector
# Camera files store the no-hint policy vector. The leading angle-to-goal
# column is already gone, so speed is [0] and the forward ray (flat index
# 17) sits at [16]. The 2-wide ablation keeps only speed and sideslip.
CAMERA_SPEED_IDX = 0
CAMERA_FWD_IDX = 16


def _pct(a, p):
    return float(np.percentile(a, p))


def _resolve_dir(job_or_dir):
    """Accept a bare job id, `job_<id>`, or a full /tfrecords path."""
    if os.path.isdir(job_or_dir):
        return job_or_dir
    name = job_or_dir if job_or_dir.startswith("job_") else f"job_{job_or_dir}"
    return os.path.join("/tfrecords", name)


def _read_obs_act(directory):
    traj = collect_training_data.read_files_from_directory(directory)
    obs, act = traj.observation, traj.action
    obs = obs.numpy() if hasattr(obs, "numpy") else np.asarray(obs)
    act = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
    return obs, act


def record_layout(path):
    """Column indexes for one tfrecord, from whether it carries an image.

    A camera file (``image`` feature present) uses the policy vector:
    speed at 0, forward ray at 16 when the vector is wide enough to hold
    it. A flat file keeps SPEED_IDX / FWD_IDX.
    """
    example = collect_training_data._peek_example(path)
    if example is None:
        raise SystemExit(f"empty tfrecord: {path}")
    image_len = collect_training_data._image_byte_len(example)
    if image_len is None:
        return {"kind": "flat", "course": None,
                "speed_idx": SPEED_IDX, "fwd_idx": FWD_IDX}
    width = collect_training_data._float_len(example, "observation")
    course = None
    for name, want in collect_training_data.CAMERA_VECTOR_WIDTHS.items():
        if want == width:
            course = name
            break
    if course is None:
        known = sorted(collect_training_data.CAMERA_VECTOR_WIDTHS.values())
        raise SystemExit(
            f"{path} has an image but observation length {width} "
            f"is not a camera width {known}")
    fwd_idx = CAMERA_FWD_IDX if width > CAMERA_FWD_IDX else None
    return {"kind": "camera", "course": course,
            "speed_idx": CAMERA_SPEED_IDX, "fwd_idx": fwd_idx}


def _read_vectors(directory, layout):
    """Observation matrix for ``directory``. Camera rows keep the vector only."""
    if layout["kind"] == "flat":
        return _read_obs_act(directory)[0]
    rows = []
    for row in collect_training_data.camera_demo_dataset(
            directory, layout["course"], min_accel=None):
        rows.append(row["observation"]["vector"].numpy())
    if not rows:
        raise SystemExit(f"no rows in {directory}")
    return np.stack(rows)


def _segment_episodes(speed, speed_moving, speed_stop):
    """Split a temporally-ordered speed trace into episodes at respawns."""
    starts = [0]
    armed = False  # True once we've seen real driving speed this episode
    for i in range(1, len(speed)):
        if speed[i] > speed_moving:
            armed = True
        if armed and speed[i] < speed_stop and speed[i - 1] >= speed_stop:
            starts.append(i)
            armed = False
    bounds = starts + [len(speed)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def analyze(name, obs, args, speed_idx=SPEED_IDX, fwd_idx=FWD_IDX):
    speed = obs[:, speed_idx]
    n = len(speed)
    print(f"\n===== {name}  ({n} steps) =====")
    print(f"  SPEED  mean={speed.mean():.3f} median={np.median(speed):.3f} "
          f"std={speed.std():.3f} min={speed.min():.3f} max={speed.max():.3f} m/s")
    print(f"    pctiles p5={_pct(speed,5):.2f} p25={_pct(speed,25):.2f} "
          f"p50={_pct(speed,50):.2f} p75={_pct(speed,75):.2f} p95={_pct(speed,95):.2f}")
    print(f"    %reverse(<0): {100.0*(speed<0).mean():.1f}%   "
          f"%stopped(<{args.speed_stop}): {100.0*(speed<args.speed_stop).mean():.1f}%   "
          f"%cruise(>2.5): {100.0*(speed>2.5).mean():.1f}%")

    eps = _segment_episodes(speed, args.speed_moving, args.speed_stop)
    ep_lens = np.array([b - a for a, b in eps])
    n_ep = len(eps)
    print(f"  EPISODES  n={n_ep}  mean_len={ep_lens.mean():.0f} steps "
          f"(min={ep_lens.min()}, max={ep_lens.max()})")
    print(f"    goals/episode ~= {args.goal_budget}/{n_ep} = "
          f"{args.goal_budget/max(n_ep,1):.1f}  (budget-based estimate; "
          f"crash-free stage ~= {int(np.ceil(args.goal_budget/args.goal_cap))} eps)")
    crashy = None
    if fwd_idx is None or obs.shape[1] <= fwd_idx:
        print("    est crash-ended episodes: skipped "
              "(this vector has no forward ray)")
    else:
        fwd = obs[:, fwd_idx]
        fwd_at_end = np.array([fwd[b - 1] for a, b in eps])
        crashy = int((fwd_at_end < _pct(fwd, 10)).sum())
        print(f"    est crash-ended episodes (fwd clearance<p10 at end): "
              f"{crashy}/{n_ep}")
    return {"n": n, "speed_mean": float(speed.mean()), "crash_ended": crashy}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", help="DEMO job id, job_<id>, or full /tfrecords dir")
    ap.add_argument("--goal-budget", type=int, default=100,
                    help="goals collected per stage (robotaxi.GOALS_PER_STAGE)")
    ap.add_argument("--goal-cap", type=int, default=30,
                    help="episode goal cap (donut_course.GOALS_PER_EPISODE_CAP)")
    ap.add_argument("--speed-moving", type=float, default=1.0,
                    help="m/s above which the car is 'driving'")
    ap.add_argument("--speed-stop", type=float, default=0.3,
                    help="m/s below which the car is 'stopped' (respawn)")
    args = ap.parse_args()

    directory = _resolve_dir(args.job)
    if not os.path.isdir(directory):
        raise SystemExit(f"tfrecords dir not found: {directory}")
    files = sorted(glob.glob(os.path.join(directory, "*.tfrecord")))
    if not files:
        raise SystemExit(f"no .tfrecord files in {directory}")

    print("################ OVERALL ################")
    layout = record_layout(files[0])
    if layout["kind"] == "camera":
        print(f"camera corpus {layout['course']}: "
              f"speed at vector[{layout['speed_idx']}]")
    all_obs = _read_vectors(directory, layout)
    analyze("ALL STAGES combined", all_obs, args,
            layout["speed_idx"], layout["fwd_idx"])

    print("\n################ PER STAGE ################")
    for f in files:
        stage = os.path.basename(f).split("_")[0].lstrip("0") or "0"
        stage_layout = record_layout(f)
        td = tempfile.mkdtemp()
        try:
            shutil.copy(f, td)
            obs = _read_vectors(td, stage_layout)
            analyze(f"stage {stage}", obs, args,
                    stage_layout["speed_idx"], stage_layout["fwd_idx"])
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
