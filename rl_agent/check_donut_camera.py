"""Phase 5 assess: donut_camera dict spec + one live reset/step.

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python check_donut_camera.py"

Unity Play (or the gym) must be up. Do not run this against the same
ros-server a TRAIN job is stepping — reset/step will fight that actor.

Optional args: ros-server address (default ros-server:50051), `--spec-only`
to skip the live env (no Unity).
"""
import sys

import numpy as np

from collect_training_data import COURSE_OBS_KIND, COURSE_OBSERVATION_SIZES
from environments.courses.donut_course_camera import DonutCourseCamera

ADDR = "ros-server:50051"
SPEC_ONLY = False
for a in sys.argv[1:]:
    if a == "--spec-only":
        SPEC_ONLY = True
    else:
        ADDR = a


def _check_spec():
    course = DonutCourseCamera(None, None)
    spec = course.observation_spec
    assert isinstance(spec, dict), spec
    assert set(spec.keys()) == {"vector", "image"}, spec.keys()
    assert spec["vector"].shape == (31,), spec["vector"].shape
    assert spec["image"].shape == (84, 84, 3), spec["image"].shape
    assert spec["image"].dtype == np.float32
    empty = course.get_empty_state()
    assert empty["vector"].shape == (31,)
    assert empty["image"].shape == (84, 84, 3)
    assert COURSE_OBS_KIND["donut_camera"] == "dict"
    assert "donut_camera" not in COURSE_OBSERVATION_SIZES
    dummy = {"car": {k: 0.0 for k in (
        "dist_from_traj", "speed", "goal_2", "left", "forward_left",
        "forward_left_left", "n_27_50", "n_25_00", "n_22_50", "n_20_00",
        "n_17_50", "n_15_00", "n_12_50", "n_10_00", "n_07_50", "n_05_00",
        "n_02_50", "forward", "p_02_50", "p_05_00", "p_07_50", "p_10_00",
        "p_12_50", "p_15_00", "p_17_50", "p_20_00", "p_22_50", "p_25_00",
        "p_27_50", "forward_right_right", "forward_right", "right")}}
    vec = np.asarray(course.scene_data_array(dummy))
    assert vec.shape == (31,), vec.shape
    print(
        f"spec OK  vector{spec['vector'].shape} "
        f"image{spec['image'].shape} scene_data_array{vec.shape}",
        flush=True)
    return spec


def _check_live(spec):
    import time
    from envs import make_env
    env = make_env(ADDR, course_type="donut_camera")
    # Subscribe is a create_task; give the gRPC stream a moment before
    # reset publishes the first CSI frame (ROS-TCP does not latch).
    time.sleep(1.0)
    try:
        env_spec = env.observation_spec()
        assert set(env_spec.keys()) == {"vector", "image"}
        ts0 = env.reset()
        obs0 = ts0.observation
        print(
            f"reset  vector{obs0['vector'].shape} "
            f"image{obs0['image'].shape} dtype={obs0['image'].dtype} "
            f"min={float(obs0['image'].min()):.3f} "
            f"max={float(obs0['image'].max()):.3f}",
            flush=True)
        assert obs0["vector"].shape == spec["vector"].shape
        assert obs0["image"].shape == spec["image"].shape
        assert obs0["image"].dtype == np.float32
        assert 0.0 <= float(obs0["image"].min()) and float(obs0["image"].max()) <= 1.0
        ts1 = env.step(np.array([0.2, 0.0], dtype=np.float32))
        obs1 = ts1.observation
        print(
            f"step   vector{obs1['vector'].shape} "
            f"image{obs1['image'].shape} step_type={int(ts1.step_type)}",
            flush=True)
        assert obs1["vector"].shape == (31,)
        assert obs1["image"].shape == (84, 84, 3)
        live = float(obs0["image"].max()) > 0.0 or float(obs1["image"].max()) > 0.0
        if not live:
            print("FAIL: image is all zeros (no CSI frame / timeout fallback).",
                  flush=True)
            return 1
        print("OK - donut_camera TimeStep is {vector: (31,), image: (84, 84, 3)}.",
              flush=True)
        return 0
    finally:
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"env.close skipped: {e}", flush=True)


def main():
    spec = _check_spec()
    if SPEC_ONLY:
        print("spec-only: skip live reset/step.", flush=True)
        return 0
    return _check_live(spec)


if __name__ == "__main__":
    sys.exit(main())
