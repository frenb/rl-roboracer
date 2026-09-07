"""Ablation assess: donut_camera_no_rays dict spec + one live reset/step.

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python check_donut_camera_no_rays.py"

Unity Play (or the gym) must be up. Do not run this against the same
ros-server a TRAIN job is stepping - reset/step will fight that actor.

Optional args: ros-server address (default ros-server:50051), `--spec-only`
to skip the live env (no Unity).

The load-bearing assertion is that the policy-visible vector is (2,), NOT
(31,). If the raycasts leak back into the observation this arm becomes a
second donut_camera run and the ablation measures nothing.
"""
import sys

import numpy as np

from collect_training_data import COURSE_OBS_KIND, COURSE_OBSERVATION_SIZES
from environments.courses.donut_course_camera_no_rays import (
    DonutCourseCameraNoRays, POLICY_VECTOR_DIMS)

ADDR = "ros-server:50051"
SPEC_ONLY = False
for a in sys.argv[1:]:
    if a == "--spec-only":
        SPEC_ONLY = True
    else:
        ADDR = a

DUMMY_KEYS = (
    "dist_from_traj", "speed", "goal_2", "left", "forward_left",
    "forward_left_left", "n_27_50", "n_25_00", "n_22_50", "n_20_00",
    "n_17_50", "n_15_00", "n_12_50", "n_10_00", "n_07_50", "n_05_00",
    "n_02_50", "forward", "p_02_50", "p_05_00", "p_07_50", "p_10_00",
    "p_12_50", "p_15_00", "p_17_50", "p_20_00", "p_22_50", "p_25_00",
    "p_27_50", "forward_right_right", "forward_right", "right")


def _check_spec():
    course = DonutCourseCameraNoRays(None, None)
    spec = course.observation_spec
    assert isinstance(spec, dict), spec
    assert set(spec.keys()) == {"vector", "image"}, spec.keys()
    assert spec["vector"].shape == (POLICY_VECTOR_DIMS,), spec["vector"].shape
    assert spec["image"].shape == (84, 84, 3), spec["image"].shape
    assert spec["image"].dtype == np.float32
    # speed +/-10, sideslip +/-1 - sliced from the no-hint vector's bounds.
    assert list(spec["vector"].minimum) == [-10, -1], spec["vector"].minimum
    assert list(spec["vector"].maximum) == [10, 1], spec["vector"].maximum
    empty = course.get_empty_state()
    assert empty["vector"].shape == (POLICY_VECTOR_DIMS,)
    assert empty["image"].shape == (84, 84, 3)
    assert COURSE_OBS_KIND["donut_camera_no_rays"] == "dict"
    assert "donut_camera_no_rays" not in COURSE_OBSERVATION_SIZES

    # scene_data_array stays 31-D: rewards and _step's data_arr[7] step cost
    # index it exactly as on every other donut course.
    dummy = {"car": {k: 0.0 for k in DUMMY_KEYS}}
    dummy["car"]["speed"] = 3.5
    dummy["car"]["goal_2"] = -0.25
    vec = np.asarray(course.scene_data_array(dummy))
    assert vec.shape == (31,), vec.shape

    pol = np.asarray(course.policy_vector(vec))
    assert pol.shape == (POLICY_VECTOR_DIMS,), pol.shape
    assert pol.dtype == np.float32, pol.dtype
    assert float(pol[0]) == 3.5 and float(pol[1]) == -0.25, pol
    print(
        f"spec OK  vector{spec['vector'].shape} "
        f"image{spec['image'].shape} scene_data_array{vec.shape} "
        f"policy_vector={pol}",
        flush=True)
    return spec


def _check_env_pack():
    """Offline check of the env's dict packing (no Unity, stubbed camera).

    Guards the split that makes the ablation valid: the TimeStep carries the
    2-D policy vector while the 31-D scene array is still what gets handed to
    the reward path.
    """
    from environments.robotaxi_env import RobotaxiEnv

    scene = np.arange(31, dtype=np.float32)  # 0..30, so slices are identifiable
    for course_type, want_vec in (("donut_camera", 31),
                                  ("donut_camera_no_rays", POLICY_VECTOR_DIMS)):
        env = RobotaxiEnv(None, course_type=course_type)
        assert env._dict_obs is True, course_type
        spec = env.observation_spec()
        assert spec["vector"].shape == (want_vec,), (course_type, spec["vector"].shape)
        env._front_camera_image = lambda _src: np.zeros((84, 84, 3), dtype=np.float32)
        obs = env._pack_observation(scene, data={})
        assert obs["vector"].shape == (want_vec,), (course_type, obs["vector"].shape)
        assert obs["image"].shape == (84, 84, 3)
        # Packed vector must be the leading slice of the scene array, in order.
        assert np.array_equal(obs["vector"], scene[:want_vec]), obs["vector"]
        print(f"pack OK  {course_type} -> vector{obs['vector'].shape} "
              f"from scene{scene.shape}", flush=True)

    # Vector courses must be untouched by the dict packing path.
    env = RobotaxiEnv(None, course_type="donut_no_hint")
    assert env._dict_obs is False
    packed = env._pack_observation(scene)
    assert packed.shape == (31,), packed.shape
    print("pack OK  donut_no_hint -> plain vector(31,) (not a dict)", flush=True)


def _check_live(spec):
    import time
    from envs import make_env
    env = make_env(ADDR, course_type="donut_camera_no_rays")
    # Subscribe is a create_task; give the gRPC stream a moment before
    # reset publishes the first CSI frame (ROS-TCP does not latch).
    time.sleep(1.0)
    try:
        env_spec = env.observation_spec()
        assert set(env_spec.keys()) == {"vector", "image"}
        assert env_spec["vector"].shape == (POLICY_VECTOR_DIMS,), env_spec["vector"].shape
        ts0 = env.reset()
        obs0 = ts0.observation
        print(
            f"reset  vector{obs0['vector'].shape}={obs0['vector']} "
            f"image{obs0['image'].shape} dtype={obs0['image'].dtype} "
            f"min={float(obs0['image'].min()):.3f} "
            f"max={float(obs0['image'].max()):.3f}",
            flush=True)
        assert obs0["vector"].shape == (POLICY_VECTOR_DIMS,), obs0["vector"].shape
        assert obs0["image"].shape == (84, 84, 3)
        assert obs0["image"].dtype == np.float32
        assert 0.0 <= float(obs0["image"].min()) and float(obs0["image"].max()) <= 1.0
        ts1 = env.step(np.array([0.2, 0.0], dtype=np.float32))
        obs1 = ts1.observation
        print(
            f"step   vector{obs1['vector'].shape}={obs1['vector']} "
            f"image{obs1['image'].shape} step_type={int(ts1.step_type)}",
            flush=True)
        assert obs1["vector"].shape == (POLICY_VECTOR_DIMS,), obs1["vector"].shape
        assert obs1["image"].shape == (84, 84, 3)
        live = float(obs0["image"].max()) > 0.0 or float(obs1["image"].max()) > 0.0
        if not live:
            print("FAIL: image is all zeros (no CSI frame / timeout fallback).",
                  flush=True)
            return 1
        print("OK - donut_camera_no_rays TimeStep is "
              "{vector: (2,), image: (84, 84, 3)}.", flush=True)
        return 0
    finally:
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"env.close skipped: {e}", flush=True)


def main():
    spec = _check_spec()
    _check_env_pack()
    if SPEC_ONLY:
        print("spec-only: skip live reset/step.", flush=True)
        return 0
    return _check_live(spec)


if __name__ == "__main__":
    sys.exit(main())
