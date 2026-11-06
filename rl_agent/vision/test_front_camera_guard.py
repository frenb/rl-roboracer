"""A malformed CSI frame never reaches the policy or fails the job.

No gym and no ROS: RobotApi and RobotaxiEnv are built without __init__.

    docker compose -f docker-compose.yml -f compose/scale.yml exec -e CUDA_VISIBLE_DEVICES= sim-controller \
        sh -c "cd /python_ws/src && python vision/test_front_camera_guard.py"

Exit 0 = all assertions passed.
"""
from __future__ import annotations

import base64
import os
import sys
import traceback
from collections import OrderedDict

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

import api
from environments.robotaxi_env import RobotaxiEnv

_passed = 0
_failed = 0


def _expect(label, predicate, detail=""):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception:
        ok = False
        detail = (detail + "\n" + traceback.format_exc()).strip()
    if ok:
        _passed += 1
        print("PASS  " + label, flush=True)
    else:
        _failed += 1
        extra = ("  " + detail) if detail else ""
        print("FAIL  " + label + extra, flush=True)


def _frame(seq, nbytes=api.FRONT_CAMERA_BYTES, fill=7, width=84, height=84):
    return {"frame": {
        "header": {"seq": seq, "stamp": {}, "frame_id": "camera_visual"},
        "height": height, "width": width, "encoding": "rgb8",
        "is_bigendian": 0, "step": width * 3,
        "data": base64.b64encode(bytes([fill]) * nbytes).decode("ascii"),
    }}


def _bare_api():
    r = object.__new__(api.RobotApi)
    r.latest_front_camera_frame = None
    r.front_camera_frames = OrderedDict()
    r.front_camera_events = {}
    r._front_camera_logged = True
    for name in ("reset_timeouts", "apply_force_timeouts", "scene_data_timeouts",
                 "front_camera_timeouts", "publish_timeouts", "front_camera_bad_frames"):
        setattr(r, name, 0)
    return r


class _FakeEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class _StubApi:
    def __init__(self, frame):
        self.frame = frame
        self.last_apply_force_cmd_id = 1

    def GetFrontCameraFrameBlocking(self, _cmd_id):
        return self.frame


def main():
    problem = api._front_camera_frame_problem
    _expect("good frame passes", lambda: problem(_frame(1)) is None)
    _expect("13082-byte frame rejected", lambda: "13082" in problem(_frame(1, nbytes=13082)))
    _expect("wrong width rejected", lambda: problem(_frame(1, width=64)) is not None)
    _expect("missing frame rejected", lambda: problem({}) is not None)
    _expect("zeros fallback is well formed",
            lambda: problem(api._empty_front_camera_frame(3)) is None)

    r = _bare_api()
    r._on_front_camera_frame(_frame(1, fill=1))
    _expect("good frame becomes last good",
            lambda: r.latest_front_camera_frame["frame"]["header"]["seq"] == 1)

    ev = _FakeEvent()
    r.front_camera_events[2] = ev
    r._on_front_camera_frame(_frame(2, nbytes=13082))
    _expect("bad frame counted", lambda: r.front_camera_bad_frames == 1)
    _expect("bad frame does not replace last good",
            lambda: r.latest_front_camera_frame["frame"]["header"]["seq"] == 1)
    _expect("waiter for the bad step is released", lambda: ev.was_set)
    _expect("bad step is answered with the last good image",
            lambda: problem(r.front_camera_frames[2]) is None)
    _expect("bad frames reported with the counters",
            lambda: r.get_timeout_counts()["front_camera_bad_frames"] == 1)

    fresh = _bare_api()
    fresh._on_front_camera_frame(_frame(5, nbytes=10))
    _expect("bad first frame is not cached", lambda: 5 not in fresh.front_camera_frames)

    env = object.__new__(RobotaxiEnv)
    env._api = _StubApi(_frame(1, fill=51))
    good = env._front_camera_image(None)
    _expect("env decodes a good frame", lambda: good.shape == (84, 84, 3))

    env._api = _StubApi(_frame(2, nbytes=13082))
    _expect("env reuses last good image on a bad frame",
            lambda: np.array_equal(env._front_camera_image(None), good))

    cold = object.__new__(RobotaxiEnv)
    cold._api = _StubApi(_frame(2, nbytes=13082))
    _expect("env returns zeros with no good image yet",
            lambda: not np.any(cold._front_camera_image(None)))

    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
