"""Check jsrl.JsrlPolicy switching and the JsrlCurriculum schedule.

Run in sim-controller (CPU is fine):
    CUDA_VISIBLE_DEVICES= python test_jsrl.py
"""
import sys
import tempfile

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

import jsrl

_failures = []
_passed = 0


def _expect(name, ok):
    global _passed
    if ok:
        _passed += 1
        print("PASS  " + name)
    else:
        _failures.append(name)
        print("FAIL  " + name)


class _Const(py_policy.PyPolicy):
    def __init__(self, value):
        obs = array_spec.ArraySpec((1,), np.float32)
        act = array_spec.BoundedArraySpec((2,), np.float32, -1.0, 1.0)
        super().__init__(ts.time_step_spec(obs), act)
        self._value = value

    def _action(self, time_step, policy_state):
        shape = np.shape(time_step.step_type) + (2,)
        return policy_step.PolicyStep(
            np.full(shape, self._value, np.float32), policy_state, ())


def _steps(types):
    types = np.asarray(types, np.int32)
    n = types.shape
    return ts.TimeStep(types, np.zeros(n, np.float32),
                       np.ones(n, np.float32),
                       np.zeros(n + (1,), np.float32))


def main():
    pol = jsrl.JsrlPolicy(_Const(1.0), _Const(-1.0), guide_steps=2)
    F, M = ts.StepType.FIRST, ts.StepType.MID
    a = [pol.action(_steps(t)).action[:, 0].tolist()
         for t in ([F, F], [M, M], [M, F], [M, M], [M, M])]
    _expect("batched: guide for the first 2 steps of each env's episode",
            a == [[1, 1], [1, 1], [-1, 1], [-1, 1], [-1, -1]])

    pol.guide_steps = 0
    _expect("guide_steps 0 is the learner alone",
            pol.action(_steps([F, F])).action[:, 0].tolist() == [-1, -1])

    single = jsrl.JsrlPolicy(_Const(1.0), _Const(-1.0), guide_steps=1)
    got = [float(single.action(_steps(t)).action[0]) for t in (F, M, F)]
    _expect("unbatched: guide on FIRST only", got == [1.0, -1.0, 1.0])

    cur = jsrl.JsrlCurriculum(stages=4, tolerance=0.9, consecutive=2)
    cur.start(horizon=80, reference=50.0)
    _expect("stage 0 hands the guide the whole horizon", cur.guide_steps == 80)
    _expect("threshold is 90% of the guide return", abs(cur.threshold - 45.0) < 1e-9)
    adv = [cur.update(r) for r in (46, 30, 46, 47)]
    _expect("a miss resets the streak; two hits in a row advance",
            adv == [False, False, False, True] and cur.guide_steps == 60)
    for _ in range(6):
        cur.update(100.0)
    _expect("finishes at guide_steps 0", cur.finished and cur.guide_steps == 0)
    _expect("no advance past the last stage", cur.update(100.0) is False)

    neg = jsrl.JsrlCurriculum(stages=2, tolerance=0.9)
    neg.start(horizon=10, reference=-10.0)
    _expect("negative reference threshold sits below it",
            abs(neg.threshold + 11.0) < 1e-9)

    capped = jsrl.JsrlCurriculum(stages=5, tolerance=0.85, consecutive=1)
    capped.start(horizon=min(655, 200), reference=93.0)
    _expect("capped horizon sets 200->160->120->80->40->0",
            capped.guide_steps == 200 and capped.update(80.0) and capped.guide_steps == 160)

    with tempfile.TemporaryDirectory() as d:
        jsrl.save_state(d, cur)
        back = jsrl.JsrlCurriculum(stages=4, tolerance=0.9, consecutive=2)
        _expect("state round-trips", jsrl.load_state(d, back)
                and back.to_dict() == cur.to_dict())
        _expect("missing guide weights report False",
                jsrl.load_guide(d, None) is False)

    print(f"{_passed} passed, {len(_failures)} failed")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
