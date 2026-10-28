"""Check robotaxi.ensure_first_reset against a real ParallelPyEnvironment.

Run in sim-controller (CPU is fine):
    CUDA_VISIBLE_DEVICES= python test_first_reset.py
"""
import sys
import time

import numpy as np
from tf_agents.environments import parallel_py_environment, py_environment
from tf_agents.specs import array_spec
from tf_agents.system import system_multiprocessing
from tf_agents.trajectories import time_step as ts

import robotaxi


class _Env(py_environment.PyEnvironment):
    def __init__(self, hang):
        super().__init__()
        self._hang = hang

    def observation_spec(self):
        return array_spec.ArraySpec((1,), np.float32)

    def action_spec(self):
        return array_spec.BoundedArraySpec((1,), np.float32, -1.0, 1.0)

    def _reset(self):
        while self._hang:
            time.sleep(1)
        return ts.restart(np.zeros(1, np.float32))

    def _step(self, action):
        return ts.transition(np.zeros(1, np.float32), 0.0)


def _parallel(hang):
    return parallel_py_environment.ParallelPyEnvironment(
        [lambda: _Env(hang), lambda: _Env(False)])


def main():
    failures = []

    hung = _parallel(True)
    built = []

    def rebuild():
        built.append(_parallel(False))
        return built[-1]

    env = robotaxi.ensure_first_reset(hung, rebuild, timeout_s=3, attempts=2)
    hung_dead = all(not pe._process.is_alive() for pe in hung._envs)
    if not (env is built[0] and hung_dead and len(built) == 1):
        failures.append("hung env not replaced by the rebuilt one")
    else:
        print("PASS  hung env is killed and the rebuilt env is returned")
    env.close()

    always = []

    def rebuild_hung():
        always.append(_parallel(True))
        return always[-1]

    try:
        robotaxi.ensure_first_reset(_parallel(True), rebuild_hung,
                                    timeout_s=2, attempts=2)
        failures.append("an env that always hangs did not raise")
    except RuntimeError as e:
        if "hung 2 times" in str(e) and len(always) == 1:
            print("PASS  gives up after the attempt limit")
        else:
            failures.append(f"unexpected error: {e}")

    healthy = _parallel(False)
    if robotaxi.ensure_first_reset(healthy, rebuild) is healthy:
        print("PASS  a healthy env is returned unchanged")
    else:
        failures.append("healthy env was replaced")
    healthy.close()

    for f in failures:
        print("FAIL  " + f)
    print(f"{3 - len(failures)} passed, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    system_multiprocessing.handle_main(lambda _argv: main())
