"""A tf-agents PyPolicy that drives from the frozen fly connectome.

Step 6 of docs/flybrain-driver-plan.md: encoder -> brain -> ridge readout,
behind the same interface as any other eval policy, so the normal EVAL path
supplies AverageReturn, goals per episode and a leaderboard row directly
comparable to SAC on identical geometry.

Nothing here is learned at run time. The connectome is fixed wiring and the
readout is the ridge fit from step 5.
"""
import os

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

from fly_brain.client import FlyBrainClient, SUBSTEPS
from fly_brain.encoder import RayEncoder, resolve_cells
from fly_brain.viz import FlyBrainViz

DEFAULT_READOUT = os.environ.get(
    "FLY_READOUT", "/saved_models/robotaxi/FlyPyPolicy/0/readout.npz")


class FlyPyPolicy(py_policy.PyPolicy):
    """Frozen connectome + ridge readout, as a PyPolicy.

    Stateful in a way ordinary policies are not: the brain's voltages carry
    across steps, so it is reset on every StepType.FIRST. The seed advances
    per episode, which keeps runs reproducible while still varying the noise
    between episodes -- see the eval-variance note in step 6, since greedy SAC
    eval is a deterministic tanh(mu) and this is a point estimate with spiking
    noise underneath.
    """

    def __init__(self, time_step_spec, action_spec,
                 readout_path=DEFAULT_READOUT, target=None, substeps=SUBSTEPS):
        super(FlyPyPolicy, self).__init__(time_step_spec, action_spec)
        if not os.path.exists(readout_path):
            raise IOError(
                "fly readout not found at %s - run step 5 "
                "(python -m fly_brain.step5_readout) and copy its output there"
                % readout_path)
        z = np.load(readout_path)
        self._w = z["w"].astype(np.float32)
        self._mu = z["mu"].astype(np.float32)
        self._sd = z["sd"].astype(np.float32)
        self._ym = z["y_mean"].astype(np.float32)

        self._client = FlyBrainClient(target) if target else FlyBrainClient()
        if self._client.info.trace_len != self._w.shape[0]:
            raise ValueError(
                "readout expects a %d-wide trace but the brain serves %d"
                % (self._w.shape[0], self._client.info.trace_len))
        self._enc = RayEncoder()
        self._cells = resolve_cells(self._client)
        self._substeps = int(substeps)
        self._episode = 0
        self._steps = 0
        # The overlay rides along on the step we already make: asking for the
        # snapshot here costs one 2 KB field instead of a second round trip.
        self._viz = FlyBrainViz(self._client)

        self._lo = np.asarray(action_spec.minimum, np.float32)
        self._hi = np.asarray(action_spec.maximum, np.float32)
        print("FlyPyPolicy: readout %s, trace_len=%d, device=%s"
              % (readout_path, self._client.info.trace_len,
                 self._client.info.device), flush=True)

    def _action(self, time_step, policy_state):
        obs = np.asarray(time_step.observation, np.float32)
        batched = obs.ndim == 2
        row = obs[0] if batched else obs

        step_type = np.asarray(time_step.step_type).reshape(-1)
        if step_type.size and step_type[0] == ts.StepType.FIRST:
            self._client.reset(seed=self._episode)
            self._enc.reset()
            self._episode += 1

        _, inject = self._enc.encode(row, self._cells)
        trace, snap, spikes = self._client.step(
            inject, substeps=self._substeps, want_snapshot=self._viz.enabled)
        self._steps += 1
        self._viz.submit(snap, step=self._steps, spikes=spikes)

        act = ((trace - self._mu) / self._sd) @ self._w + self._ym
        # The corpus the readout was fit on holds accel values below the
        # course's own 0.05 floor, so clipping is load-bearing, not cosmetic.
        act = np.clip(act, self._lo, self._hi).astype(np.float32)

        return policy_step.PolicyStep(act[None, :] if batched else act,
                                      policy_state)

    def close(self):
        self._viz.stop()
        self._client.close()
