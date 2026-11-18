"""Jump-Start RL (Uchendu et al. 2023, arXiv 2204.02372).

A frozen guide policy (the BC actor) drives the first ``guide_steps`` of
every episode and the learning policy drives the rest. The handoff moves
earlier one stage at a time, gated either on combined-policy evals against
the guide alone, or (paper A.4) on a moving average of collect-episode
returns against the best moving average so far.
"""

import json
import os

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts

GUIDE_WEIGHTS_FILE = "jsrl_guide.npz"
STATE_FILE = "jsrl_state.json"


class JsrlPolicy(py_policy.PyPolicy):
    """Guide for the first ``guide_steps`` of each episode, learner after.

    Works on batched (parallel env) and unbatched time steps; the step
    counter restarts per env on each FIRST step.
    """

    def __init__(self, guide, learner, guide_steps=0):
        if learner.info_spec != ():
            raise ValueError("JsrlPolicy needs a learner without policy info")
        super().__init__(learner.time_step_spec, learner.action_spec,
                         learner.policy_state_spec, learner.info_spec)
        self._guide = guide
        self._learner = learner
        self.guide_steps = int(guide_steps)
        self._t = None

    def _get_initial_state(self, batch_size):
        self._t = None
        return self._learner.get_initial_state(batch_size)

    def _action(self, time_step, policy_state):
        step_type = np.asarray(time_step.step_type)
        first = np.atleast_1d(step_type == ts.StepType.FIRST)
        if self._t is None or self._t.shape != first.shape:
            self._t = np.zeros(first.shape, np.int64)
        self._t[first] = 0
        use_guide = self._t < self.guide_steps
        self._t += 1

        learner_step = self._learner.action(time_step, policy_state)
        if not use_guide.any():
            return learner_step
        guide_action = np.asarray(self._guide.action(time_step).action)
        if step_type.ndim == 0:
            return learner_step._replace(action=guide_action)
        learner_action = np.asarray(learner_step.action)
        mask = use_guide.reshape(
            use_guide.shape + (1,) * (learner_action.ndim - 1))
        return learner_step._replace(
            action=np.where(mask, guide_action, learner_action))


class JsrlCurriculum:
    """Handoff schedule: ``horizon`` guide steps down to 0 in ``stages`` steps.

    With ``window`` 0, ``reference`` is the guide-alone eval return and a
    stage advances after ``consecutive`` evals in a row where the combined
    policy scores at least ``reference - (1 - tolerance) * |reference|``.

    With ``window`` > 0 (the paper's A.4 rule), a stage advances once the
    mean of the last ``window`` samples taken in that stage is at least
    ``best - (1 - tolerance) * |best|``, where ``best`` is the highest such
    moving average seen in any earlier window of the run. Stage 0 windows
    are almost all guide driving, so they neither set ``best`` nor get
    gated: stage 0 advances on its first full window, and the first full
    window of stage 1 only sets ``best``.
    """

    def __init__(self, stages, tolerance, consecutive=2, window=0):
        self.stages = int(stages)
        self.tolerance = float(tolerance)
        self.consecutive = int(consecutive)
        self.window = int(window)
        self.horizon = 0
        self.reference = None
        self.stage = 0
        self.hits = 0
        self.samples = []
        self.best = None
        self.last_average = None

    def start(self, horizon, reference):
        self.horizon = max(1, int(round(horizon)))
        self.reference = float(reference)
        self.stage = 0
        self.hits = 0
        self.samples = []
        self.best = None

    @property
    def guide_steps(self):
        return int(round(self.horizon * (self.stages - self.stage)
                         / self.stages))

    @property
    def threshold(self):
        bar = self.best if self.window > 0 else self.reference
        if bar is None:
            return float("-inf")
        return bar - (1.0 - self.tolerance) * abs(bar)

    @property
    def moving_average(self):
        if self.window <= 0 or len(self.samples) < self.window:
            return None
        return float(np.mean(self.samples[-self.window:]))

    @property
    def finished(self):
        return self.stage >= self.stages

    def update(self, combined_return):
        """Record one combined-policy sample. True when the stage advanced."""
        if self.finished:
            return False
        if self.window > 0:
            self.samples = (self.samples + [float(combined_return)])[-self.window:]
            avg = self.last_average = self.moving_average
            if avg is None:
                return False
            if self.stage > 0:
                if self.best is None:
                    self.best = avg
                    return False
                passed = avg >= self.threshold
                self.best = max(self.best, avg)
                if not passed:
                    return False
            self.samples = []
        else:
            self.hits = self.hits + 1 if combined_return >= self.threshold else 0
            if self.hits < self.consecutive:
                return False
        self.stage += 1
        self.hits = 0
        return True

    def to_dict(self):
        return {"stages": self.stages, "tolerance": self.tolerance,
                "consecutive": self.consecutive, "window": self.window,
                "horizon": self.horizon, "reference": self.reference,
                "stage": self.stage, "hits": self.hits,
                "samples": self.samples, "best": self.best}

    def load_dict(self, state):
        self.horizon = int(state["horizon"])
        self.reference = float(state["reference"])
        self.stage = min(int(state["stage"]), self.stages)
        self.hits = int(state.get("hits", 0))
        self.samples = [float(s) for s in state.get("samples", [])]
        best = state.get("best")
        self.best = None if best is None else float(best)


class TrainingEpisodeGate:
    """Turns collect-episode returns into curriculum samples.

    Each sample is the mean return of ``episodes`` finished collect episodes
    that also started in the current stage (episodes in flight across an
    advance ran under two handoff points, so they are dropped). Returns
    are summed the same way as tf-agents' AverageReturnMetric. Add as a
    collect actor observer and call ``drain`` after each collect run.
    """

    def __init__(self, curriculum, episodes):
        self.curriculum = curriculum
        self.episodes = max(1, int(episodes))
        self._returns = None
        self._start_stage = None
        self._finished = []

    def __call__(self, trajectory):
        reward = np.atleast_1d(np.asarray(trajectory.reward, np.float64))
        first = np.atleast_1d(trajectory.is_first())
        last = np.atleast_1d(trajectory.is_last())
        if self._returns is None or self._returns.shape != reward.shape:
            self._returns = np.zeros(reward.shape, np.float64)
            self._start_stage = np.full(reward.shape, -1, np.int64)
        self._returns[first] = 0.0
        self._start_stage[first] = self.curriculum.stage
        self._returns += reward
        for i in np.flatnonzero(last):
            if self._start_stage[i] == self.curriculum.stage:
                self._finished.append(float(self._returns[i]))
            self._start_stage[i] = -1

    def drain(self):
        """[(sample, advanced)] for every full group of finished episodes."""
        out = []
        while len(self._finished) >= self.episodes:
            group = self._finished[:self.episodes]
            self._finished = self._finished[self.episodes:]
            sample = float(np.mean(group))
            advanced = self.curriculum.update(sample)
            out.append((sample, advanced))
            if advanced:
                self._finished = []
        return out


def save_guide(directory, network):
    weights = network.get_weights()
    np.savez(os.path.join(directory, GUIDE_WEIGHTS_FILE),
             **{f"w{i:04d}": w for i, w in enumerate(weights)})


def load_guide(directory, network):
    """Load saved guide weights into ``network``. False if none saved."""
    path = os.path.join(directory, GUIDE_WEIGHTS_FILE)
    if not os.path.exists(path):
        return False
    with np.load(path) as saved:
        network.set_weights([saved[k] for k in sorted(saved.files)])
    return True


def save_state(directory, curriculum):
    with open(os.path.join(directory, STATE_FILE), "w") as f:
        json.dump(curriculum.to_dict(), f)


def load_state(directory, curriculum):
    """Restore the curriculum. False if no state was saved."""
    path = os.path.join(directory, STATE_FILE)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        curriculum.load_dict(json.load(f))
    return True
