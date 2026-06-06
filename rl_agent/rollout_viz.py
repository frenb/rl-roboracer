"""Policy trajectory-rollout visualization (Phase 1, open-loop).

While a job trains/evaluates, sample K action sequences (each H steps) from
the live policy's action head at the CURRENT observation and publish them to
Unity on the `policy_rollouts` topic. Unity's TrajectoryRolloutViz forward-
simulates each sequence with a bicycle model and draws a fan of candidate
paths. See docs/trajectory-rollout-viz.md for the full design.

Phase 1 is OPEN-LOOP: every sequence is H i.i.d. draws from the same current
distribution (we never re-query the policy along the predicted path), so it
shows the policy's immediate action *spread*, not a true closed-loop rollout.

Sampling avoids `policy.distribution()` (which doesn't serialize on a loaded
SavedModel) by calling `policy.action()` on a tiled batch - each call samples
from the tanh-normal head, so K*H stochastic actions come back in one batched
call. A deterministic (greedy) policy yields a degenerate fan; that's expected
and harmless.

Everything here is gated by ROLLOUT_VIZ_ENABLED (default off) so normal
training pays zero cost.
"""
import os
import json
import time

import numpy as np
from tf_agents.trajectories import time_step as ts


def get_config():
    """Read viz knobs from the environment. All optional; safe defaults."""
    def _int(key, default):
        try:
            return int(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float(key, default):
        try:
            return float(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    enabled = os.environ.get("ROLLOUT_VIZ_ENABLED", "").lower() in (
        "1", "true", "yes", "on")
    return {
        "enabled": enabled,
        "every_n": max(1, _int("ROLLOUT_VIZ_EVERY_N_STEPS", 10)),
        "K": max(1, _int("ROLLOUT_VIZ_K", 8)),
        "H": max(1, _int("ROLLOUT_VIZ_HORIZON", 15)),
        "dt": _float("ROLLOUT_VIZ_DT", 0.1),
        "actor_index": max(0, _int("ROLLOUT_VIZ_ACTOR_INDEX", 0)),
    }


def _actor_observation(time_step, actor_index):
    """Pull a single actor's observation (1-D float array) from a (possibly
    batched) TimeStep."""
    obs = np.asarray(time_step.observation, dtype=np.float32)
    if obs.ndim == 1:
        return obs
    idx = actor_index if actor_index < obs.shape[0] else 0
    return obs[idx]


def sample_rollouts(policy, time_step, K, H, actor_index=0):
    """Return an (K, H, action_dim) array of sampled actions.

    Tiles the current observation into a batch of K*H and calls
    policy.action() once; each row is an independent sample from the action
    head, which we reshape into K sequences of H actions.
    """
    obs0 = _actor_observation(time_step, actor_index)
    n = K * H
    tiled = np.repeat(obs0[None, :], n, axis=0).astype(np.float32)
    batched = ts.restart(tiled, batch_size=n)
    action = policy.action(batched).action
    action = np.asarray(action, dtype=np.float32).reshape(K, H, -1)
    return action


def build_payload(samples, step, dt, H):
    """JSON-serializable payload. samples: (K, H, action_dim>=2) with
    action[..,0]=accel, action[..,1]=steer (matching robotaxi_env._do_action)."""
    seqs = []
    for k in range(samples.shape[0]):
        seqs.append([
            {"accel": float(samples[k, h, 0]), "steer": float(samples[k, h, 1])}
            for h in range(samples.shape[1])
        ])
    return {
        "stamp": time.time(),
        "step": int(step),
        "dt": float(dt),
        "horizon": int(H),
        "samples": seqs,
    }


def _publish_via_env(env, payload_json, actor_index=0):
    """Hand the JSON payload to the env's RobotApi to publish to Unity.

    For a ParallelPyEnvironment we dispatch to a single actor's subprocess
    (so only that actor's Unity client renders). For a plain RobotaxiEnv we
    call directly. Mirrors configure_env's dispatch pattern.
    """
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        envs = env._envs
        idx = actor_index if actor_index < len(envs) else 0
        promise = envs[idx].call('publish_rollouts', payload_json)
        promise()  # wait for the async ProcessPyEnvironment.call to complete
    elif hasattr(env, "publish_rollouts"):
        env.publish_rollouts(payload_json)
    else:
        raise RuntimeError(f"env {type(env).__name__} has no publish_rollouts route")


class RolloutViz:
    """Throttled rollout sampler+publisher. One instance per training/eval run.

    Usage:
        viz = RolloutViz()
        # in the loop, after a step:
        viz.maybe_publish(policy, step, ts_env, publish_env)
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        self._last_step = None
        if self.cfg["enabled"]:
            print(f"[rollout_viz] enabled: every {self.cfg['every_n']} steps, "
                  f"K={self.cfg['K']} H={self.cfg['H']} dt={self.cfg['dt']} "
                  f"actor={self.cfg['actor_index']}", flush=True)

    def maybe_publish(self, policy, step, ts_env, publish_env=None):
        """Sample + publish if enabled and the cadence has elapsed.

        ts_env: env to read current_time_step() from (the batched training
                env, or the eval BatchedPyEnvironment).
        publish_env: env that owns the RobotApi to publish through (the
                ParallelPyEnvironment or the underlying RobotaxiEnv). Defaults
                to ts_env when they're the same object.
        Never raises - viz failures must not disturb training/eval.
        """
        if not self.cfg["enabled"]:
            return
        if self._last_step is not None and (step - self._last_step) < self.cfg["every_n"]:
            return
        publish_env = publish_env if publish_env is not None else ts_env
        try:
            cur = ts_env.current_time_step()
            samples = sample_rollouts(
                policy, cur, self.cfg["K"], self.cfg["H"], self.cfg["actor_index"])
            payload = build_payload(samples, step, self.cfg["dt"], self.cfg["H"])
            _publish_via_env(publish_env, json.dumps(payload), self.cfg["actor_index"])
            self._last_step = step
        except Exception as e:  # noqa: BLE001 - never break the training loop
            print(f"[rollout_viz] publish skipped (non-fatal): {e}", flush=True)
