"""Policy trajectory-rollout visualization (Phase 1, open-loop).

A background thread samples K action sequences (each H steps) from the live
policy at the latest observation and publishes them to Unity on the
`policy_rollouts` topic at a FIXED rate (default 20 Hz). Unity's
TrajectoryRolloutViz forward-simulates each sequence with a bicycle model and
draws a fan of candidate paths. See docs/trajectory-rollout-viz.md.

WHY A BACKGROUND THREAD (not loop-driven):
The training loop runs at a few Hz and stalls for 20-30s during periodic eval
cycles, so publishing from inside the loop made the fan update at ~3 Hz and go
dark during every eval window. The background thread decouples the viz from the
loop: the loop just feeds it the latest (policy, observation) via
update_context()/maybe_publish(), and the thread re-samples + publishes at a
steady 20 Hz regardless of whether the trainer is collecting, learning, or
evaluating. The result is an always-on fan that genuinely re-renders new
rollouts every tick.

Phase 1 is OPEN-LOOP: every sequence is H i.i.d. draws from the same current
distribution (we never re-query the policy along the predicted path), so it
shows the policy's immediate action *spread*, not a true closed-loop rollout.

Sampling avoids `policy.distribution()` (which doesn't serialize on a loaded
SavedModel) by calling `policy.action()` on a tiled batch - each call samples
from the tanh-normal head, so K*H stochastic actions come back in one batched
call. A deterministic (greedy) policy yields a degenerate fan; that's expected.

PUBLISH PATH: the thread owns its OWN synchronous gRPC channel straight to
actor-0's ros-server (ros-server-0:50051) rather than routing through the
ParallelPyEnvironment subprocess pipes - that avoids racing the training
thread's env.step() on the same pipe.

Everything here is gated by ROLLOUT_VIZ_ENABLED (default off) so normal
training pays zero cost.
"""
import os
import json
import time
import threading
import contextlib

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
        "K": max(1, _int("ROLLOUT_VIZ_K", 8)),
        "H": max(1, _int("ROLLOUT_VIZ_HORIZON", 25)),
        "dt": _float("ROLLOUT_VIZ_DT", 0.1),
        "actor_index": max(0, _int("ROLLOUT_VIZ_ACTOR_INDEX", 0)),
        # Publish a fan for EVERY actor in the batch (each to its own
        # ros-server-{i}) so all clients render during the collect phase. Set
        # 0/false to restrict to the single `actor_index` (legacy behavior).
        "all_actors": os.environ.get("ROLLOUT_VIZ_ALL_ACTORS", "1").lower()
        in ("1", "true", "yes", "on"),
        # Run the viz policy inference on CPU. Default OFF: measured worse than
        # the (contended) GPU - the small actor net runs faster on GPU even
        # while training, and CPU pinning also copies the GPU-resident weights
        # each call. Set 1 to force CPU.
        "cpu_inference": os.environ.get("ROLLOUT_VIZ_CPU", "0").lower()
        in ("1", "true", "yes", "on"),
        # Fixed publish/redraw rate, decoupled from the training loop.
        "hz": max(1.0, _float("ROLLOUT_VIZ_HZ", 20.0)),
        # Per-actor ros-server gRPC endpoint template ({i} = actor index). Each
        # actor i's env is wired to ros-server-{i}, so we publish actor i's
        # rollouts there; the matching Unity client subscribes locally.
        "ros_addr_template": os.environ.get(
            "ROLLOUT_VIZ_ROS_ADDR_TEMPLATE", "ros-server-{i}:50051"),
        # ---- Render-style knobs, published in each payload so Unity can be
        # tuned WITHOUT a rebuild (just set the env var + restart the trainer).
        # Unity falls back to its own inspector defaults if these are absent.
        "line_width": _float("ROLLOUT_VIZ_LINE_WIDTH", 1.5),
        "prob_falloff": _float("ROLLOUT_VIZ_PROB_FALLOFF", 1.0),
        "min_alpha": _float("ROLLOUT_VIZ_MIN_ALPHA", 0.30),
        "max_alpha": _float("ROLLOUT_VIZ_MAX_ALPHA", 0.95),
        "min_width": _float("ROLLOUT_VIZ_MIN_WIDTH", 0.40),
        "max_width": _float("ROLLOUT_VIZ_MAX_WIDTH", 1.6),
    }


def _actor_observation(time_step, actor_index):
    """Pull a single actor's observation (1-D float array) from a (possibly
    batched) TimeStep."""
    obs = np.asarray(time_step.observation, dtype=np.float32)
    if obs.ndim == 1:
        return obs
    idx = actor_index if actor_index < obs.shape[0] else 0
    return obs[idx]


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(e, 1.0 / max(1, e.size))
    return (e / s).astype(np.float32)


def _sample_actions_logp(policy, tiled, cpu=True):
    """Run the policy on a tiled obs batch -> (actions (N,A), logp (N,) or None).

    Shared by the single- and multi-actor samplers. Returns logp=None (caller
    uses uniform weights) only if BOTH log-prob methods fail.

    When cpu=True the inference runs under tf.device('/CPU:0') so it executes on
    the CPU in parallel with the GPU training steps instead of queueing behind
    them (the actor net is small, so CPU inference is cheap and frees the GPU).

    Getting log-probs is attempted two ways, because tf_agents'
    policy.distribution() rebuilds the distribution from a DistributionSpecV2
    and that rebuild throws "__init__() got an unexpected keyword argument
    'loc'" with this tfp version:
      1. policy.distribution(ts).action  (works on the LIVE policy)
      2. the policy's actor network called directly (live tfp distribution,
         skips the spec rebuild that triggers the bug).
    """
    import tensorflow as tf
    tiled = np.asarray(tiled, dtype=np.float32)
    n = tiled.shape[0]
    # The trainer hands us a PyTFEagerPolicy (a numpy wrapper); unwrap to the
    # underlying tf policy which actually exposes distribution()/_actor_network.
    base = getattr(policy, "_policy", None) or policy

    err1 = None
    err2 = None
    dev_ctx = tf.device('/CPU:0') if cpu else contextlib.nullcontext()
    with dev_ctx:
        obs_t = tf.constant(tiled)
        ts_tf = ts.restart(obs_t, batch_size=n)
        try:
            dist = base.distribution(ts_tf).action
            a = dist.sample()
            return (np.asarray(a, dtype=np.float32),
                    np.asarray(dist.log_prob(a), dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            err1 = e
        try:
            net = getattr(base, "_actor_network", None)
            if net is None:
                raise RuntimeError("policy has no _actor_network")
            out, _ = net(obs_t, step_type=ts_tf.step_type,
                         network_state=(), training=False)
            dist = out[0] if isinstance(out, (list, tuple)) else out
            a = dist.sample()
            return (np.asarray(a, dtype=np.float32),
                    np.asarray(dist.log_prob(a), dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            err2 = e

    if not getattr(_sample_actions_logp, "_warned", False):
        print(f"[rollout_viz] log-prob unavailable; UNIFORM weights. "
              f"distribution(): {type(err1).__name__}: {err1} | "
              f"actor_net: {type(err2).__name__}: {err2}", flush=True)
        _sample_actions_logp._warned = True
    a = policy.action(ts.restart(tiled, batch_size=n)).action
    return np.asarray(a, dtype=np.float32), None


def _weights_from_logp(logp, M, K, H):
    """Per-actor softmax over each sequence's joint (summed-over-horizon) logp.

    logp: (M*K*H,) or (M*K*H, A). Returns (M, K)."""
    logp = np.asarray(logp, dtype=np.float32)
    if logp.ndim == 2:
        logp = logp.sum(axis=-1)         # joint over action dims
    traj_logp = logp.reshape(M, K, H).sum(axis=2)   # (M, K)
    return np.stack([_softmax(traj_logp[m]) for m in range(M)], axis=0)


def sample_rollouts_multi(policy, obs_arr, K, H, cpu=True):
    """Batched multi-actor sampling: ONE policy call for all M actors.

    obs_arr: (M, obs_dim). Tiles to a single (M*K*H, obs_dim) batch (actor-
    major), runs the policy once, and splits back per actor. A single big
    inference is much cheaper than M separate calls. cpu=True runs it on the
    CPU (off the training GPU).

    Returns (samples (M, K, H, A), weights (M, K)). Weights are the open-loop
    joint-likelihood softmax across the K sequences, computed per actor; uniform
    if the policy can't yield log-probs (e.g. the greedy eval policy).
    """
    obs_arr = np.asarray(obs_arr, dtype=np.float32)
    M = obs_arr.shape[0]
    n_per = K * H
    tiled = np.repeat(obs_arr, n_per, axis=0)        # (M*n_per, obs_dim)
    actions, logp = _sample_actions_logp(policy, tiled, cpu=cpu)
    A = actions.shape[-1]
    samples = actions.reshape(M, K, H, A)
    if logp is not None:
        weights = _weights_from_logp(logp, M, K, H)
    else:
        weights = np.full((M, K), 1.0 / max(1, K), dtype=np.float32)
    return samples, weights


def sample_rollouts_from_obs(policy, obs0, K, H):
    """Single-observation convenience wrapper around sample_rollouts_multi.

    Returns (samples (K,H,A), weights (K,)). See sample_rollouts_multi for the
    open-loop sampling + per-trajectory weighting details.
    """
    samples, weights = sample_rollouts_multi(
        policy, np.asarray(obs0, dtype=np.float32)[None, :], K, H)
    return samples[0], weights[0]


def build_payload(samples, weights, step, dt, H, style=None):
    """JSON-serializable payload for the Unity renderer.

    samples: (K, H, action_dim>=2) with action[..,0]=accel, action[..,1]=steer
    (matching robotaxi_env._do_action).
    weights: (K,) relative probability per trajectory (softmax over joint logp);
    Unity maps these to per-line opacity/width.
    style: optional dict of render knobs (line_width, prob_falloff, min/max_alpha,
    min/max_width) published so Unity can be tuned without a rebuild.

    We emit FLAT, k-major parallel arrays (accel[k*H + h], steer[k*H + h])
    rather than nested lists so Unity's JsonUtility can parse it without a JSON
    library. The renderer reshapes back to K sequences of H via index = k*H + h.
    """
    K = int(samples.shape[0])
    # Flatten k-major (k outer, h inner -> index k*H + h). Vectorized .tolist()
    # is much cheaper than a Python double-loop at K*H entries per actor.
    s = np.asarray(samples, dtype=np.float32)
    accel = s[:, :, 0].reshape(-1).tolist()
    steer = s[:, :, 1].reshape(-1).tolist()
    payload = {
        "stamp": time.time(),
        "step": int(step),
        "dt": float(dt),
        "horizon": int(H),
        "k": K,
        "accel": accel,
        "steer": steer,
        "weights": [float(w) for w in weights],
    }
    if style:
        # Flat keys matching Unity's RolloutPayload fields. maxAlpha>0 is
        # Unity's "style present" sentinel (JsonUtility zero-fills absentees).
        payload["lineWidth"] = float(style.get("line_width", 1.5))
        payload["probFalloff"] = float(style.get("prob_falloff", 1.0))
        payload["minAlpha"] = float(style.get("min_alpha", 0.30))
        payload["maxAlpha"] = float(style.get("max_alpha", 0.95))
        payload["minWidth"] = float(style.get("min_width", 0.40))
        payload["maxWidth"] = float(style.get("max_width", 1.6))
    return payload


class _DirectPublisher:
    """Synchronous gRPC publisher straight to a ros-server's RosNode endpoint.

    Mirrors api.RpcClient.Publish's wire format (data is the JSON-encoded ROS
    message dict) but blocking, so it can be called from the viz thread without
    an asyncio loop.
    """

    def __init__(self, addr):
        import grpc
        from virtual_endpoint.proto import ros_service_pb2
        from virtual_endpoint.proto import ros_service_pb2_grpc
        self._pb2 = ros_service_pb2
        self._channel = grpc.insecure_channel(addr)
        self._stub = ros_service_pb2_grpc.RosNodeStub(self._channel)

    def publish_rollout(self, payload_json):
        req = self._pb2.PublishRequest(
            topic="policy_rollouts",
            msg_type="std_msgs/String",
            data=json.dumps({"data": payload_json}))
        self._stub.Publish(req, timeout=3.0)


class RolloutViz:
    """Background sampler+publisher. Construct once (see get_viz singleton).

    Usage from the training/eval loops:
        viz = get_viz()
        # each iteration, hand it the live policy + env to read obs from:
        viz.maybe_publish(policy, step, ts_env)   # back-compat shim
        # or equivalently:
        viz.update_context(policy, ts_env, step)
    The background thread does the actual sampling + 20 Hz publishing.
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        self._policy = None
        self._obs = None            # full batched obs (num_actors, obs_dim)
        self._step = 0
        self._mode = "train"        # "train" or "eval" (shown in the Unity HUD)
        self._ctx_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pubs = {}             # addr -> _DirectPublisher (one per server)
        self._pub_fail = set()      # addrs that failed to init (don't retry)
        self._err_count = 0
        if self.cfg["enabled"]:
            self._start_background()

    def _start_background(self):
        self._thread = threading.Thread(
            target=self._run, name="rollout-viz", daemon=True)
        self._thread.start()
        target = ("all actors via " + self.cfg["ros_addr_template"]
                  if self.cfg["all_actors"]
                  else self.cfg["ros_addr_template"].format(
                      i=self.cfg["actor_index"]))
        print(f"[rollout_viz] background sampler @ {self.cfg['hz']}Hz -> "
              f"{target} (K={self.cfg['K']} H={self.cfg['H']} "
              f"dt={self.cfg['dt']} all_actors={self.cfg['all_actors']} "
              f"device={'CPU' if self.cfg['cpu_inference'] else 'GPU'})",
              flush=True)

    def _publisher_for(self, i):
        """Lazily create (and cache) the gRPC publisher for actor i's server."""
        addr = self.cfg["ros_addr_template"].format(i=i)
        pub = self._pubs.get(addr)
        if pub is None and addr not in self._pub_fail:
            try:
                pub = _DirectPublisher(addr)
                self._pubs[addr] = pub
            except Exception as e:  # noqa: BLE001
                self._pub_fail.add(addr)
                print(f"[rollout_viz] publisher init failed for {addr} ({e}); "
                      f"skipping that client", flush=True)
                return None
        return pub

    def update_context(self, policy, ts_env, step=None, mode="train"):
        """Hand the background thread the latest policy + FULL batched obs.

        Stores every actor's observation (shape (num_actors, obs_dim)); the
        thread samples + publishes one fan per actor to its own ros-server.
        `mode` ("train"/"eval") is forwarded in the payload for the Unity HUD.
        Cheap (a small numpy copy). Called from the main thread only, so
        there's no concurrent env access. Never raises.
        """
        if not self.cfg["enabled"] or self._thread is None:
            return
        try:
            cur = ts_env.current_time_step()
            obs = np.asarray(cur.observation, dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[None, :]   # (obs_dim,) -> (1, obs_dim)
            obs = obs.copy()
        except Exception:  # noqa: BLE001 - viz must never disturb the loop
            return
        with self._ctx_lock:
            self._policy = policy
            self._obs = obs
            self._mode = mode
            if step is not None:
                self._step = int(step)

    # Back-compat shim: existing call sites call maybe_publish(policy, step,
    # ts_env, publish_env). It now just feeds context to the background sampler
    # (publish_env is ignored - the thread owns its own publish channel).
    def maybe_publish(self, policy, step, ts_env, publish_env=None):
        self.update_context(policy, ts_env, step)

    def _run(self):
        period = 1.0 / max(1.0, float(self.cfg["hz"]))
        while not self._stop.is_set():
            t0 = time.time()
            with self._ctx_lock:
                policy = self._policy
                obs = self._obs
                step = self._step
                mode = self._mode
            if policy is not None and obs is not None:
                n = obs.shape[0]
                # Which actors to publish for. During eval the batch is size 1
                # (the single eval env on ros-server-0).
                if self.cfg["all_actors"]:
                    indices = list(range(n))
                else:
                    indices = [min(self.cfg["actor_index"], n - 1)]
                try:
                    # ONE batched inference for all selected actors, then split
                    # + publish each to its own ros-server-{i}. A single big
                    # policy call is far cheaper than one call per actor.
                    obs_sel = obs[indices]
                    samples_m, weights_m = sample_rollouts_multi(
                        policy, obs_sel, self.cfg["K"], self.cfg["H"],
                        cpu=self.cfg["cpu_inference"])
                    for j, i in enumerate(indices):
                        pub = self._publisher_for(i)
                        if pub is None:
                            continue
                        payload = build_payload(
                            samples_m[j], weights_m[j], step, self.cfg["dt"],
                            self.cfg["H"], self.cfg)
                        payload["mode"] = mode      # train/eval, for the HUD
                        payload["actor"] = int(i)   # actor index (HUD cross-check)
                        pub.publish_rollout(json.dumps(payload))
                except Exception as e:  # noqa: BLE001 - never kill the thread
                    self._err_count += 1
                    if self._err_count <= 5 or self._err_count % 200 == 0:
                        print(f"[rollout_viz] sample/publish error "
                              f"#{self._err_count}: {e}", flush=True)
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, period - elapsed))

    def stop(self):
        self._stop.set()


# Module-level singleton so the train loop and the eval loop share ONE
# background sampler (one thread, one publish channel). Both feed it context;
# whichever phase is active provides the freshest policy + observation.
_GLOBAL_VIZ = None
_GLOBAL_VIZ_LOCK = threading.Lock()


def get_viz():
    # Double-checked locking: the eval-viz helper thread (robotaxi.py) and the
    # main loop can both call this, so guard the lazy singleton construction.
    global _GLOBAL_VIZ
    if _GLOBAL_VIZ is None:
        with _GLOBAL_VIZ_LOCK:
            if _GLOBAL_VIZ is None:
                _GLOBAL_VIZ = RolloutViz()
    return _GLOBAL_VIZ
