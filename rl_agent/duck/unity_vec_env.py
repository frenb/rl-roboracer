"""rsl_rl VecEnv over the Unity duck server, in place of RslRlVecEnvWrapper(ManagerBasedRlEnv).

The server (unity_env.DuckServer) runs the exported Velocity-Flat task on CPU threads; this
class moves its replies to torch tensors on the trainer's device and keeps the parts of
ManagerBasedRlEnv the runners touch:

  - observations as a TensorDict with the env cfg's group names ("actor", "critic")
  - extras["time_outs"], and extras["log"] with Episode_Reward/<term> and
    Episode_Termination/<term> written when ducks reset, as mjlab's _reset_idx does
  - episode_length_buf, whose setter (rsl_rl's init_at_random_ep_len) reaches the server
  - unwrapped.common_step_counter, saved in checkpoints and restored (or zeroed by
    MICRODUCK_WARM_START) on load, which moves the server's curriculum

ONNX export metadata needs the robot entity and compiled model, which only mjlab has: any
other attribute of ``unwrapped`` comes from a one-duck ManagerBasedRlEnv built on first use.
"""
import copy

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

OBS_FIELDS = {"actor": "obs/policy", "critic": "obs/critic"}


class UnityVecEnv(VecEnv):
    def __init__(self, server, env_cfg, device, clip_actions=None):
        self.server = server
        self.cfg = env_cfg
        self.device = torch.device(device)
        self.num_envs = server.num_envs
        self.num_actions = server.act_dim
        self.step_dt = server.control_dt
        self.max_episode_length = server.episode_steps
        self.max_episode_length_s = self.max_episode_length * self.step_dt
        self.clip_actions = clip_actions
        self.reward_names = list(server.reward_names)
        self.termination_names = list(server.termination_names)
        self._common_step_counter = int(server.header.get("global_step", 0))
        self._episode_sums = torch.zeros(self.num_envs, len(self.reward_names), device=self.device)
        self._episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras = {}
        self._unwrapped = _Unwrapped(self)
        self._obs = self._observations(server.reset())

    # ------------------------------------------------------------------ VecEnv

    @property
    def unwrapped(self):
        return self._unwrapped

    @property
    def episode_length_buf(self):
        return self._episode_length

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        lengths = torch.as_tensor(value).to(torch.int64).clamp(0, self.max_episode_length - 1)
        self.server.set_episode_lengths(lengths.cpu().numpy())
        self._episode_length = lengths.to(self.device)

    def get_observations(self):
        return self._obs

    def reset(self):
        self._obs = self._observations(self.server.reset())
        self._episode_sums.zero_()
        self._episode_length.zero_()
        return self._obs, self.extras

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        out = self.server.step(actions.detach().to("cpu", torch.float32).numpy())
        self._common_step_counter += 1
        self._obs = self._observations(out)
        reward = self._tensor(out["reward"])
        terminated = self._tensor(out["terminated"]).bool()
        time_outs = self._tensor(out["truncated"]).bool()
        dones = terminated | time_outs
        self._episode_sums += self._tensor(out["reward_terms"]) * self.step_dt
        if dones.any():
            self._log_episodes(dones, self._tensor(out["terminations"]).bool())
        self._episode_length = torch.where(dones, 0, self._tensor(out["episode_length"]).long())
        self.extras["time_outs"] = time_outs
        return self._obs, reward, dones.long(), self.extras

    def close(self):
        self._unwrapped.close()
        self.server.close()

    # ------------------------------------------------------------------ helpers

    def _tensor(self, a):
        # Reply arrays are views into a buffer the next call overwrites; .to() copies.
        t = torch.from_numpy(np.asarray(a))
        return t.to(self.device) if self.device.type != "cpu" else t.clone()

    def _observations(self, out):
        return TensorDict({group: self._tensor(out[field]) for group, field in OBS_FIELDS.items()},
                          batch_size=[self.num_envs], device=self.device)

    def _log_episodes(self, dones, hits):
        """mjlab's reward and termination manager reset() logs for the ducks that just reset."""
        log = {}
        ended = self._episode_sums[dones]
        for k, name in enumerate(self.reward_names):
            log["Episode_Reward/" + name] = ended[:, k].mean() / self.max_episode_length_s
        self._episode_sums[dones] = 0.0
        for k, name in enumerate(self.termination_names):
            log["Episode_Termination/" + name] = int(hits[dones, k].sum())
        self.extras["log"] = log


class _Unwrapped:
    """``env.unwrapped`` for the runners: the step counter lives on the server, everything
    else (scene, action manager, sim, ...) on a one-duck mjlab env made when first asked."""

    def __init__(self, env):
        self._env = env
        self._shadow = None

    @property
    def common_step_counter(self):
        return self._env._common_step_counter

    @common_step_counter.setter
    def common_step_counter(self, step):
        self._env._common_step_counter = int(step)
        self._env.server.set_global_step(int(step))

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def step_dt(self):
        return self._env.step_dt

    @property
    def max_episode_length(self):
        return self._env.max_episode_length

    @property
    def max_episode_length_s(self):
        return self._env.max_episode_length_s

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._shadow is None:
            from mjlab.envs import ManagerBasedRlEnv

            cfg = copy.deepcopy(self._env.cfg)
            cfg.scene.num_envs = 1
            self._shadow = ManagerBasedRlEnv(cfg=cfg, device=str(self._env.device))
        return getattr(self._shadow, name)

    def close(self):
        if self._shadow is not None:
            self._shadow.close()
            self._shadow = None
