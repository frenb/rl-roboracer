"""Shared checkpoint clip & progression video renderer for mjlab + rsl_rl robots."""
import argparse
import importlib
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import Media, font, label, stitch  # noqa: E402
from mjlab_core import launch as core_launch  # noqa: E402

WATCH_POLL_S = 5
CHECKPOINT_SETTLE_S = 5


class ClipRenderer:
    def __init__(self, task_id, seconds, device, *, camera=None, clip_size=(640, 480),
                 clip_dir="clips", unity_tasks=None, register_modules=()):
        self.task_id = (unity_tasks or {}).get(task_id, task_id)
        self.seconds = seconds
        self.device = device
        self.camera = camera or {"distance": 1.2, "elevation": -15.0}
        self.clip_size = clip_size
        self.clip_dir = clip_dir
        self.register_modules = tuple(register_modules)
        self._env = None

    def _build(self):
        from rsl_rl.runners import OnPolicyRunner

        for mod in self.register_modules:
            importlib.import_module(mod)
        import mjlab.tasks  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        env_cfg = load_env_cfg(self.task_id, play=True)
        core_launch.match_training_intervals(env_cfg, load_env_cfg(self.task_id))
        env_cfg.scene.num_envs = 1
        env_cfg.viewer.width, env_cfg.viewer.height = self.clip_size
        env_cfg.viewer.distance = self.camera["distance"]
        env_cfg.viewer.elevation = self.camera["elevation"]
        agent_cfg = load_rl_cfg(self.task_id)
        self._raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device, render_mode="rgb_array")
        self._env = RslRlVecEnvWrapper(self._raw_env, clip_actions=agent_cfg.clip_actions)
        self._runner = (load_runner_cls(self.task_id) or OnPolicyRunner)(
            self._env, asdict(agent_cfg), device=self.device)
        self.fps = round(1.0 / self._raw_env.step_dt)
        self._font = font(28)

    def clip(self, checkpoint):
        out = core_launch.clip_path(checkpoint, clip_dir=self.clip_dir)
        if out.exists():
            return out
        if self._env is None:
            self._build()
        self._runner.load(str(checkpoint), map_location=self.device)
        policy = self._runner.get_inference_policy(device=self.device)
        text = f"iteration {core_launch.checkpoint_iteration(checkpoint)}"
        started = time.time()
        media = Media(out, self.fps)
        steps = int(self.seconds * self.fps)
        with torch.inference_mode():
            obs, _ = self._env.reset()
            for _ in range(steps):
                obs, _, dones, _ = self._env.step(policy(obs))
                if hasattr(policy, "reset") and dones.bool().any():
                    policy.reset(dones.bool())
                media.add(label(self._raw_env.render(), text, self._font))
        media.close()
        print(f"rendered {out}: {steps} frames in {time.time() - started:.1f}s", flush=True)
        return out

    def close(self):
        if self._env is not None:
            self._env.close()


def watch(renderer, run_dir, clip_dir="clips"):
    failed = set()
    while os.getppid() != 1:
        for _, checkpoint in core_launch.checkpoints([run_dir]):
            if checkpoint in failed or core_launch.clip_path(checkpoint, clip_dir=clip_dir).exists():
                continue
            if time.time() - checkpoint.stat().st_mtime < CHECKPOINT_SETTLE_S:
                continue
            try:
                renderer.clip(checkpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"clip of {checkpoint} failed: {exc!r}", flush=True)
                failed.add(checkpoint)
        time.sleep(WATCH_POLL_S)


def run_cli(camera=None, clip_size=(640, 480), clip_dir="clips",
            default_seconds=5, unity_tasks=None, register_modules=()):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_id")
    parser.add_argument("--watch", type=Path, metavar="RUN_DIR")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=[])
    parser.add_argument("--progression", nargs="+", type=Path, default=[])
    parser.add_argument("--out", help="the progression video")
    parser.add_argument("--seconds", type=float, default=default_seconds)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.out and args.checkpoints and not args.progression:
        args.progression, args.checkpoints = args.checkpoints, []
    if bool(args.watch) == bool(args.checkpoints or args.progression):
        parser.error("give --watch, or --checkpoints and/or --progression")
    if bool(args.progression) != bool(args.out):
        parser.error("--progression and --out go together")
    renderer = ClipRenderer(args.task_id, args.seconds, args.device,
                            camera=camera, clip_size=clip_size, clip_dir=clip_dir,
                            unity_tasks=unity_tasks, register_modules=register_modules)
    try:
        if args.watch:
            watch(renderer, args.watch, clip_dir=clip_dir)
        for checkpoint in args.checkpoints:
            renderer.clip(checkpoint)
        if args.progression:
            stitch([renderer.clip(c) for c in args.progression], args.out)
    finally:
        renderer.close()


if __name__ == "__main__":
    run_cli()
