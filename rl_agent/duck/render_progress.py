"""Render one video that shows a policy improving across checkpoints.

Runs inside the duck-trainer venv, from /microduck_rl:

    python /python_ws/src/duck/render_progress.py <TASK_ID> --seconds 5 \
        --out progress.mp4 --checkpoints run/model_0.pt run/model_500.pt ...

Each checkpoint gets ``--seconds`` of footage of one robot, labelled with its
iteration. The environment is built once and the weights are swapped in,
so the Warp compile cost is paid once. Writes ``<out>.mp4`` and a smaller
``<out>.gif`` next to it.
"""
import argparse
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duck import launch  # noqa: E402

GIF_WIDTH = 360
GIF_FRAME_STRIDE = 4


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size bitmap font
        return ImageFont.load_default()


def label(frame, text, font):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((12, 10), text, font=font)
    draw.rectangle((left - 6, top - 4, right + 6, bottom + 4), fill=(0, 0, 0))
    draw.text((12, 10), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def iteration_of(path):
    match = re.search(r"model_(\d+)\.pt$", str(path))
    return int(match.group(1)) if match else None


def render(task_id, checkpoints, seconds, out_mp4, device):
    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=True)
    launch.match_training_intervals(env_cfg, load_env_cfg(task_id))
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width, env_cfg.viewer.height = launch.PROGRESS_VIDEO_SIZE
    env_cfg.viewer.distance = launch.CAMERA["distance"]
    env_cfg.viewer.elevation = launch.CAMERA["elevation"]
    agent_cfg = load_rl_cfg(task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task_id) or OnPolicyRunner)(env, asdict(agent_cfg), device=device)

    fps = round(1.0 / raw_env.step_dt)
    steps = int(seconds * fps)
    font = _font(28)
    frames = []
    for checkpoint in checkpoints:
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        text = f"iteration {iteration_of(checkpoint)}"
        started = time.time()
        # Reset inside inference mode too: the env's buffers become inference
        # tensors during the first rollout and can't be updated outside it.
        with torch.inference_mode():
            obs, _ = env.reset()
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                frames.append(label(raw_env.render(), text, font))
        print(f"rendered {text}: {steps} frames in {time.time() - started:.1f}s", flush=True)
    env.close()

    out_mp4 = Path(out_mp4)
    imageio.mimsave(out_mp4, frames, fps=fps, macro_block_size=1)
    scale = GIF_WIDTH / frames[0].shape[1]
    size = (GIF_WIDTH, int(frames[0].shape[0] * scale))
    gif_frames = [np.asarray(Image.fromarray(f).resize(size)) for f in frames[::GIF_FRAME_STRIDE]]
    out_gif = out_mp4.with_suffix(".gif")
    imageio.mimsave(out_gif, gif_frames, duration=1000 * GIF_FRAME_STRIDE / fps, loop=0)
    print(f"Written {out_mp4} and {out_gif}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    render(args.task_id, args.checkpoints, args.seconds, args.out, args.device)


if __name__ == "__main__":
    main()
