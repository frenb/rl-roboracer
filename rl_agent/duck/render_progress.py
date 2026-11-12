"""Checkpoint clips and progression videos for microduck training runs.

Every checkpoint gets one clip: ``--seconds`` of one robot running the policy in
the mjlab task (a Unity task's mjlab twin), labelled with its iteration, saved as
``<run dir>/clips/clip_<N>.mp4`` plus a smaller ``.gif``. A progression video
stitches the clips of several checkpoints, so jobs trained in either simulator
get the same videos. Runs inside the duck-trainer venv, from /microduck_rl:

    # beside a training run: a clip for each checkpoint it saves, until killed
    python /python_ws/src/duck/render_progress.py <TASK_ID> --watch <RUN_DIR>

    # clips for --checkpoints, and --progression's clips stitched into one video
    # (<out>.mp4 and <out>.gif); only missing clips are rendered
    python /python_ws/src/duck/render_progress.py <TASK_ID> [--checkpoints run/model_50.pt ...] \
        [--out progress.mp4 --progression run/model_0.pt run/model_500.pt ...]

The environment is built once per process, on the first clip it renders, and
the weights are swapped in, so the Warp compile cost is paid once.
"""
import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duck import launch  # noqa: E402

GIF_WIDTH = 360
GIF_FRAME_STRIDE = 4
WATCH_POLL_S = 5
# rsl_rl writes a checkpoint in one torch.save; give it time to finish.
CHECKPOINT_SETTLE_S = 5


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


def gif_frame(frame):
    scale = GIF_WIDTH / frame.shape[1]
    return np.asarray(Image.fromarray(frame).resize((GIF_WIDTH, int(frame.shape[0] * scale))))


class Media:
    """Streams frames into ``<out>.mp4`` and a subsampled ``<out>.gif``.

    Both are written under temporary names and renamed when complete, the MP4
    last, so an existing MP4 always means a finished pair.
    """

    def __init__(self, out_mp4, fps):
        self.mp4 = Path(out_mp4)
        self.gif = self.mp4.with_suffix(".gif")
        self.fps = fps
        self.mp4.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_mp4 = self.mp4.with_name(self.mp4.stem + ".part.mp4")
        self._writer = imageio.get_writer(self._tmp_mp4, fps=fps, macro_block_size=1)
        self._gif_frames = []
        self._count = 0

    def add(self, frame):
        self._writer.append_data(frame)
        if self._count % GIF_FRAME_STRIDE == 0:
            self._gif_frames.append(gif_frame(frame))
        self._count += 1

    def close(self):
        self._writer.close()
        tmp_gif = self.gif.with_name(self.gif.stem + ".part.gif")
        imageio.mimsave(tmp_gif, self._gif_frames, duration=1000 * GIF_FRAME_STRIDE / self.fps, loop=0)
        os.replace(tmp_gif, self.gif)
        os.replace(self._tmp_mp4, self.mp4)


class ClipRenderer:
    def __init__(self, task_id, seconds, device):
        self.task_id = launch.UNITY_TASKS.get(task_id, task_id)
        self.seconds = seconds
        self.device = device
        self._env = None

    def _build(self):
        from rsl_rl.runners import OnPolicyRunner

        import mjlab.tasks  # noqa: F401  (populates the task registry)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        env_cfg = load_env_cfg(self.task_id, play=True)
        launch.match_training_intervals(env_cfg, load_env_cfg(self.task_id))
        env_cfg.scene.num_envs = 1
        env_cfg.viewer.width, env_cfg.viewer.height = launch.CLIP_SIZE
        env_cfg.viewer.distance = launch.CAMERA["distance"]
        env_cfg.viewer.elevation = launch.CAMERA["elevation"]
        agent_cfg = load_rl_cfg(self.task_id)
        self._raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device, render_mode="rgb_array")
        self._env = RslRlVecEnvWrapper(self._raw_env, clip_actions=agent_cfg.clip_actions)
        self._runner = (load_runner_cls(self.task_id) or OnPolicyRunner)(
            self._env, asdict(agent_cfg), device=self.device)
        self.fps = round(1.0 / self._raw_env.step_dt)
        self._font = _font(28)

    def clip(self, checkpoint):
        """The clip of ``checkpoint``, rendering it if it doesn't exist yet."""
        out = launch.clip_path(checkpoint)
        if out.exists():
            return out
        if self._env is None:
            self._build()
        self._runner.load(str(checkpoint), map_location=self.device)
        policy = self._runner.get_inference_policy(device=self.device)
        text = f"iteration {launch.checkpoint_iteration(checkpoint)}"
        started = time.time()
        media = Media(out, self.fps)
        steps = int(self.seconds * self.fps)
        # Reset inside inference mode too: the env's buffers become inference
        # tensors during the first rollout and can't be updated outside it.
        with torch.inference_mode():
            obs, _ = self._env.reset()
            for _ in range(steps):
                obs, _, _, _ = self._env.step(policy(obs))
                media.add(label(self._raw_env.render(), text, self._font))
        media.close()
        print(f"rendered {out}: {steps} frames in {time.time() - started:.1f}s", flush=True)
        return out

    def close(self):
        if self._env is not None:
            self._env.close()


def stitch(clips, out_mp4):
    """One video of ``clips`` back to back, with its GIF."""
    reader = imageio.get_reader(clips[0])
    fps = reader.get_meta_data()["fps"]
    reader.close()
    media = Media(out_mp4, fps)
    for clip in clips:
        reader = imageio.get_reader(clip)
        for frame in reader:
            media.add(frame)
        reader.close()
    media.close()
    print(f"Written {media.mp4} and {media.gif} from {len(clips)} clips", flush=True)


def watch(renderer, run_dir):
    """Clip each checkpoint ``run_dir`` gets, oldest first, until killed or orphaned."""
    failed = set()
    while os.getppid() != 1:
        for _, checkpoint in launch.checkpoints([run_dir]):
            if checkpoint in failed or launch.clip_path(checkpoint).exists():
                continue
            if time.time() - checkpoint.stat().st_mtime < CHECKPOINT_SETTLE_S:
                continue
            try:
                renderer.clip(checkpoint)
            except Exception as exc:  # noqa: BLE001 - one bad checkpoint must not end the watch
                print(f"clip of {checkpoint} failed: {exc!r}", flush=True)
                failed.add(checkpoint)
        time.sleep(WATCH_POLL_S)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_id")
    parser.add_argument("--watch", type=Path, metavar="RUN_DIR")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=[])
    parser.add_argument("--progression", nargs="+", type=Path, default=[])
    parser.add_argument("--out", help="the progression video")
    parser.add_argument("--seconds", type=float, default=launch.DEFAULTS["seconds_per_checkpoint"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.out and args.checkpoints and not args.progression:
        # The older command line: --out <video> --checkpoints <stitched ones>.
        args.progression, args.checkpoints = args.checkpoints, []
    if bool(args.watch) == bool(args.checkpoints or args.progression):
        parser.error("give --watch, or --checkpoints and/or --progression")
    if bool(args.progression) != bool(args.out):
        parser.error("--progression and --out go together")
    renderer = ClipRenderer(args.task_id, args.seconds, args.device)
    try:
        if args.watch:
            watch(renderer, args.watch)
        for checkpoint in args.checkpoints:
            renderer.clip(checkpoint)
        if args.progression:
            stitch([renderer.clip(c) for c in args.progression], args.out)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
