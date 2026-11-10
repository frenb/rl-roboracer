"""Builds microduck_rl command lines from job documents.

Pure functions only (no Mongo, no mjlab import) so they are unit-testable on a
CPU-only CI runner.

Job fields read here (all optional except ``job_type``):

    task_id          mjlab task, e.g. "Mjlab-Velocity-Flat-Backlash-MicroDuck"
    num_envs         parallel simulated robots
    max_iterations   total PPO iterations for the job, including resumed ones
    save_interval    iterations between checkpoints; bounds the work lost on pause
    seed             PPO seed
    init_checkpoint  model_<N>.pt to warm-start a fresh TRAIN job from
    reward_design_id reward_designs document to apply (see reward_design.py)
    train_args       extra tyro flags, e.g. ["--agent.algorithm.entropy-coef", "0.005"]
    checkpoint       model_<N>.pt to EXPORT or EVAL
    eval_num_envs    robots simulated by EVAL
    eval_steps       control steps simulated by EVAL (50 Hz)
    video            TRAIN: record a clip of robot 0 every ``video_every`` iterations
    video_every      iterations between training clips
    video_seconds    length of each training clip
    progress_video   TRAIN: render a progression video when training finishes
    video_checkpoints  VIDEO: how many checkpoints, evenly spaced, to show
    seconds_per_checkpoint  VIDEO: footage per checkpoint
"""
import re
from pathlib import Path

DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEFAULTS = {
    "num_envs": 2048,
    "max_iterations": 1000,
    "save_interval": 50,
    "eval_num_envs": 512,
    "eval_steps": 1000,
    "video_every": 50,
    "video_seconds": 6,
    "video_checkpoints": 10,
    "seconds_per_checkpoint": 5,
}
# Every microduck task collects 24 environment steps per PPO iteration and
# runs its policy at 50 Hz; mjlab's video trigger counts environment steps.
STEPS_PER_ITERATION = 24
CONTROL_HZ = 50
# The task cameras sit 3 m away at 320x240, leaving the 30 cm duck a few
# pixels tall. Training clips pause training while they render (~80 ms a
# frame on the WSL2 GPU path), so they stay smaller than progression videos.
CAMERA = {"distance": 1.2, "elevation": -15.0}
TRAIN_CLIP_SIZE = (480, 360)
PROGRESS_VIDEO_SIZE = (640, 480)
CFG_HOOK = Path(__file__).resolve().parent / "cfg_hook.py"

_CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
_VIDEO_RE = re.compile(r"^rl-video-step-(\d+)\.mp4$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
_MEAN_REWARD_RE = re.compile(r"Mean (?:total )?reward:\s*(-?[\d.]+(?:e-?\d+)?)")
_EPISODE_LENGTH_RE = re.compile(r"Mean episode length:\s*(-?[\d.]+)")
# tyro flags are --dotted.kebab-case; anything else could smuggle a positional
# argument (e.g. a different task id) into the command line.
_FLAG_RE = re.compile(r"^--[a-z][a-z0-9_.-]*$")


class JobSpecError(ValueError):
    """The job document cannot be turned into a valid command."""


def setting(job, key):
    value = job.get(key)
    return DEFAULTS[key] if value is None else value


def task_id(job):
    return job.get("task_id") or DEFAULT_TASK


def match_training_intervals(play_cfg, train_cfg):
    """Give a play config the training config's interval-event timing.

    microduck's play configs push the robot every 0.5-1 s instead of every
    3-6 s, so a policy that walks fine in training is shown falling over.
    """
    for name, term in (play_cfg.events or {}).items():
        trained = (train_cfg.events or {}).get(name)
        if term is not None and trained is not None and hasattr(term, "interval_range_s"):
            term.interval_range_s = trained.interval_range_s


def positive_int(job, key):
    value = setting(job, key)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobSpecError(f"{key} must be a positive integer, got {value!r}")
    return value


def _extra_args(job):
    extra = job.get("train_args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise JobSpecError("train_args must be a list of strings")
    flags = [a for a in extra if a.startswith("--")]
    for flag in flags:
        name = flag.split("=", 1)[0]
        if not _FLAG_RE.match(name):
            raise JobSpecError(f"train_args: bad flag {flag!r}")
    if extra and not extra[0].startswith("--"):
        raise JobSpecError(f"train_args must start with a flag, got {extra[0]!r}")
    return list(extra)


def checkpoint_iteration(path):
    match = _CHECKPOINT_RE.match(Path(path).name)
    if not match:
        raise JobSpecError(f"not a model_<N>.pt checkpoint: {path}")
    return int(match.group(1))


def latest_checkpoint(run_dirs):
    """Highest-iteration ``model_<N>.pt`` across ``run_dirs`` as ``(iteration, Path)``."""
    found = checkpoints(run_dirs)
    return found[-1] if found else None


def run_name(job, attempt):
    """Suffix mjlab appends to the timestamped run directory name."""
    return f"job_{job['_id']}_{attempt}"


def _load_flags(checkpoint):
    checkpoint = Path(checkpoint)
    # mjlab resolves --agent.load-run / --agent.load-checkpoint as regexes
    # under logs/rsl_rl/<experiment>/, so anchor them to exact names.
    return [
        "--agent.resume", "True",
        "--agent.load-run", re.escape(checkpoint.parent.name) + "$",
        "--agent.load-checkpoint", re.escape(checkpoint.name) + "$",
    ]


def iteration_offset(job):
    """Iteration the job's counter starts at.

    rsl_rl restores the iteration counter from any loaded checkpoint, so a job
    warm-started from ``model_900.pt`` counts 900, 901, ...
    """
    return checkpoint_iteration(job["init_checkpoint"]) if job.get("init_checkpoint") else 0


def train_command(job, attempt, resume_from=None, reward_design_file=None):
    """``(argv, env)`` for one TRAIN attempt.

    ``resume_from`` is this job's own latest checkpoint after a pause or
    crash; only the remaining iterations are requested. ``init_checkpoint``
    instead warm-starts a fresh job from another run with the full budget.
    ``reward_design_file`` is a reward design JSON that ``cfg_hook.py``
    applies to the task's reward terms inside the trainer.
    """
    total = iteration_offset(job) + positive_int(job, "max_iterations")
    argv = [
        "train", task_id(job),
        "--env.scene.num-envs", str(positive_int(job, "num_envs")),
        "--agent.save-interval", str(positive_int(job, "save_interval")),
        "--agent.logger", "tensorboard",
        "--agent.run-name", run_name(job, attempt),
    ]
    env = {}
    if reward_design_file is not None:
        env["MICRODUCK_CFG_HOOK"] = f"{CFG_HOOK}:apply"
        env["MICRODUCK_REWARD_DESIGN"] = str(reward_design_file)
    if resume_from is not None:
        remaining = total - checkpoint_iteration(resume_from)
        if remaining <= 0:
            return None, env
        argv += ["--agent.max-iterations", str(remaining)] + _load_flags(resume_from)
    else:
        argv += ["--agent.max-iterations", str(positive_int(job, "max_iterations"))]
        if job.get("init_checkpoint"):
            argv += _load_flags(job["init_checkpoint"])
            env["MICRODUCK_WARM_START"] = "1"
    if job.get("seed") is not None:
        argv += ["--agent.seed", str(int(job["seed"]))]
    if job.get("video"):
        argv += [
            "--video", "True",
            "--video-interval", str(positive_int(job, "video_every") * STEPS_PER_ITERATION),
            "--video-length", str(positive_int(job, "video_seconds") * CONTROL_HZ),
            "--env.viewer.width", str(TRAIN_CLIP_SIZE[0]),
            "--env.viewer.height", str(TRAIN_CLIP_SIZE[1]),
            "--env.viewer.distance", str(CAMERA["distance"]),
            "--env.viewer.elevation", str(CAMERA["elevation"]),
        ]
    return argv + _extra_args(job), env


def checkpoints(run_dirs):
    """Every ``(iteration, Path)`` checkpoint across ``run_dirs``, oldest first."""
    found = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if run_dir.is_dir():
            for f in run_dir.iterdir():
                match = _CHECKPOINT_RE.match(f.name)
                if match:
                    found[int(match.group(1))] = f
    return sorted(found.items())


def pick_evenly(items, count):
    """Up to ``count`` items spread evenly, always keeping the first and last."""
    if count <= 1 or len(items) <= count:
        return list(items) if count > 1 else list(items[-1:])
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def training_videos(run_dir, start_iteration):
    """``(iteration, Path)`` for clips mjlab recorded in one run directory.

    Clip names count environment steps from the start of that run, so a
    resumed run is offset by the iteration it resumed from.
    """
    folder = Path(run_dir) / "videos" / "train"
    if not folder.is_dir():
        return []
    clips = []
    for f in folder.iterdir():
        match = _VIDEO_RE.match(f.name)
        if match:
            clips.append((start_iteration + int(match.group(1)) // STEPS_PER_ITERATION, f))
    return sorted(clips)


def export_command(job, checkpoint, onnx_path):
    checkpoint_iteration(checkpoint)
    return [
        "python", "scripts/export.py", task_id(job),
        "--checkpoint-file", str(checkpoint),
        "--onnx-file", str(onnx_path),
    ]


def eval_command(job, checkpoint, out_json, script):
    checkpoint_iteration(checkpoint)
    return [
        "python", str(script), task_id(job),
        "--checkpoint", str(checkpoint),
        "--num-envs", str(positive_int(job, "eval_num_envs")),
        "--steps", str(positive_int(job, "eval_steps")),
        "--out", str(out_json),
    ]


def progress_video_command(job, checkpoint_paths, out_mp4, script):
    if not checkpoint_paths:
        raise JobSpecError("no checkpoints to render")
    for path in checkpoint_paths:
        checkpoint_iteration(path)
    return [
        "python", str(script), task_id(job),
        "--seconds", str(positive_int(job, "seconds_per_checkpoint")),
        "--out", str(out_mp4),
        "--checkpoints", *[str(p) for p in checkpoint_paths],
    ]


def parse_progress(line):
    """Fields found in one line of rsl_rl console output."""
    line = _ANSI_RE.sub("", line)
    found = {}
    match = _ITERATION_RE.search(line)
    if match:
        found["iteration"] = int(match.group(1))
        found["tot_iter"] = int(match.group(2))
    match = _MEAN_REWARD_RE.search(line)
    if match:
        found["mean_reward"] = float(match.group(1))
    match = _EPISODE_LENGTH_RE.search(line)
    if match:
        found["mean_episode_length"] = float(match.group(1))
    return found
