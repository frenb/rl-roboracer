"""Shared mjlab/rsl_rl command-line builders, checkpoint discovery, and progress parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from media import pick_evenly  # noqa: F401

CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
MEAN_REWARD_RE = re.compile(r"Mean (?:total )?reward:\s*(-?[\d.]+(?:e-?\d+)?)")
EPISODE_LENGTH_RE = re.compile(r"Mean episode length:\s*(-?[\d.]+)")
FLAG_RE = re.compile(r"^--[a-z][a-z0-9_.-]*$")


class JobSpecError(ValueError):
    """The job document cannot be turned into a valid command."""


@dataclass(frozen=True)
class LaunchConfig:
    default_task: str
    defaults: dict
    steps_per_iteration: int
    clip_dir: str = "clips"
    unity_tasks: dict = field(default_factory=dict)
    train_argv: tuple = ("train",)
    unity_train_script: Path | None = None
    export_argv: tuple = ("python", "scripts/export.py")
    export_ckpt_flag: str = "--checkpoint-file"
    export_onnx_flag: str = "--onnx-file"
    cfg_hook: Path | None = None
    rl_hook_env: str = "MJLAB_RL_CFG_HOOK"
    cfg_hook_env: str = "MJLAB_CFG_HOOK"
    exp_design_env: str = "MJLAB_EXPERIMENT_DESIGN"
    reward_design_env: str = "MJLAB_REWARD_DESIGN"
    warm_start_env: str = "MJLAB_WARM_START"


def setting(job, key, defaults):
    value = job.get(key)
    return defaults[key] if value is None else value


def task_id(job, cfg: LaunchConfig):
    return job.get("task_id") or cfg.default_task


def is_unity(job, cfg: LaunchConfig):
    return task_id(job, cfg) in cfg.unity_tasks


def mjlab_task_id(job, cfg: LaunchConfig):
    tid = task_id(job, cfg)
    return cfg.unity_tasks.get(tid, tid)


def steps_per_iteration(job, cfg: LaunchConfig):
    return job.get("num_steps_per_env") or cfg.steps_per_iteration


def design_env(cfg: LaunchConfig, reward_design_file=None, experiment_design_file=None, train=True):
    env = {}
    if experiment_design_file is not None and cfg.cfg_hook is not None:
        env[cfg.rl_hook_env] = f"{cfg.cfg_hook}:apply_agent"
        env[cfg.exp_design_env] = str(experiment_design_file)
    if train and reward_design_file is not None:
        env[cfg.reward_design_env] = str(reward_design_file)
    if train and cfg.cfg_hook is not None and (reward_design_file is not None or experiment_design_file is not None):
        env[cfg.cfg_hook_env] = f"{cfg.cfg_hook}:apply"
    return env


def match_training_intervals(play_cfg, train_cfg):
    for name, term in (play_cfg.events or {}).items():
        trained = (train_cfg.events or {}).get(name)
        if term is not None and trained is not None and hasattr(term, "interval_range_s"):
            term.interval_range_s = trained.interval_range_s


def positive_int(job, key, defaults):
    value = setting(job, key, defaults)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobSpecError(f"{key} must be a positive integer, got {value!r}")
    return value


def extra_args(job):
    extra = job.get("train_args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise JobSpecError("train_args must be a list of strings")
    for flag in [a for a in extra if a.startswith("--")]:
        if not FLAG_RE.match(flag.split("=", 1)[0]):
            raise JobSpecError(f"train_args: bad flag {flag!r}")
    if extra and not extra[0].startswith("--"):
        raise JobSpecError(f"train_args must start with a flag, got {extra[0]!r}")
    return list(extra)


def checkpoint_iteration(path):
    match = CHECKPOINT_RE.match(Path(path).name)
    if not match:
        raise JobSpecError(f"not a model_<N>.pt checkpoint: {path}")
    return int(match.group(1))


def checkpoints(run_dirs):
    found = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if run_dir.is_dir():
            for f in run_dir.iterdir():
                match = CHECKPOINT_RE.match(f.name)
                if match:
                    found[int(match.group(1))] = f
    return sorted(found.items())


def latest_checkpoint(run_dirs):
    found = checkpoints(run_dirs)
    return found[-1] if found else None


def run_name(job, attempt):
    return f"job_{job['_id']}_{attempt}"


def load_flags(checkpoint):
    checkpoint = Path(checkpoint)
    return [
        "--agent.resume", "True",
        "--agent.load-run", re.escape(checkpoint.parent.name) + "$",
        "--agent.load-checkpoint", re.escape(checkpoint.name) + "$",
    ]


def iteration_offset(job):
    return checkpoint_iteration(job["init_checkpoint"]) if job.get("init_checkpoint") else 0


def train_command(job, attempt, cfg: LaunchConfig, resume_from=None,
                  reward_design_file=None, experiment_design_file=None):
    total = iteration_offset(job) + positive_int(job, "max_iterations", cfg.defaults)
    if is_unity(job, cfg) and cfg.unity_train_script is not None:
        trainer = ["python", str(cfg.unity_train_script)]
    else:
        trainer = list(cfg.train_argv)
    argv = trainer + [
        task_id(job, cfg),
        "--env.scene.num-envs", str(positive_int(job, "num_envs", cfg.defaults)),
        "--agent.save-interval", str(positive_int(job, "save_interval", cfg.defaults)),
        "--agent.logger", "tensorboard",
        "--agent.run-name", run_name(job, attempt),
    ]
    env = design_env(cfg, reward_design_file, experiment_design_file)
    if resume_from is not None:
        remaining = total - checkpoint_iteration(resume_from)
        if remaining <= 0:
            return None, env
        argv += ["--agent.max-iterations", str(remaining)] + load_flags(resume_from)
    else:
        argv += ["--agent.max-iterations", str(positive_int(job, "max_iterations", cfg.defaults))]
        if job.get("init_checkpoint"):
            argv += load_flags(job["init_checkpoint"])
            env[cfg.warm_start_env] = "1"
    if job.get("seed") is not None:
        argv += ["--agent.seed", str(int(job["seed"]))]
    return argv + extra_args(job), env


def clip_path(checkpoint, clip_dir="clips"):
    checkpoint = Path(checkpoint)
    return checkpoint.parent / clip_dir / f"clip_{checkpoint_iteration(checkpoint)}.mp4"


def checkpoint_clips(run_dirs, clip_dir="clips"):
    clips = []
    for iteration, checkpoint in checkpoints(run_dirs):
        mp4 = clip_path(checkpoint, clip_dir=clip_dir)
        if mp4.exists():
            clips.append({"iteration": iteration, "path": str(mp4), "gif": str(mp4.with_suffix(".gif"))})
    return clips


def clip_watch_command(job, run_dir, script, cfg: LaunchConfig):
    return [
        "python", str(script), mjlab_task_id(job, cfg),
        "--seconds", str(positive_int(job, "seconds_per_checkpoint", cfg.defaults)),
        "--watch", str(run_dir),
    ]


def export_command(job, checkpoint, onnx_path, cfg: LaunchConfig):
    checkpoint_iteration(checkpoint)
    return list(cfg.export_argv) + [
        mjlab_task_id(job, cfg),
        cfg.export_ckpt_flag, str(checkpoint),
        cfg.export_onnx_flag, str(onnx_path),
    ]


def eval_command(job, checkpoint, out_json, script, cfg: LaunchConfig):
    checkpoint_iteration(checkpoint)
    return [
        "python", str(script), mjlab_task_id(job, cfg),
        "--checkpoint", str(checkpoint),
        "--num-envs", str(positive_int(job, "eval_num_envs", cfg.defaults)),
        "--steps", str(positive_int(job, "eval_steps", cfg.defaults)),
        "--out", str(out_json),
    ]


def video_command(job, script, cfg: LaunchConfig, clips=(), progression=(), out_mp4=None):
    if not clips and not progression:
        raise JobSpecError("no checkpoints to render")
    if progression and not out_mp4:
        raise JobSpecError("a progression video needs an output path")
    for path in [*clips, *progression]:
        checkpoint_iteration(path)
    argv = ["python", str(script), mjlab_task_id(job, cfg),
            "--seconds", str(positive_int(job, "seconds_per_checkpoint", cfg.defaults))]
    if clips:
        argv += ["--checkpoints", *[str(p) for p in clips]]
    if progression:
        argv += ["--out", str(out_mp4), "--progression", *[str(p) for p in progression]]
    return argv


def parse_progress(line):
    line = ANSI_RE.sub("", line)
    found = {}
    match = ITERATION_RE.search(line)
    if match:
        found["iteration"] = int(match.group(1))
        found["tot_iter"] = int(match.group(2))
    match = MEAN_REWARD_RE.search(line)
    if match:
        found["mean_reward"] = float(match.group(1))
    match = EPISODE_LENGTH_RE.search(line)
    if match:
        found["mean_episode_length"] = float(match.group(1))
    return found
