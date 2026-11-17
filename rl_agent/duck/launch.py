"""Builds microduck_rl command lines from job documents (delegates to mjlab_core.launch)."""
from pathlib import Path
import sys

_RL_AGENT = str(Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from mjlab_core import launch as _core  # noqa: E402
from mjlab_core.launch import (  # noqa: E402,F401
    JobSpecError, checkpoint_iteration, checkpoints, iteration_offset,
    latest_checkpoint, match_training_intervals, parse_progress, pick_evenly,
    run_name,
)

DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEFAULTS = {
    "num_envs": 2048,
    "max_iterations": 1000,
    "save_interval": 50,
    "eval_num_envs": 512,
    "eval_steps": 1000,
    "video": True,
    "video_checkpoints": 10,
    "seconds_per_checkpoint": 5,
}
STEPS_PER_ITERATION = 24
CAMERA = {"distance": 1.2, "elevation": -15.0}
CLIP_SIZE = (640, 480)
CLIP_DIR = "clips"
CFG_HOOK = Path(__file__).resolve().parent / "cfg_hook.py"
UNITY_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_unity.py"
UNITY_TASKS = {"Unity-Velocity-Flat-MicroDuck": "Mjlab-Velocity-Flat-MicroDuck"}

CONFIG = _core.LaunchConfig(
    default_task=DEFAULT_TASK,
    defaults=DEFAULTS,
    steps_per_iteration=STEPS_PER_ITERATION,
    clip_dir=CLIP_DIR,
    unity_tasks=UNITY_TASKS,
    train_argv=("train",),
    unity_train_script=UNITY_TRAIN_SCRIPT,
    export_argv=("python", "scripts/export.py"),
    export_ckpt_flag="--checkpoint-file",
    export_onnx_flag="--onnx-file",
    cfg_hook=CFG_HOOK,
    rl_hook_env="MICRODUCK_RL_CFG_HOOK",
    cfg_hook_env="MICRODUCK_CFG_HOOK",
    exp_design_env="MICRODUCK_EXPERIMENT_DESIGN",
    reward_design_env="MICRODUCK_REWARD_DESIGN",
    warm_start_env="MICRODUCK_WARM_START",
)


def setting(job, key):
    return _core.setting(job, key, DEFAULTS)


def task_id(job):
    return _core.task_id(job, CONFIG)


def is_unity(job):
    return _core.is_unity(job, CONFIG)


def mjlab_task_id(job):
    return _core.mjlab_task_id(job, CONFIG)


def steps_per_iteration(job):
    return _core.steps_per_iteration(job, CONFIG)


def design_env(reward_design_file=None, experiment_design_file=None, train=True):
    return _core.design_env(CONFIG, reward_design_file, experiment_design_file, train=train)


def positive_int(job, key):
    return _core.positive_int(job, key, DEFAULTS)


def train_command(job, attempt, resume_from=None, reward_design_file=None,
                  experiment_design_file=None):
    return _core.train_command(job, attempt, CONFIG, resume_from=resume_from,
                               reward_design_file=reward_design_file,
                               experiment_design_file=experiment_design_file)


def clip_path(checkpoint):
    return _core.clip_path(checkpoint, clip_dir=CLIP_DIR)


def checkpoint_clips(run_dirs):
    return _core.checkpoint_clips(run_dirs, clip_dir=CLIP_DIR)


def clip_watch_command(job, run_dir, script):
    return _core.clip_watch_command(job, run_dir, script, CONFIG)


def export_command(job, checkpoint, onnx_path):
    return _core.export_command(job, checkpoint, onnx_path, CONFIG)


def eval_command(job, checkpoint, out_json, script):
    return _core.eval_command(job, checkpoint, out_json, script, CONFIG)


def video_command(job, script, clips=(), progression=(), out_mp4=None):
    return _core.video_command(job, script, CONFIG, clips=clips,
                               progression=progression, out_mp4=out_mp4)
