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
    train_args       extra tyro flags, e.g. ["--agent.algorithm.entropy-coef", "0.005"]
    checkpoint       model_<N>.pt to EXPORT or EVAL
    eval_num_envs    robots simulated by EVAL
    eval_steps       control steps simulated by EVAL (50 Hz)
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
}

_CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
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
    best = None
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            continue
        for f in run_dir.iterdir():
            match = _CHECKPOINT_RE.match(f.name)
            if match and (best is None or int(match.group(1)) > best[0]):
                best = (int(match.group(1)), f)
    return best


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


def train_command(job, attempt, resume_from=None):
    """``(argv, env)`` for one TRAIN attempt.

    ``resume_from`` is this job's own latest checkpoint after a pause or
    crash; rsl_rl restores its iteration counter, so only the remaining
    iterations are requested. ``init_checkpoint`` instead warm-starts a fresh
    job from another run with the full iteration budget.
    """
    total = positive_int(job, "max_iterations")
    argv = [
        "train", task_id(job),
        "--env.scene.num-envs", str(positive_int(job, "num_envs")),
        "--agent.save-interval", str(positive_int(job, "save_interval")),
        "--agent.logger", "tensorboard",
        "--agent.run-name", run_name(job, attempt),
    ]
    env = {}
    if resume_from is not None:
        remaining = total - checkpoint_iteration(resume_from)
        if remaining <= 0:
            return None, env
        argv += ["--agent.max-iterations", str(remaining)] + _load_flags(resume_from)
    else:
        argv += ["--agent.max-iterations", str(total)]
        if job.get("init_checkpoint"):
            checkpoint_iteration(job["init_checkpoint"])
            argv += _load_flags(job["init_checkpoint"])
            env["MICRODUCK_WARM_START"] = "1"
    if job.get("seed") is not None:
        argv += ["--agent.seed", str(int(job["seed"]))]
    return argv + _extra_args(job), env


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
