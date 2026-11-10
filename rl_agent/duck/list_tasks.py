"""Print the registered microduck tasks as JSON, for duck/gyms.py.

Runs inside the duck-trainer venv (imports mjlab), from /microduck_rl:

    python /python_ws/src/duck/list_tasks.py

One object per task: task_id, terrain_type ("plane" or "generator"),
episode_s and the names of its reward terms. A separate process so the
long-running worker never imports torch and warp.
"""
import json
import sys

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.tasks.registry import list_tasks, load_env_cfg


def describe(task_id):
    cfg = load_env_cfg(task_id)
    terrain = getattr(cfg.scene, "terrain", None)
    rewards = cfg.rewards if isinstance(cfg.rewards, dict) else vars(cfg.rewards)
    return {
        "task_id": task_id,
        "terrain_type": getattr(terrain, "terrain_type", None),
        "episode_s": cfg.episode_length_s,
        "reward_terms": sorted(name for name, term in rewards.items() if term is not None),
    }


def main():
    tasks = [describe(t) for t in sorted(list_tasks()) if "MicroDuck" in t]
    json.dump(tasks, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
