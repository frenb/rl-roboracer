"""Checkpoint clips and progression videos for microduck training runs."""
from pathlib import Path
import sys

_RL_AGENT = str(Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from duck import launch  # noqa: E402
from mjlab_core import render_progress as _core  # noqa: E402
from mjlab_core.render_progress import CHECKPOINT_SETTLE_S, WATCH_POLL_S, watch  # noqa: E402,F401


class ClipRenderer(_core.ClipRenderer):
    def __init__(self, task_id, seconds, device):
        super().__init__(
            task_id, seconds, device,
            camera=launch.CAMERA, clip_size=launch.CLIP_SIZE,
            clip_dir=launch.CLIP_DIR, unity_tasks=launch.UNITY_TASKS,
        )


def main():
    _core.run_cli(
        camera=launch.CAMERA, clip_size=launch.CLIP_SIZE, clip_dir=launch.CLIP_DIR,
        default_seconds=launch.DEFAULTS["seconds_per_checkpoint"],
        unity_tasks=launch.UNITY_TASKS,
    )


if __name__ == "__main__":
    main()
