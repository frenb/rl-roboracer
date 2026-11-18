"""Runs ``robot="openbeni"`` jobs from the shared ``jobs`` queue.

Configured wrapper around ``mjlab_core.worker.MjlabWorker``.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mjlab_core import worker as _core  # noqa: E402
from mjlab_core.worker import DESIGN_FIELDS, EXPERIMENT_FIELDS, validate_onnx  # noqa: E402,F401
from openbeni import experiment_design, gyms, launch, reward_design  # noqa: E402

ROBOT = "openbeni"
WORKDIR = pathlib.Path(os.environ.get("OPENBENI_RL_DIR", "/openbeni_rl"))
SHARED_LOG = pathlib.Path(os.environ.get("BENI_SHARED_LOG", "/shared_tmp/beni_trainer.log"))
EVALUATE_SCRIPT = pathlib.Path(__file__).resolve().parent / "evaluate.py"
RENDER_PROGRESS_SCRIPT = pathlib.Path(__file__).resolve().parent / "render_progress.py"
LIST_TASKS_SCRIPT = pathlib.Path(__file__).resolve().parent / "list_tasks.py"
DEPLOY_SHAPE = {"Beni-": (40, 7)}

CONFIG = _core.MjlabRobotConfig(
    robot=ROBOT,
    workdir=WORKDIR,
    shared_log=SHARED_LOG,
    launch_cfg=launch.CONFIG,
    evaluate_script=EVALUATE_SCRIPT,
    render_progress_script=RENDER_PROGRESS_SCRIPT,
    list_tasks_script=LIST_TASKS_SCRIPT,
    gym_id_fn=gyms.gym_id,
    task_for_job_fn=gyms.task_for_job,
    sync_gyms_fn=gyms.sync,
    lint_reward_fn=reward_design.lint,
    seed_reward_fn=reward_design.seed,
    validate_experiment_fn=experiment_design.validate,
    effective_job_fn=experiment_design.effective_job,
    check_warm_start_fn=experiment_design.check_warm_start,
    seed_experiment_fn=experiment_design.seed,
    experiment_error_cls=experiment_design.ExperimentDesignError,
    deploy_shapes=DEPLOY_SHAPE,
    run_dirs_field="openbeni_run_dirs",
    checkpoint_field="openbeni_checkpoint",
    iteration_field="openbeni_iteration",
    log_prefix="beni-worker",
)


def main():
    _core.MjlabWorker(CONFIG).run_forever()


if __name__ == "__main__":
    main()
