"""CPU-only tests for the microduck command builder (no Mongo, no mjlab)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from duck import launch  # noqa: E402

RUN = "/microduck_rl/logs/rsl_rl/velocity/2026-11-09_10-00-00_job_abc_0"


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_fresh_train_uses_defaults():
    argv, env = launch.train_command({"_id": "abc", "job_type": "TRAIN"}, attempt=0)
    assert argv[:2] == ["train", launch.DEFAULT_TASK]
    assert flag(argv, "--env.scene.num-envs") == "2048"
    assert flag(argv, "--agent.max-iterations") == "1000"
    assert flag(argv, "--agent.logger") == "tensorboard"
    assert flag(argv, "--agent.run-name") == "job_abc_0"
    assert "--agent.resume" not in argv
    assert env == {}


def test_resume_requests_only_remaining_iterations():
    job = {"_id": "abc", "max_iterations": 300}
    argv, _ = launch.train_command(job, attempt=1, resume_from=f"{RUN}/model_100.pt")
    assert flag(argv, "--agent.max-iterations") == "200"
    assert flag(argv, "--agent.resume") == "True"
    assert flag(argv, "--agent.load-run") == "2026\\-11\\-09_10\\-00\\-00_job_abc_0$"
    assert flag(argv, "--agent.load-checkpoint") == "model_100\\.pt$"
    assert flag(argv, "--agent.run-name") == "job_abc_1"


def test_resume_past_budget_returns_no_command():
    argv, _ = launch.train_command({"_id": "abc", "max_iterations": 100}, 1,
                                   resume_from=f"{RUN}/model_100.pt")
    assert argv is None


def test_init_checkpoint_warm_starts_with_full_budget():
    job = {"_id": "abc", "max_iterations": 500, "init_checkpoint": f"{RUN}/model_900.pt"}
    argv, env = launch.train_command(job, attempt=0)
    assert flag(argv, "--agent.max-iterations") == "500"
    assert flag(argv, "--agent.load-checkpoint") == "model_900\\.pt$"
    assert env == {"MICRODUCK_WARM_START": "1"}


def test_whole_floats_from_the_dashboard_are_accepted():
    argv, _ = launch.train_command({"_id": "abc", "num_envs": 512.0}, attempt=0)
    assert flag(argv, "--env.scene.num-envs") == "512"


def test_warm_started_job_resumes_against_its_offset_budget():
    job = {"_id": "abc", "max_iterations": 500, "init_checkpoint": f"{RUN}/model_900.pt"}
    assert launch.iteration_offset(job) == 900
    argv, _ = launch.train_command(job, attempt=1, resume_from=f"{RUN}/model_1100.pt")
    assert flag(argv, "--agent.max-iterations") == "300"


def test_training_video_flags_convert_iterations_to_env_steps():
    argv, _ = launch.train_command({"_id": "abc", "video": True, "video_every": 100}, attempt=0)
    assert flag(argv, "--video") == "True"
    assert flag(argv, "--video-interval") == str(100 * launch.STEPS_PER_ITERATION)
    assert flag(argv, "--video-length") == str(launch.DEFAULTS["video_seconds"] * launch.CONTROL_HZ)
    assert flag(argv, "--env.viewer.distance") == str(launch.CAMERA["distance"])
    argv, _ = launch.train_command({"_id": "abc"}, attempt=0)
    assert "--video" not in argv


def test_training_videos_are_offset_by_the_resume_iteration(tmp_path):
    clips = tmp_path / "videos" / "train"
    clips.mkdir(parents=True)
    (clips / "rl-video-step-0.mp4").touch()
    (clips / f"rl-video-step-{50 * launch.STEPS_PER_ITERATION}.mp4").touch()
    (clips / "notes.txt").touch()
    assert [it for it, _ in launch.training_videos(tmp_path, 200)] == [200, 250]
    assert launch.training_videos(tmp_path / "missing", 0) == []


def test_pick_evenly_keeps_first_and_last():
    assert launch.pick_evenly(list(range(11)), 3) == [0, 5, 10]
    assert launch.pick_evenly([1, 2], 10) == [1, 2]
    assert launch.pick_evenly([1, 2, 3], 1) == [3]
    assert launch.pick_evenly([], 5) == []


def test_progress_video_command():
    argv = launch.progress_video_command(
        {"task_id": "T"}, [f"{RUN}/model_0.pt", f"{RUN}/model_50.pt"], "/o/p.mp4", "r.py")
    assert argv[:3] == ["python", "r.py", "T"]
    assert flag(argv, "--seconds") == "5"
    assert argv[argv.index("--checkpoints") + 1:] == [f"{RUN}/model_0.pt", f"{RUN}/model_50.pt"]
    with pytest.raises(launch.JobSpecError):
        launch.progress_video_command({}, [], "/o/p.mp4", "r.py")


def test_seed_task_and_extra_args():
    job = {"_id": "abc", "task_id": "Mjlab-Velocity-Rough-Backlash-MicroDuck", "seed": 7,
           "train_args": ["--agent.algorithm.entropy-coef", "0.005"]}
    argv, _ = launch.train_command(job, attempt=0)
    assert argv[1] == "Mjlab-Velocity-Rough-Backlash-MicroDuck"
    assert flag(argv, "--agent.seed") == "7"
    assert argv[-2:] == ["--agent.algorithm.entropy-coef", "0.005"]


@pytest.mark.parametrize("job", [
    {"_id": "abc", "num_envs": 0},
    {"_id": "abc", "max_iterations": "100"},
    {"_id": "abc", "save_interval": True},
    {"_id": "abc", "train_args": "--agent.seed 1"},
    {"_id": "abc", "train_args": ["Mjlab-Spin-Flat-MicroDuck"]},
    {"_id": "abc", "train_args": ["--Bad;flag", "1"]},
    {"_id": "abc", "init_checkpoint": "/tmp/policy.onnx"},
])
def test_bad_job_fields_are_rejected(job):
    with pytest.raises(launch.JobSpecError):
        launch.train_command(job, attempt=0)


def test_export_and_eval_commands():
    ckpt = f"{RUN}/model_250.pt"
    export = launch.export_command({"task_id": "T"}, ckpt, "/out/policy_250.onnx")
    assert export == ["python", "scripts/export.py", "T", "--checkpoint-file", ckpt,
                      "--onnx-file", "/out/policy_250.onnx"]
    ev = launch.eval_command({"eval_steps": 50}, ckpt, "/out/r.json", "evaluate.py")
    assert flag(ev, "--steps") == "50"
    assert flag(ev, "--num-envs") == "512"


def test_latest_checkpoint_spans_run_dirs(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for d, n in ((first, 50), (first, 100), (second, 150), (second, 1000)):
        (d / f"model_{n}.pt").touch()
    (second / "policy_1000.onnx").touch()
    assert launch.latest_checkpoint([first, second, tmp_path / "missing"]) == (
        1000, second / "model_1000.pt")
    assert launch.latest_checkpoint([tmp_path / "missing"]) is None


def test_parse_progress_reads_rsl_rl_output():
    assert launch.parse_progress("\x1b[1m Learning iteration 41/200 \x1b[0m") == {
        "iteration": 41, "tot_iter": 200}
    assert launch.parse_progress("   Mean reward:  88.21") == {"mean_reward": 88.21}
    assert launch.parse_progress("Mean episode length: 866.80") == {"mean_episode_length": 866.8}
    assert launch.parse_progress("Computation: 51234 steps/s") == {}
