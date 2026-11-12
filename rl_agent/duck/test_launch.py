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


def test_unity_task_trains_in_the_unity_server_and_exports_its_mjlab_twin():
    job = {"_id": "abc", "task_id": "Unity-Velocity-Flat-MicroDuck"}
    argv, _ = launch.train_command(job, attempt=0)
    assert argv[:3] == ["python", str(launch.UNITY_TRAIN_SCRIPT), "Unity-Velocity-Flat-MicroDuck"]
    assert launch.UNITY_TRAIN_SCRIPT.exists()
    assert flag(argv, "--agent.run-name") == "job_abc_0"
    assert launch.export_command(job, f"{RUN}/model_100.pt", "/tmp/p.onnx")[2] == "Mjlab-Velocity-Flat-MicroDuck"
    assert launch.eval_command(job, f"{RUN}/model_100.pt", "/tmp/e.json", "evaluate.py")[2] \
        == "Mjlab-Velocity-Flat-MicroDuck"


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


def test_reward_design_is_handed_to_the_fork_hook():
    _, env = launch.train_command({"_id": "abc"}, 0, reward_design_file="/logs/jobs/abc.json")
    path, _, func = env["MICRODUCK_CFG_HOOK"].rpartition(":")
    assert pathlib.Path(path) == launch.CFG_HOOK and launch.CFG_HOOK.exists() and func == "apply"
    assert env["MICRODUCK_REWARD_DESIGN"] == "/logs/jobs/abc.json"
    _, env = launch.train_command({"_id": "abc"}, 0)
    assert "MICRODUCK_CFG_HOOK" not in env


def test_whole_floats_from_the_dashboard_are_accepted():
    argv, _ = launch.train_command({"_id": "abc", "num_envs": 512.0}, attempt=0)
    assert flag(argv, "--env.scene.num-envs") == "512"


def test_warm_started_job_resumes_against_its_offset_budget():
    job = {"_id": "abc", "max_iterations": 500, "init_checkpoint": f"{RUN}/model_900.pt"}
    assert launch.iteration_offset(job) == 900
    argv, _ = launch.train_command(job, attempt=1, resume_from=f"{RUN}/model_1100.pt")
    assert flag(argv, "--agent.max-iterations") == "300"


def test_training_never_records_video_itself():
    for job in ({"_id": "abc", "video": True}, {"_id": "abc", "task_id": "Unity-Velocity-Flat-MicroDuck"}):
        argv, _ = launch.train_command(job, attempt=0)
        assert "--video" not in argv
    assert launch.setting({}, "video") is True
    assert launch.setting({"video": False}, "video") is False


def test_checkpoint_clips_cover_every_run_directory(tmp_path):
    first, second = tmp_path / "run_0", tmp_path / "run_1"
    for run, its in ((first, (0, 50)), (second, (100,))):
        (run / launch.CLIP_DIR).mkdir(parents=True)
        for it in its:
            (run / f"model_{it}.pt").touch()
    assert launch.clip_path(first / "model_50.pt") == first / "clips" / "clip_50.mp4"
    (first / "clips" / "clip_0.mp4").touch()
    (second / "clips" / "clip_100.mp4").touch()
    (first / "clips" / "clip_50.part.mp4").touch()
    assert launch.checkpoint_clips([first, second, tmp_path / "missing"]) == [
        {"iteration": 0, "path": str(first / "clips" / "clip_0.mp4"),
         "gif": str(first / "clips" / "clip_0.gif")},
        {"iteration": 100, "path": str(second / "clips" / "clip_100.mp4"),
         "gif": str(second / "clips" / "clip_100.gif")},
    ]


def test_clip_watch_command_renders_the_mjlab_twin():
    argv = launch.clip_watch_command({"task_id": "Unity-Velocity-Flat-MicroDuck"}, RUN, "r.py")
    assert argv[:3] == ["python", "r.py", "Mjlab-Velocity-Flat-MicroDuck"]
    assert flag(argv, "--watch") == RUN
    assert flag(argv, "--seconds") == "5"


def test_pick_evenly_keeps_first_and_last():
    assert launch.pick_evenly(list(range(11)), 3) == [0, 5, 10]
    assert launch.pick_evenly([1, 2], 10) == [1, 2]
    assert launch.pick_evenly([1, 2, 3], 1) == [3]
    assert launch.pick_evenly([], 5) == []


def test_video_command():
    a, b, c = f"{RUN}/model_0.pt", f"{RUN}/model_50.pt", f"{RUN}/model_100.pt"
    argv = launch.video_command({"task_id": "T"}, "r.py", clips=[b], progression=[a, c], out_mp4="/o/p.mp4")
    assert argv[:3] == ["python", "r.py", "T"]
    assert flag(argv, "--seconds") == "5"
    assert argv[argv.index("--checkpoints") + 1:argv.index("--out")] == [b]
    assert flag(argv, "--out") == "/o/p.mp4"
    assert argv[argv.index("--progression") + 1:] == [a, c]
    assert "--out" not in launch.video_command({"task_id": "T"}, "r.py", clips=[b])
    with pytest.raises(launch.JobSpecError):
        launch.video_command({}, "r.py")
    with pytest.raises(launch.JobSpecError):
        launch.video_command({}, "r.py", progression=[a])


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


def test_play_config_gets_training_push_timing():
    from types import SimpleNamespace as NS
    play = NS(events={"push_robot": NS(interval_range_s=(0.5, 1.0)), "reset_base": NS(), "gone": None})
    train = NS(events={"push_robot": NS(interval_range_s=(3.0, 6.0)), "reset_base": NS()})
    launch.match_training_intervals(play, train)
    assert play.events["push_robot"].interval_range_s == (3.0, 6.0)
    assert not hasattr(play.events["reset_base"], "interval_range_s")
