"""CPU-only tests for the OpenBeni command builder (no Mongo, no mjlab)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openbeni import launch  # noqa: E402

RUN = "/openbeni_rl/logs/rsl_rl/beni_velocity/2026-11-18_10-00-00_job_abc_0"


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_fresh_train_runs_the_entry_point_on_the_velocity_task():
    argv, env = launch.train_command({"_id": "abc", "job_type": "TRAIN"}, attempt=0)
    assert argv[:3] == ["python", str(launch.TRAIN_ENTRY), "Beni-Velocity-Flat"]
    assert launch.TRAIN_ENTRY.exists()
    assert flag(argv, "--env.scene.num-envs") == "4096"
    assert flag(argv, "--agent.max-iterations") == "1000"
    assert flag(argv, "--agent.save-interval") == "100"
    assert flag(argv, "--agent.run-name") == "job_abc_0"
    assert "--agent.resume" not in argv
    assert env == {}


def test_jump_warm_start_keeps_its_budget_and_flags_the_warm_start():
    job = {"_id": "abc", "task_id": "Beni-Jump-Flat", "max_iterations": 500,
           "init_checkpoint": f"{RUN}/model_900.pt"}
    argv, env = launch.train_command(job, attempt=0)
    assert argv[2] == "Beni-Jump-Flat"
    assert flag(argv, "--agent.max-iterations") == "500"
    assert flag(argv, "--agent.load-checkpoint") == "model_900\\.pt$"
    assert env == {"OPENBENI_WARM_START": "1"}


def test_resume_requests_only_remaining_iterations():
    argv, _ = launch.train_command({"_id": "abc", "max_iterations": 300}, 1,
                                   resume_from=f"{RUN}/model_100.pt")
    assert flag(argv, "--agent.max-iterations") == "200"
    assert flag(argv, "--agent.run-name") == "job_abc_1"
    assert flag(argv, "--agent.load-run") == "2026\\-11\\-18_10\\-00\\-00_job_abc_0$"


def test_designs_reach_the_hooks_under_openbeni_env_names():
    _, env = launch.train_command({"_id": "abc"}, 0, reward_design_file="/r.json",
                                  experiment_design_file="/e.json")
    path, _, func = env["OPENBENI_CFG_HOOK"].rpartition(":")
    assert pathlib.Path(path) == launch.CFG_HOOK and launch.CFG_HOOK.exists() and func == "apply"
    assert env["OPENBENI_RL_CFG_HOOK"].endswith("cfg_hook.py:apply_agent")
    assert env["OPENBENI_REWARD_DESIGN"] == "/r.json"
    assert env["OPENBENI_EXPERIMENT_DESIGN"] == "/e.json"
    assert not any(k.startswith("MICRODUCK_") for k in env)


def test_loading_a_checkpoint_only_needs_the_agent_hook():
    load = launch.design_env(experiment_design_file="e.json", train=False)
    assert set(load) == {"OPENBENI_RL_CFG_HOOK", "OPENBENI_EXPERIMENT_DESIGN"}


def test_export_and_eval_commands():
    ckpt = f"{RUN}/model_100.pt"
    argv = launch.export_command({"_id": "abc"}, ckpt, "/tmp/p.onnx")
    assert argv[:3] == ["python", str(launch.EXPORT_SCRIPT), "Beni-Velocity-Flat"]
    assert launch.EXPORT_SCRIPT.exists()
    assert flag(argv, "--checkpoint") == ckpt
    assert flag(argv, "--onnx-file") == "/tmp/p.onnx"
    argv = launch.eval_command({"_id": "abc"}, ckpt, "/tmp/e.json", "evaluate.py")
    assert argv[2] == "Beni-Velocity-Flat"
    assert flag(argv, "--num-envs") == "256"
    assert flag(argv, "--steps") == "1000"
