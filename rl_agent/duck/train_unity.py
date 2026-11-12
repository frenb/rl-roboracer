"""Train a MicroDuck task in the Unity duck server instead of MuJoCo Warp.

    cd /microduck_rl && python /python_ws/src/duck/train_unity.py Unity-Velocity-Flat-MicroDuck \
        --env.scene.num-envs 2048 --agent.max-iterations 200 --agent.run-name test

Takes the flags of ``train Mjlab-Velocity-Flat-MicroDuck`` and writes the same run directory
(checkpoints, ONNX, params, TensorBoard events): mjlab.scripts.train runs unchanged, except
that where run_train builds ManagerBasedRlEnv and wraps it in RslRlVecEnvWrapper it gets a
UnityVecEnv over a DuckServer. The server's task spec is exported from the env cfg train
resolved, so reward / experiment design hooks and --env flags reach Unity as they would reach
mjlab (anything the C# task doesn't implement fails the export). The run directory also gets
params/unity_task.json and unity_server.log.

DUCK_UNITY_THREADS caps the server's worker threads (default: all CPUs).
"""
import json
import os
import sys

import export_task_spec
import unity_env
import unity_vec_env
from launch import UNITY_TASKS


def install(train):
    """Point mjlab.scripts.train's run_train at the Unity server."""
    log_dirs = []
    run_train = train.run_train

    def unity_run_train(task_id, cfg, log_dir):
        if cfg.video:
            raise SystemExit("--video is not supported with the Unity server (no renderer)")
        log_dirs.append(log_dir)
        return run_train(task_id, cfg, log_dir)

    def make_env(cfg, device, render_mode=None):
        log_dir = log_dirs[-1]
        (log_dir / "params").mkdir(parents=True, exist_ok=True)
        spec = export_task_spec.task_spec(cfg)
        with open(log_dir / "params" / "unity_task.json", "w") as f:
            json.dump(spec, f, indent=1)
        threads = int(os.environ.get("DUCK_UNITY_THREADS", "0")) or None
        server = unity_env.DuckServer(num_envs=cfg.scene.num_envs, seed=max(int(cfg.seed or 0), 0),
                                      threads=threads, task=spec,
                                      log_file=str((log_dir / "unity_server.log").resolve()))
        print(f"[INFO] Unity duck server: {server.num_envs} ducks on {server.header['threads']} threads, "
              f"task {spec['task']}, {len(server.reward_names)} reward terms")
        return unity_vec_env.UnityVecEnv(server, cfg, device)

    def wrap(env, clip_actions=None):
        env.clip_actions = clip_actions
        return env

    train.run_train = unity_run_train
    train.ManagerBasedRlEnv = make_env
    train.RslRlVecEnvWrapper = wrap


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in UNITY_TASKS:
        raise SystemExit(f"usage: {sys.argv[0]} <{' | '.join(UNITY_TASKS)}> [train flags]")
    sys.argv[1] = UNITY_TASKS[sys.argv[1]]
    from mjlab.scripts import train

    install(train)
    train.main()


if __name__ == "__main__":
    main()
