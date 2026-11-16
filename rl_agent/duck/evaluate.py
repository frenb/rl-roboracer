"""Score a microduck checkpoint headlessly and write the metrics as JSON.

Runs inside the duck-trainer venv (imports mjlab), from /microduck_rl:

    python /python_ws/src/duck/evaluate.py <TASK_ID> --checkpoint model_N.pt \
        --num-envs 512 --steps 1000 --out result.json

Uses the training configuration, not the play one: microduck's play configs
push the robot every 0.5-1 s instead of every 3-6 s, a stress demo that made a
policy losing 2.5% of robots in 10 s under training conditions lose 26%.
Curricula start at their first stage. An episode is a fall when it terminates
rather than times out (walking off the terrain counts as a time-out).
"""
import argparse
import json
import time
from dataclasses import asdict

import torch
from rsl_rl.runners import OnPolicyRunner

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def evaluate(task_id, checkpoint, num_envs, steps, device):
    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device),
                             clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    returns = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, device=device)
    done_returns, done_lengths = [], []
    falls = timeouts = 0
    start = time.time()
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(steps):
            obs, rew, dones, extras = env.step(policy(obs))
            returns += rew
            lengths += 1
            done = dones.bool()
            if done.any():
                time_outs = extras.get("time_outs")
                timed_out = (time_outs.bool() & done) if time_outs is not None else torch.zeros_like(done)
                timeouts += int(timed_out.sum())
                falls += int((done & ~timed_out).sum())
                done_returns += returns[done].tolist()
                done_lengths += lengths[done].tolist()
                returns[done] = 0
                lengths[done] = 0
    dt = getattr(env.unwrapped, "step_dt", None)
    env.close()

    # Robots still upright at the end count as complete, unfinished episodes.
    all_returns = done_returns + returns.tolist()
    all_lengths = done_lengths + lengths.tolist()
    episodes = len(done_returns) + num_envs
    return {
        "task_id": task_id,
        "checkpoint": str(checkpoint),
        "num_envs": num_envs,
        "steps": steps,
        "control_dt": float(dt) if dt is not None else None,
        "episodes": episodes,
        "falls": falls,
        "timeouts": timeouts,
        "fall_rate": falls / episodes,
        "mean_return": sum(all_returns) / len(all_returns),
        "mean_episode_length": sum(all_lengths) / len(all_lengths),
        "mean_reward_per_step": sum(all_returns) / max(sum(all_lengths), 1),
        "wall_seconds": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(args.task_id, args.checkpoint, args.num_envs, args.steps, args.device)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
