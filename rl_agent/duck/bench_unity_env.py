"""Throughput of the Unity duck env server on the exported task, through unity_env.DuckServer.

    python bench_unity_env.py [--envs 4096] [--steps 300] [--threads N]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from duck import unity_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--action-std", type=float, default=0.25)
    args = ap.parse_args()

    t0 = time.perf_counter()
    with unity_env.DuckServer(num_envs=args.envs, seed=1, threads=args.threads) as env:
        start = time.perf_counter() - t0
        env.reset()
        rng = np.random.default_rng(0)
        actions = rng.normal(0, args.action_std, (32, env.num_envs, env.act_dim)).astype(np.float32)
        for k in range(10):
            env.step(actions[k % 32])
        times, ended, terms = [], 0, []
        t1 = time.perf_counter()
        for k in range(args.steps):
            t = time.perf_counter()
            out = env.step(actions[k % 32])
            times.append(time.perf_counter() - t)
            ended += int(out["terminated"].sum() + out["truncated"].sum())
            terms.append(out["reward_terms"].mean(0))
        wall = time.perf_counter() - t1
        for key in ("obs/policy", "obs/critic", "reward_terms"):
            if not np.isfinite(out[key]).all():
                raise RuntimeError(f"non-finite {key}")
        ms = np.array(times) * 1e3
        print(json.dumps({
            "num_envs": env.num_envs, "threads": env.header["threads"],
            "launch_to_connected_s": round(start, 1),
            "step_ms": {"p50": round(float(np.percentile(ms, 50)), 1), "p95": round(float(np.percentile(ms, 95)), 1),
                        "max": round(float(ms.max()), 1)},
            "duck_steps_per_s": round(args.steps * env.num_envs / wall),
            "episodes_ended": ended,
            "mean_reward_terms": dict(zip(env.reward_names, np.round(np.mean(terms, 0), 4).tolist())),
        }, indent=1))


if __name__ == "__main__":
    main()
