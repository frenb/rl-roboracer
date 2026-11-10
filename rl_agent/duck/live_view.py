"""Live 3D view of microduck training, served on :8080 (host :8090) with viser.

Runs inside the duck-trainer container, from /microduck_rl:

    python /python_ws/src/duck/live_view.py [--job-id ID] [--num-envs 2]

Follows the running microduck TRAIN job (or, when none is running, the most
recent one with a checkpoint): every new checkpoint it saves is loaded as it
appears, so the robots on screen improve as PPO does. The checkpoint picker
lists every checkpoint across the job's resumed attempts; untick "Follow
newest checkpoint" to stay on an older one.

When another job becomes the one to follow, the process exits so
scripts/Watch-Duck.ps1's supervisor restarts it on that job; the browser
window reconnects to the same address by itself.
"""
import argparse
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch
import viser
from bson import ObjectId
from pymongo import MongoClient

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import job_queue  # noqa: E402
from duck import launch  # noqa: E402

POLL_S = 10
EXIT_JOB_CHANGED = 3
FRAME_RATE = 30.0
IDLE_SLEEP_S = 0.25


class LiveViewer(ViserPlayViewer):
    """Play viewer that stops simulating while no browser is connected.

    The view shares the GPU with training, so with nobody watching it only
    drains queued actions (job switches, checkpoint loads) and sleeps.
    """

    def tick(self):
        if self._server.get_clients():
            return super().tick()
        self._process_actions()
        self._last_tick_time = time.perf_counter()
        time.sleep(IDLE_SLEEP_S)
        return False

db = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))[
    os.environ.get("DATABASE_NAME", "robotaxi")]


def followed_job(job_id):
    """The job to show: pinned, else the running TRAIN job, else the latest one."""
    if job_id:
        return db.jobs.find_one({"_id": ObjectId(job_id) if ObjectId.is_valid(job_id) else job_id})
    base = {"robot": "microduck", "job_type": "TRAIN", "duck_checkpoint": {"$ne": None}}
    return (db.jobs.find_one({**base, "status": "IN_PROGRESS"})
            or max(db.jobs.find(base), key=job_queue.created_at, default=None))


def job_checkpoints(job):
    """``{"model_N.pt": Path}`` across every run directory of ``job``."""
    return {p.name: p for _, p in launch.checkpoints(job.get("duck_run_dirs") or [])}


def caption(job):
    reward = job.get("mean_reward")
    return (f"**{job.get('name') or 'microduck job'}** ({job.get('status')})  \n"
            f"`{job['_id']}` · {launch.task_id(job)}  \n"
            f"iteration {job.get('duck_iteration', 0)}"
            f" of {launch.iteration_offset(job) + launch.positive_int(job, 'max_iterations')}"
            + (f" · mean reward {reward:.2f}" if reward is not None else ""))


def wait_for_job(server, job_id):
    note = server.gui.add_markdown("Waiting for a microduck training job with a checkpoint...")
    while (job := followed_job(job_id)) is None or not job_checkpoints(job):
        time.sleep(POLL_S)
    note.remove()
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label="microduck")
    job = wait_for_job(server, args.job_id)
    task = launch.task_id(job)
    info = server.gui.add_markdown(caption(job))
    follow = server.gui.add_checkbox("Follow newest checkpoint", initial_value=True)

    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device),
                             clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)

    state = {"job": job, "paths": job_checkpoints(job)}

    def load(name):
        runner.load(str(state["paths"][name]), load_cfg={"actor": True}, strict=True,
                    map_location=device)
        return runner.get_inference_policy(device=device)

    def available():
        now = time.time()
        return [(name, format_time_ago(int(now - path.stat().st_mtime)))
                for name, path in sorted(state["paths"].items(),
                                         key=lambda kv: launch.checkpoint_iteration(kv[0]))]

    newest = max(state["paths"], key=launch.checkpoint_iteration)
    manager = CheckpointManager(current_name=newest, fetch_available=available,
                                load_checkpoint=load, run_name=str(job["_id"]))
    viewer = LiveViewer(env, load(newest), frame_rate=FRAME_RATE, viser_server=server,
                        checkpoint_manager=manager)
    exit_code = {"value": 0}

    def watch():
        while True:
            time.sleep(POLL_S)
            try:
                current = followed_job(args.job_id)
            except Exception as exc:  # noqa: BLE001 - a Mongo blip must not end the view
                print(f"job lookup failed: {exc}", flush=True)
                continue
            if current is None:
                continue
            if current["_id"] != state["job"]["_id"]:
                print(f"[live-view] now following job {current['_id']}; restarting", flush=True)
                exit_code["value"] = EXIT_JOB_CHANGED
                viewer._interrupted = True
                return
            state["job"] = current
            state["paths"] = job_checkpoints(current)
            info.content = caption(current)
            latest = max(state["paths"], key=launch.checkpoint_iteration, default=None)
            if follow.value and latest and latest != manager.current_name:
                viewer.request_action("FETCH_CHECKPOINT", "latest")

    threading.Thread(target=watch, daemon=True).start()
    print(f"[live-view] following job {job['_id']} ({task}) from {newest}", flush=True)
    viewer.run()
    env.close()
    server.stop()
    sys.exit(exit_code["value"])


if __name__ == "__main__":
    main()
