"""Shared queue worker for mjlab + rsl_rl robots (e.g. microduck, openbeni)."""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Callable, Optional

from pymongo import MongoClient

_RL_AGENT = pathlib.Path(__file__).resolve().parents[1]
if str(_RL_AGENT) not in sys.path:
    sys.path.insert(0, str(_RL_AGENT))

import job_queue  # noqa: E402
from mjlab_core import launch as core_launch  # noqa: E402
from specs import spec as reward_spec  # noqa: E402

DESIGN_FIELDS = ("reward_design_id", "reward_design_name", "reward_design_version", "reward_design_code")
EXPERIMENT_FIELDS = ("experiment_design_id", "experiment_design_name", "experiment_design_version",
                     "experiment_design_fields")


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def _maybe_object_id(value):
    from bson import ObjectId
    try:
        return ObjectId(str(value))
    except Exception:  # noqa: BLE001
        return value


def validate_onnx(task: str, onnx_path: pathlib.Path, deploy_shapes: dict | None = None) -> dict:
    """Shapes, metadata, and finite forward pass for feedforward or recurrent ONNX policies."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    feed = {}
    for inp in inputs:
        shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=np.float32)
    obs_dim = int(inputs[0].shape[-1])
    act_dim = int(outputs[0].shape[-1])
    action = session.run(None, feed)[0]
    report = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "recurrent": len(inputs) > 1,
        "inputs": [i.name for i in inputs],
        "outputs": [o.name for o in outputs],
        "finite_on_zero_obs": bool(np.isfinite(action).all()),
        "metadata": {p.key: p.value for p in model.metadata_props},
    }
    problems = []
    if not report["finite_on_zero_obs"]:
        problems.append("non-finite action for a zero observation")
    for prefix, shape in (deploy_shapes or {}).items():
        if task.startswith(prefix) and (obs_dim, act_dim) != tuple(shape):
            problems.append(f"expected {shape[0]} inputs / {shape[1]} outputs, got {obs_dim} / {act_dim}")
    report["ok"] = not problems
    report["problems"] = problems
    return report


@dataclass
class MjlabRobotConfig:
    robot: str
    workdir: pathlib.Path
    shared_log: pathlib.Path
    launch_cfg: core_launch.LaunchConfig
    evaluate_script: pathlib.Path
    render_progress_script: pathlib.Path
    list_tasks_script: pathlib.Path
    gym_id_fn: Callable[[str], str]
    task_for_job_fn: Callable[[object, dict], Optional[str]]
    sync_gyms_fn: Callable[[object, list], tuple]
    lint_reward_fn: Callable[[str, Optional[list], Optional[list]], dict]
    seed_reward_fn: Callable[[object, list, datetime.datetime], None]
    validate_experiment_fn: Callable[[dict], dict]
    effective_job_fn: Callable[[dict], dict]
    check_warm_start_fn: Callable[[dict, dict], None]
    seed_experiment_fn: Callable[[object, datetime.datetime], None]
    experiment_error_cls: type = ValueError
    deploy_shapes: dict = field(default_factory=dict)
    run_dirs_field: str = "run_dirs"
    checkpoint_field: str = "checkpoint_path"
    iteration_field: str = "training_steps"
    poll_s: float = 5.0
    status_poll_s: float = 2.0
    progress_write_s: float = 5.0
    stop_grace_s: float = 30.0
    log_prefix: Optional[str] = None

    @property
    def log_root(self) -> pathlib.Path:
        return self.workdir / "logs"

    @property
    def job_log_dir(self) -> pathlib.Path:
        return self.log_root / "jobs"

    @property
    def tag(self) -> str:
        return self.log_prefix or f"{self.robot}-worker"


class JobLog:
    """Tees subprocess output to the job's own log file and the shared panel log."""

    def __init__(self, job_log_dir: pathlib.Path, shared_log: pathlib.Path, job_id):
        job_log_dir.mkdir(parents=True, exist_ok=True)
        self.path = job_log_dir / f"{job_id}.log"
        self._files = [open(self.path, "a", buffering=1)]
        try:
            shared_log.parent.mkdir(parents=True, exist_ok=True)
            self._files.append(open(shared_log, "w", buffering=1))
        except OSError as exc:
            print(f"shared log unavailable: {exc}", flush=True)

    def write(self, line: str):
        sys.stdout.write(line)
        for f in self._files:
            f.write(line)

    def close(self):
        for f in self._files:
            f.close()


class MjlabWorker:
    """Drains one mjlab robot's slice of the shared ``jobs`` queue."""

    def __init__(self, cfg: MjlabRobotConfig, db=None, stopping: threading.Event | None = None):
        self.cfg = cfg
        self.stopping = stopping or threading.Event()
        if db is None:
            db = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))[
                os.environ.get("DATABASE_NAME", "robotaxi")]
        self.db = db

    def update_job(self, job_id, **fields):
        self.db.jobs.update_one({"_id": job_id}, {"$set": fields})

    def job_status(self, job_id) -> str:
        try:
            doc = self.db.jobs.find_one({"_id": job_id}, {"status": 1})
        except Exception as exc:  # noqa: BLE001
            print(f"status lookup failed ({exc}); continuing", flush=True)
            return "IN_PROGRESS"
        return doc["status"] if doc else "DELETED"

    def is_queue_paused(self) -> bool:
        try:
            doc = self.db.queue_control.find_one({"_id": "singleton"})
            return bool(doc and doc.get("paused"))
        except Exception as exc:  # noqa: BLE001
            print(f"queue pause check failed (treating as not paused): {exc}", flush=True)
            return False

    def next_job(self):
        mine = job_queue.robot_filter(self.cfg.robot)
        head = (self.db.jobs.find_one({"status": "IN_PROGRESS", **mine}, {"_id": 1})
                or job_queue.oldest(self.db, {"status": "NOT_STARTED", **mine}))
        return self.db.jobs.find_one({"_id": head["_id"]}) if head else None

    def stop_process(self, proc):
        for sig, wait_s in ((signal.SIGINT, self.cfg.stop_grace_s), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue

    def run_process(self, job_id, argv, log, extra_env=None, on_line=None, on_tick=None):
        log.write(f"$ {' '.join(argv)}\n")
        proc = subprocess.Popen(
            argv, cwd=self.cfg.workdir, env={**os.environ, **(extra_env or {})},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
        lines = queue.Queue()

        def read():
            for line in proc.stdout:
                lines.put(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        tail = []
        outcome = "finished"
        next_status_check = 0.0

        def handle(line):
            log.write(line)
            tail.append(line)
            del tail[:-40]
            if on_line:
                on_line(line)

        def drain():
            while True:
                try:
                    handle(lines.get_nowait())
                except queue.Empty:
                    return

        while proc.poll() is None:
            try:
                handle(lines.get(timeout=0.5))
            except queue.Empty:
                pass
            drain()
            if on_tick:
                on_tick()
            if self.stopping.is_set():
                log.write(f"[{self.cfg.tag}] shutting down; stopping training, job stays IN_PROGRESS to resume\n")
                self.stop_process(proc)
                sys.exit(0)
            if time.time() >= next_status_check:
                next_status_check = time.time() + self.cfg.status_poll_s
                status = self.job_status(job_id)
                if status == "PAUSE_REQUESTED":
                    outcome = "pause"
                elif status != "IN_PROGRESS":
                    outcome = "cancel"
                if outcome != "finished":
                    log.write(f"[{self.cfg.tag}] job status is {status}; stopping training\n")
                    self.stop_process(proc)
                    break
        reader.join(timeout=5)
        drain()
        return outcome, proc.wait(), "".join(tail)

    def find_run_dir(self, name):
        matches = sorted(self.cfg.log_root.glob(f"rsl_rl/*/*_{name}"))
        return matches[-1] if matches else None

    def resolve_source_job(self, job):
        source_id = job.get("source_job_id")
        if source_id is None:
            raise core_launch.JobSpecError("set source_job_id")
        source = self.db.jobs.find_one({"_id": source_id}) or self.db.jobs.find_one(
            {"_id": _maybe_object_id(source_id)})
        if not source or not source.get(self.cfg.checkpoint_field):
            raise core_launch.JobSpecError(f"job {source_id} has no {self.cfg.robot} checkpoint")
        return source

    def resolve_checkpoint(self, job):
        if job.get("checkpoint"):
            return pathlib.Path(job["checkpoint"])
        return pathlib.Path(self.resolve_source_job(job)[self.cfg.checkpoint_field])

    def job_reward_design(self, job, resume):
        if resume and job.get("reward_design_code") is not None:
            return {k: job.get(k) for k in DESIGN_FIELDS}
        rd_id = job.get("reward_design_id")
        if not rd_id:
            return None
        doc = self.db.reward_designs.find_one({"_id": _maybe_object_id(rd_id)}) or self.db.reward_designs.find_one(
            {"_id": str(rd_id)})
        if doc is None:
            raise core_launch.JobSpecError(f"reward_design_id {rd_id!r} not found")
        if doc.get("robot") != self.cfg.robot:
            raise core_launch.JobSpecError(
                f"reward design {doc.get('name')!r} is a {doc.get('robot') or 'roboracer'} design; "
                f"pick one made for the {self.cfg.robot}")
        blocked = reward_spec.unusable_reason(doc, reward_spec.strict_generated(self.db))
        if blocked:
            raise core_launch.JobSpecError(blocked)
        task = core_launch.task_id(job, self.cfg.launch_cfg)
        gym = self.db.gyms.find_one({"_id": self.cfg.gym_id_fn(task)}, {"reward_terms": 1}) or {}
        catalog = self.db.reward_term_catalog.find_one({"_id": self.cfg.robot}, {"functions.name": 1}) or {}
        library = [f["name"] for f in catalog.get("functions") or []] or None
        result = self.cfg.lint_reward_fn(doc.get("code") or "", gym.get("reward_terms"), library)
        if not result["ok"]:
            raise core_launch.JobSpecError(f"reward design {doc.get('name')!r} on {task}: {result['error']}")
        return {"reward_design_id": str(doc["_id"]), "reward_design_name": doc.get("name"),
                "reward_design_version": doc.get("version"), "reward_design_code": doc["code"]}

    def job_experiment_design(self, job, resume):
        if resume and job.get("experiment_design_fields") is not None:
            return {k: job.get(k) for k in EXPERIMENT_FIELDS}
        ed_id = job.get("experiment_design_id")
        if not ed_id:
            return None
        doc = self.db.experiment_designs.find_one({"_id": _maybe_object_id(ed_id)}) or self.db.experiment_designs.find_one(
            {"_id": str(ed_id)})
        if doc is None:
            raise core_launch.JobSpecError(f"experiment_design_id {ed_id!r} not found")
        if doc.get("robot") != self.cfg.robot:
            raise core_launch.JobSpecError(
                f"experiment design {doc.get('name')!r} is a {doc.get('robot') or 'roboracer'} design; "
                f"pick one made for the {self.cfg.robot}")
        try:
            fields = self.cfg.validate_experiment_fn(doc)
        except self.cfg.experiment_error_cls as exc:
            raise core_launch.JobSpecError(f"experiment design {doc.get('name')!r}: {exc}") from exc
        return {"experiment_design_id": str(doc["_id"]), "experiment_design_name": doc.get("name"),
                "experiment_design_version": doc.get("version"), "experiment_design_fields": fields}

    def write_experiment_design(self, job_id, design):
        path = self.cfg.job_log_dir / f"{job_id}.experiment_design.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_id": design["experiment_design_id"],
                                    "name": design["experiment_design_name"],
                                    "version": design["experiment_design_version"],
                                    "fields": design["experiment_design_fields"]}))
        return path

    def write_reward_design(self, job_id, design):
        path = self.cfg.job_log_dir / f"{job_id}.reward_design.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_id": design["reward_design_id"], "name": design["reward_design_name"],
                                    "version": design["reward_design_version"],
                                    "code": design["reward_design_code"]}))
        return path

    def train_job_for_checkpoint(self, checkpoint):
        return self.db.jobs.find_one({
            "robot": self.cfg.robot, "job_type": "TRAIN",
            self.cfg.run_dirs_field: str(pathlib.Path(checkpoint).parent),
        })

    def checkpoint_design_env(self, job_id, checkpoint):
        source = self.train_job_for_checkpoint(checkpoint)
        if not source or not source.get("experiment_design_fields"):
            return {}
        path = self.write_experiment_design(job_id, {k: source.get(k) for k in EXPERIMENT_FIELDS})
        return core_launch.design_env(self.cfg.launch_cfg, experiment_design_file=path, train=False)

    def upsert_model(self, job, checkpoint, iteration, **fields):
        self.db.models.update_one(
            {"robot": self.cfg.robot, "job_id": str(job["_id"])},
            {"$set": {
                "location": str(checkpoint),
                "training_iterations": iteration,
                "task_id": core_launch.task_id(job, self.cfg.launch_cfg),
                "model_type": "PPO",
                "updated_at": now(),
                **{k: job[k] for k in DESIGN_FIELDS + EXPERIMENT_FIELDS if job.get(k) is not None},
                **fields,
            }, "$setOnInsert": {
                "robot": self.cfg.robot,
                "job_id": str(job["_id"]),
                "create_date": now(),
                "notes": "NA",
            }},
            upsert=True)

    def export_checkpoint(self, job, checkpoint, log):
        iteration = core_launch.checkpoint_iteration(checkpoint)
        onnx_path = checkpoint.parent / f"policy_{iteration}.onnx"
        outcome, rc, tail = self.run_process(
            job["_id"], core_launch.export_command(job, checkpoint, onnx_path, self.cfg.launch_cfg), log,
            self.checkpoint_design_env(job["_id"], checkpoint))
        if outcome == "pause":
            self.update_job(job["_id"], status="PAUSED", paused_at_step=iteration)
        if outcome != "finished":
            return outcome, None
        if rc != 0 or not onnx_path.exists():
            raise RuntimeError(f"export exited {rc}:\n{tail}")
        report = validate_onnx(core_launch.mjlab_task_id(job, self.cfg.launch_cfg), onnx_path,
                               self.cfg.deploy_shapes)
        log.write(f"[{self.cfg.tag}] ONNX check: {json.dumps({k: v for k, v in report.items() if k != 'metadata'})}\n")
        if not report["ok"]:
            raise RuntimeError(f"exported ONNX failed validation: {report['problems']}")
        return outcome, (onnx_path, report)

    def job_videos(self, job, run_dirs):
        legacy = [v for v in job.get("videos") or [] if "gif" not in v]
        return legacy + core_launch.checkpoint_clips(run_dirs, clip_dir=self.cfg.launch_cfg.clip_dir)

    def start_clipper(self, job, run_dir, extra_env, log):
        argv = core_launch.clip_watch_command(job, run_dir, self.cfg.render_progress_script, self.cfg.launch_cfg)
        clip_dir = pathlib.Path(run_dir) / self.cfg.launch_cfg.clip_dir
        clip_dir.mkdir(parents=True, exist_ok=True)
        log.write(f"[{self.cfg.tag}] checkpoint clips: $ {' '.join(argv)}\n")
        with open(clip_dir / "render.log", "a") as out:
            return subprocess.Popen(argv, cwd=self.cfg.workdir, env={**os.environ, **extra_env},
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def render_videos(self, job, source, run_dirs, log, every_checkpoint, progression):
        found = core_launch.checkpoints(run_dirs)
        if not found:
            raise RuntimeError(f"job {source['_id']} has no checkpoints to render")
        picked = (core_launch.pick_evenly(found, core_launch.positive_int(job, "video_checkpoints", self.cfg.launch_cfg.defaults))
                  if progression else [])
        missing = [p for _, p in found if every_checkpoint and not core_launch.clip_path(p, self.cfg.launch_cfg.clip_dir).exists()]
        last_iteration, last_checkpoint = (picked or found)[-1]
        out_mp4 = last_checkpoint.parent / f"progress_{last_iteration}.mp4" if progression else None
        if missing or picked:
            argv = core_launch.video_command(
                {**job, "task_id": core_launch.task_id(source, self.cfg.launch_cfg)},
                self.cfg.render_progress_script, self.cfg.launch_cfg,
                clips=missing, progression=[p for _, p in picked], out_mp4=out_mp4)
            outcome, rc, tail = self.run_process(
                job["_id"], argv, log, self.checkpoint_design_env(job["_id"], last_checkpoint))
            if outcome == "pause":
                self.update_job(job["_id"], status="PAUSED")
            if outcome != "finished":
                return
            if rc != 0 or (out_mp4 and not out_mp4.exists()):
                raise RuntimeError(f"video render exited {rc}:\n{tail}")
        fields = {"videos": self.job_videos(source, run_dirs)}
        if out_mp4:
            fields.update(progress_video=str(out_mp4), progress_gif=str(out_mp4.with_suffix(".gif")))
        self.update_job(source["_id"], **fields)
        if out_mp4 and job["_id"] != source["_id"]:
            self.update_job(job["_id"], progress_video=fields["progress_video"], progress_gif=fields["progress_gif"])

    def run_train(self, job, resume, log):
        job_id = job["_id"]
        lcfg = self.cfg.launch_cfg
        run_dirs = list(job.get(self.cfg.run_dirs_field) or [])
        latest = core_launch.latest_checkpoint(run_dirs) if resume else None
        if not resume:
            run_dirs = []
            job["videos"] = []
            self.update_job(
                job_id,
                **{
                    self.cfg.run_dirs_field: [],
                    self.cfg.checkpoint_field: None,
                    self.cfg.iteration_field: 0,
                    "paused_at_step": None,
                    "videos": [],
                    "progress_video": bool(job.get("progress_video")),
                    "progress_gif": None,
                })
        experiment = self.job_experiment_design(job, resume)
        experiment_file = None
        if experiment:
            job.update(experiment)
            self.update_job(job_id, **experiment)
            job = self.cfg.effective_job_fn(job)
            experiment_file = self.write_experiment_design(job_id, experiment)
            log.write(f"[{self.cfg.tag}] experiment design {experiment['experiment_design_name']!r} "
                      f"v{experiment['experiment_design_version']}: "
                      f"{json.dumps(experiment['experiment_design_fields'])}\n")
        if job.get("init_checkpoint") and not resume:
            source = self.train_job_for_checkpoint(job["init_checkpoint"]) or {}
            try:
                self.cfg.check_warm_start_fn(
                    (experiment or {}).get("experiment_design_fields") or {},
                    source.get("experiment_design_fields") or {})
            except self.cfg.experiment_error_cls as exc:
                raise core_launch.JobSpecError(str(exc)) from exc
        offset = core_launch.iteration_offset(job)
        total = core_launch.positive_int(job, "max_iterations", lcfg.defaults)
        attempt = len(run_dirs)
        design = self.job_reward_design(job, resume)
        design_file = None
        if design:
            job.update(design)
            self.update_job(job_id, **design)
            design_file = self.write_reward_design(job_id, design)
            log.write(f"[{self.cfg.tag}] reward design {design['reward_design_name']!r} "
                      f"v{design['reward_design_version']}\n")
        argv, extra_env = core_launch.train_command(
            job, attempt, lcfg, resume_from=latest and latest[1],
            reward_design_file=design_file, experiment_design_file=experiment_file)
        if latest:
            log.write(f"[{self.cfg.tag}] resuming from {latest[1]} (iteration {latest[0]})\n")

        state = {"run_dir": None, "checkpoint": latest, "last_write": 0.0, "progress": {},
                 "videos": None, "clipper": None}
        name = core_launch.run_name(job, attempt)
        render_env = (core_launch.design_env(lcfg, experiment_design_file=experiment_file, train=False)
                      if experiment_file else {})

        def on_line(line):
            state["progress"].update(core_launch.parse_progress(line))

        def on_tick():
            if time.time() - state["last_write"] < self.cfg.progress_write_s:
                return
            state["last_write"] = time.time()
            if state["run_dir"] is None:
                state["run_dir"] = self.find_run_dir(name)
                if state["run_dir"] is not None:
                    run_dirs.append(str(state["run_dir"]))
                    self.update_job(job_id, **{self.cfg.run_dirs_field: run_dirs})
                    if core_launch.setting(job, "video", lcfg.defaults):
                        state["clipper"] = self.start_clipper(job, state["run_dir"], render_env, log)
            progress = state["progress"]
            fields = {}
            if "iteration" in progress:
                fields[self.cfg.iteration_field] = progress["iteration"]
                done = progress["iteration"] + 1 - offset
                fields["percent_complete"] = min(100.0, round(100.0 * done / total, 1))
            if "mean_reward" in progress:
                fields["mean_reward"] = progress["mean_reward"]
            if "mean_episode_length" in progress:
                fields["mean_episode_length"] = progress["mean_episode_length"]
            newest = core_launch.latest_checkpoint(run_dirs)
            if newest and newest != state["checkpoint"]:
                state["checkpoint"] = newest
                fields[self.cfg.checkpoint_field] = str(newest[1])
                self.upsert_model(job, newest[1], newest[0], avg_return=progress.get("mean_reward"))
            clips = self.job_videos(job, run_dirs)
            if clips != state["videos"]:
                state["videos"] = fields["videos"] = clips
            if fields:
                self.update_job(job_id, **fields)

        if argv is None:
            log.write(f"[{self.cfg.tag}] iteration budget already reached\n")
        else:
            try:
                outcome, rc, tail = self.run_process(job_id, argv, log, extra_env, on_line, on_tick)
            finally:
                if state["clipper"] is not None:
                    self.stop_process(state["clipper"])
            state["last_write"] = 0.0
            on_tick()
            if outcome == "pause":
                newest = core_launch.latest_checkpoint(run_dirs)
                step = newest[0] if newest else 0
                self.update_job(
                    job_id, status="PAUSED", paused_at_step=step,
                    percent_complete=round(100.0 * (step - offset) / total, 1),
                    **{
                        self.cfg.iteration_field: step,
                        self.cfg.checkpoint_field: str(newest[1]) if newest else None,
                    })
                log.write(f"[{self.cfg.tag}] paused at checkpoint iteration {step}\n")
                return
            if outcome == "cancel":
                return
            if rc != 0:
                raise RuntimeError(f"train exited {rc}:\n{tail}")

        newest = core_launch.latest_checkpoint(run_dirs)
        if newest is None:
            raise RuntimeError("training finished without writing a checkpoint")
        self.update_job(job_id, percent_complete=100,
                        **{self.cfg.checkpoint_field: str(newest[1]), self.cfg.iteration_field: newest[0]})
        outcome, exported = self.export_checkpoint(job, newest[1], log)
        if exported:
            onnx_path, report = exported
            self.upsert_model(job, newest[1], newest[0], onnx_location=str(onnx_path),
                              obs_dim=report["obs_dim"], act_dim=report["act_dim"],
                              avg_return=state["progress"].get("mean_reward"))
            self.update_job(job_id, onnx_location=str(onnx_path))
        clips = bool(core_launch.setting(job, "video", lcfg.defaults))
        progress = bool(job.get("progress_video"))
        if outcome == "finished" and (clips or progress):
            self.render_videos(job, job, run_dirs, log, every_checkpoint=clips, progression=progress)

    def run_video(self, job, log):
        source = self.resolve_source_job(job)
        self.render_videos(job, source, source.get(self.cfg.run_dirs_field) or [], log,
                           every_checkpoint=False, progression=True)
        self.update_job(job["_id"], percent_complete=100)

    def run_export(self, job, log):
        checkpoint = self.resolve_checkpoint(job)
        _, exported = self.export_checkpoint(job, checkpoint, log)
        if exported:
            onnx_path, report = exported
            self.update_job(job["_id"], onnx_location=str(onnx_path), export_report=report)
            self.db.models.update_many({"robot": self.cfg.robot, "location": str(checkpoint)},
                                       {"$set": {"onnx_location": str(onnx_path),
                                                 "obs_dim": report["obs_dim"],
                                                 "act_dim": report["act_dim"]}})

    def run_eval(self, job, log):
        checkpoint = self.resolve_checkpoint(job)
        out_json = self.cfg.job_log_dir / f"{job['_id']}.eval.json"
        argv = core_launch.eval_command(job, checkpoint, out_json, self.cfg.evaluate_script, self.cfg.launch_cfg)
        outcome, rc, tail = self.run_process(job["_id"], argv, log, self.checkpoint_design_env(job["_id"], checkpoint))
        if outcome == "pause":
            self.update_job(job["_id"], status="PAUSED")
        if outcome != "finished":
            return
        if rc != 0 or not out_json.exists():
            raise RuntimeError(f"eval exited {rc}:\n{tail}")
        results = json.loads(out_json.read_text())
        self.update_job(job["_id"], eval_results=results, percent_complete=100)
        self.db.models.update_many({"robot": self.cfg.robot, "location": str(checkpoint)},
                                   {"$set": {"eval_results": results, "eval_at": now(),
                                             "eval_job_id": str(job["_id"])}})

    def sync_gyms(self):
        try:
            import robots_registry
            robots_registry.seed_robots(self.db, now=now())
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.cfg.tag}] robot manifest seed failed: {exc}", flush=True)
        try:
            out = subprocess.run([sys.executable, str(self.cfg.list_tasks_script)], cwd=self.cfg.workdir,
                                 capture_output=True, text=True, timeout=600, check=True).stdout
            listing = json.loads(out.strip().splitlines()[-1])
            upserted, archived = self.cfg.sync_gyms_fn(self.db, listing["tasks"])
            ts = now()
            self.cfg.seed_reward_fn(self.db, listing["functions"], ts)
            self.cfg.seed_experiment_fn(self.db, ts)
            print(f"[{self.cfg.tag}] gyms: {upserted} {self.cfg.robot} tasks registered, "
                  f"{archived} archived; {len(listing['functions'])} library reward functions", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.cfg.tag}] gym sync failed: {exc}", flush=True)

    def do_job(self, job):
        job_id = job["_id"]
        job_type = job.get("job_type")
        if not job.get("task_id"):
            task = self.cfg.task_for_job_fn(self.db, job)
            if task:
                job["task_id"] = task
                self.update_job(job_id, task_id=task)
        resume = job_type == "TRAIN" and (
            job.get("status") == "IN_PROGRESS" or job.get("paused_at_step") is not None)
        fields = {"status": "IN_PROGRESS", "ended_at": None, "eval_error": None}
        if not resume:
            fields.update(percent_complete=0, started_at=now())
        self.update_job(job_id, **fields)
        print(f"[{self.cfg.tag}] {job_type} job {job_id} ({'resume' if resume else 'fresh'})", flush=True)

        log = JobLog(self.cfg.job_log_dir, self.cfg.shared_log, job_id)
        self.update_job(job_id, log_path=str(log.path))
        try:
            if job_type == "TRAIN":
                self.run_train(job, resume, log)
            elif job_type == "EXPORT":
                self.run_export(job, log)
            elif job_type == "EVAL":
                self.run_eval(job, log)
            elif job_type == "VIDEO":
                self.run_video(job, log)
            else:
                raise core_launch.JobSpecError(f"unsupported job_type {job_type!r} for {self.cfg.robot}")
        except Exception as exc:  # noqa: BLE001
            log.write(traceback.format_exc())
            self.update_job(job_id, status="FAILED", ended_at=now(),
                            eval_error=f"{type(exc).__name__}: {exc}"[:4000])
            return
        finally:
            log.close()

        status = self.job_status(job_id)
        if status == "PAUSED":
            return
        self.update_job(job_id, ended_at=now())
        if status == "IN_PROGRESS":
            self.update_job(job_id, status="DONE")

    def run_forever(self):
        signal.signal(signal.SIGTERM, lambda *_: self.stopping.set())
        job_queue.start_heartbeat(self.db, self.cfg.robot)
        self.sync_gyms()
        print(f"[{self.cfg.tag}] polling for {self.cfg.robot} jobs", flush=True)
        waiting_reason = None
        while not self.stopping.is_set():
            if self.is_queue_paused():
                self.stopping.wait(self.cfg.poll_s)
                continue
            allowed, reason = job_queue.may_claim(self.db, self.cfg.robot)
            if not allowed:
                if reason != waiting_reason:
                    print(f"[{self.cfg.tag}] waiting for shared queue: {reason}", flush=True)
                    waiting_reason = reason
                self.stopping.wait(self.cfg.poll_s)
                continue
            waiting_reason = None
            job = self.next_job()
            if job is None:
                self.stopping.wait(self.cfg.poll_s)
                continue
            self.do_job(job)
        print(f"[{self.cfg.tag}] SIGTERM: exiting without claiming more jobs", flush=True)
