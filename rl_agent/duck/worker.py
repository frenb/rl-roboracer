"""Runs ``robot="microduck"`` jobs from the shared ``jobs`` queue.

Started by the duck-trainer container (compose/duck.yml) from /microduck_rl:

    python -u /python_ws/src/duck/worker.py

Follows the car trainer's job lifecycle so the dashboard treats both alike:

    NOT_STARTED -> IN_PROGRESS -> DONE | FAILED
    IN_PROGRESS -> PAUSE_REQUESTED -> PAUSED (paused_at_step set)
    PAUSED -> NOT_STARTED (operator resumes) -> IN_PROGRESS ...

Any other status written while a job runs (e.g. DONE from "Set to done")
stops it without overwriting that status.

Job types: TRAIN, EXPORT, EVAL. Field reference: duck/launch.py.
"""
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

from pymongo import MongoClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import job_queue  # noqa: E402
from duck import launch  # noqa: E402

ROBOT = "microduck"
WORKDIR = pathlib.Path(os.environ.get("MICRODUCK_RL_DIR", "/microduck_rl"))
LOG_ROOT = WORKDIR / "logs"
JOB_LOG_DIR = LOG_ROOT / "jobs"
SHARED_LOG = pathlib.Path(os.environ.get("DUCK_SHARED_LOG", "/shared_tmp/duck_trainer.log"))
EVALUATE_SCRIPT = pathlib.Path(__file__).resolve().parent / "evaluate.py"
POLL_S = 5
STATUS_POLL_S = 2
PROGRESS_WRITE_S = 5
STOP_GRACE_S = 30
# Velocity policies deployed on the real duck: 48 proprioception + 13 command
# inputs -> 14 servo targets. Other tasks have their own shapes.
DEPLOY_SHAPE = {"Mjlab-Velocity-": (61, 14)}

db = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))[
    os.environ.get("DATABASE_NAME", "robotaxi")]


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def update_job(job_id, **fields):
    db.jobs.update_one({"_id": job_id}, {"$set": fields})


def job_status(job_id):
    try:
        doc = db.jobs.find_one({"_id": job_id}, {"status": 1})
    except Exception as exc:  # noqa: BLE001 - a Mongo blip must not stop training
        print(f"status lookup failed ({exc}); continuing", flush=True)
        return "IN_PROGRESS"
    return doc["status"] if doc else "DELETED"


def is_queue_paused():
    try:
        doc = db.queue_control.find_one({"_id": "singleton"})
        return bool(doc and doc.get("paused"))
    except Exception as exc:  # noqa: BLE001 - fail open like robotaxi
        print(f"queue pause check failed (treating as not paused): {exc}", flush=True)
        return False


def next_job():
    mine = job_queue.robot_filter(ROBOT)
    return (db.jobs.find_one({"status": "IN_PROGRESS", **mine})
            or db.jobs.find_one({"status": "NOT_STARTED", **mine},
                                sort=[("create_date", 1)]))


class JobLog:
    """Tees subprocess output to the job's own log file and the shared panel log."""

    def __init__(self, job_id):
        JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = JOB_LOG_DIR / f"{job_id}.log"
        self._files = [open(self.path, "a", buffering=1)]
        try:
            SHARED_LOG.parent.mkdir(parents=True, exist_ok=True)
            self._files.append(open(SHARED_LOG, "w", buffering=1))
        except OSError as exc:
            print(f"shared log unavailable: {exc}", flush=True)

    def write(self, line):
        sys.stdout.write(line)
        for f in self._files:
            f.write(line)

    def close(self):
        for f in self._files:
            f.close()


def run_process(job_id, argv, log, extra_env=None, on_line=None, on_tick=None):
    """Run ``argv`` in WORKDIR, watching the job's status.

    Returns ``(outcome, returncode, tail)`` where outcome is ``finished``,
    ``pause`` or ``cancel``.
    """
    log.write(f"$ {' '.join(argv)}\n")
    proc = subprocess.Popen(
        argv, cwd=WORKDIR, env={**os.environ, **(extra_env or {})},
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
        if time.time() >= next_status_check:
            next_status_check = time.time() + STATUS_POLL_S
            status = job_status(job_id)
            if status == "PAUSE_REQUESTED":
                outcome = "pause"
            elif status != "IN_PROGRESS":
                outcome = "cancel"
            if outcome != "finished":
                log.write(f"[duck-worker] job status is {status}; stopping training\n")
                stop_process(proc)
                break
    reader.join(timeout=5)
    drain()
    return outcome, proc.wait(), "".join(tail)


def stop_process(proc):
    """SIGINT the whole process group, then SIGKILL after a grace period."""
    for sig, wait_s in ((signal.SIGINT, STOP_GRACE_S), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def find_run_dir(name):
    matches = sorted(LOG_ROOT.glob(f"rsl_rl/*/*_{name}"))
    return matches[-1] if matches else None


def resolve_checkpoint(job):
    """Checkpoint an EXPORT or EVAL job targets: explicit, or a TRAIN job's latest."""
    if job.get("checkpoint"):
        return pathlib.Path(job["checkpoint"])
    source_id = job.get("source_job_id")
    if source_id is None:
        raise launch.JobSpecError("set checkpoint or source_job_id")
    source = db.jobs.find_one({"_id": source_id}) or db.jobs.find_one(
        {"_id": _maybe_object_id(source_id)})
    if not source or not source.get("duck_checkpoint"):
        raise launch.JobSpecError(f"job {source_id} has no microduck checkpoint")
    return pathlib.Path(source["duck_checkpoint"])


def _maybe_object_id(value):
    from bson import ObjectId
    try:
        return ObjectId(str(value))
    except Exception:  # noqa: BLE001
        return value


def upsert_model(job, checkpoint, iteration, **fields):
    """One ``models`` document per TRAIN job, pointing at its latest checkpoint."""
    db.models.update_one(
        {"robot": ROBOT, "job_id": str(job["_id"])},
        {"$set": {
            "location": str(checkpoint),
            "training_iterations": iteration,
            "task_id": launch.task_id(job),
            "model_type": "PPO",
            "updated_at": now(),
            **fields,
        }, "$setOnInsert": {
            "robot": ROBOT,
            "job_id": str(job["_id"]),
            "create_date": now(),
            "notes": "NA",
        }},
        upsert=True)


def validate_onnx(task, onnx_path):
    """Shapes, metadata and a finite forward pass for an exported policy."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    obs_dim, act_dim = inp.shape[-1], out.shape[-1]
    action = session.run(None, {inp.name: np.zeros((1, obs_dim), dtype=np.float32)})[0]
    report = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "finite_on_zero_obs": bool(np.isfinite(action).all()),
        "metadata": {p.key: p.value for p in model.metadata_props},
    }
    problems = []
    if not report["finite_on_zero_obs"]:
        problems.append("non-finite action for a zero observation")
    for prefix, shape in DEPLOY_SHAPE.items():
        if task.startswith(prefix) and (obs_dim, act_dim) != shape:
            problems.append(f"expected {shape[0]} inputs / {shape[1]} outputs, got {obs_dim} / {act_dim}")
    report["ok"] = not problems
    report["problems"] = problems
    return report


def export_checkpoint(job, checkpoint, log):
    iteration = launch.checkpoint_iteration(checkpoint)
    onnx_path = checkpoint.parent / f"policy_{iteration}.onnx"
    outcome, rc, tail = run_process(
        job["_id"], launch.export_command(job, checkpoint, onnx_path), log)
    if outcome == "pause":
        update_job(job["_id"], status="PAUSED", paused_at_step=iteration)
    if outcome != "finished":
        return outcome, None
    if rc != 0 or not onnx_path.exists():
        raise RuntimeError(f"export exited {rc}:\n{tail}")
    report = validate_onnx(launch.task_id(job), onnx_path)
    log.write(f"[duck-worker] ONNX check: {json.dumps({k: v for k, v in report.items() if k != 'metadata'})}\n")
    if not report["ok"]:
        raise RuntimeError(f"exported ONNX failed validation: {report['problems']}")
    return outcome, (onnx_path, report)


def run_train(job, resume, log):
    job_id = job["_id"]
    run_dirs = list(job.get("duck_run_dirs") or [])
    latest = launch.latest_checkpoint(run_dirs) if resume else None
    if not resume:
        run_dirs = []
        update_job(job_id, duck_run_dirs=[], duck_checkpoint=None, duck_iteration=0,
                   paused_at_step=None)
    total = launch.positive_int(job, "max_iterations")
    attempt = len(run_dirs)
    argv, extra_env = launch.train_command(job, attempt, resume_from=latest and latest[1])
    if latest:
        log.write(f"[duck-worker] resuming from {latest[1]} (iteration {latest[0]})\n")

    state = {"run_dir": None, "checkpoint": latest, "last_write": 0.0, "progress": {}}
    name = launch.run_name(job, attempt)

    def on_line(line):
        state["progress"].update(launch.parse_progress(line))

    def on_tick():
        if time.time() - state["last_write"] < PROGRESS_WRITE_S:
            return
        state["last_write"] = time.time()
        if state["run_dir"] is None:
            state["run_dir"] = find_run_dir(name)
            if state["run_dir"] is not None:
                run_dirs.append(str(state["run_dir"]))
                update_job(job_id, duck_run_dirs=run_dirs)
        progress = state["progress"]
        fields = {}
        if "iteration" in progress:
            fields["duck_iteration"] = progress["iteration"]
            fields["percent_complete"] = min(100.0, round(100.0 * (progress["iteration"] + 1) / total, 1))
        if "mean_reward" in progress:
            fields["mean_reward"] = progress["mean_reward"]
        if "mean_episode_length" in progress:
            fields["mean_episode_length"] = progress["mean_episode_length"]
        newest = launch.latest_checkpoint(run_dirs)
        if newest and newest != state["checkpoint"]:
            state["checkpoint"] = newest
            fields["duck_checkpoint"] = str(newest[1])
            upsert_model(job, newest[1], newest[0], avg_return=progress.get("mean_reward"))
        if fields:
            update_job(job_id, **fields)

    if argv is None:
        log.write("[duck-worker] iteration budget already reached\n")
    else:
        outcome, rc, tail = run_process(job_id, argv, log, extra_env, on_line, on_tick)
        state["last_write"] = 0.0
        on_tick()
        if outcome == "pause":
            newest = launch.latest_checkpoint(run_dirs)
            step = newest[0] if newest else 0
            # Iterations after the last checkpoint are lost, so report where
            # the resume will actually restart.
            update_job(job_id, status="PAUSED", paused_at_step=step, duck_iteration=step,
                       percent_complete=round(100.0 * step / total, 1),
                       duck_checkpoint=str(newest[1]) if newest else None)
            log.write(f"[duck-worker] paused at checkpoint iteration {step}\n")
            return
        if outcome == "cancel":
            return
        if rc != 0:
            raise RuntimeError(f"train exited {rc}:\n{tail}")

    newest = launch.latest_checkpoint(run_dirs)
    if newest is None:
        raise RuntimeError("training finished without writing a checkpoint")
    update_job(job_id, duck_checkpoint=str(newest[1]), duck_iteration=newest[0],
               percent_complete=100)
    outcome, exported = export_checkpoint(job, newest[1], log)
    if exported:
        onnx_path, report = exported
        upsert_model(job, newest[1], newest[0], onnx_location=str(onnx_path),
                     obs_dim=report["obs_dim"], act_dim=report["act_dim"],
                     avg_return=state["progress"].get("mean_reward"))
        update_job(job_id, onnx_location=str(onnx_path))


def run_export(job, log):
    checkpoint = resolve_checkpoint(job)
    _, exported = export_checkpoint(job, checkpoint, log)
    if exported:
        onnx_path, report = exported
        update_job(job["_id"], onnx_location=str(onnx_path), export_report=report)
        db.models.update_many({"robot": ROBOT, "location": str(checkpoint)},
                              {"$set": {"onnx_location": str(onnx_path),
                                        "obs_dim": report["obs_dim"],
                                        "act_dim": report["act_dim"]}})


def run_eval(job, log):
    checkpoint = resolve_checkpoint(job)
    out_json = JOB_LOG_DIR / f"{job['_id']}.eval.json"
    argv = launch.eval_command(job, checkpoint, out_json, EVALUATE_SCRIPT)
    outcome, rc, tail = run_process(job["_id"], argv, log)
    if outcome == "pause":
        update_job(job["_id"], status="PAUSED")
    if outcome != "finished":
        return
    if rc != 0 or not out_json.exists():
        raise RuntimeError(f"eval exited {rc}:\n{tail}")
    results = json.loads(out_json.read_text())
    update_job(job["_id"], eval_results=results, percent_complete=100)
    db.models.update_many({"robot": ROBOT, "location": str(checkpoint)},
                          {"$set": {"eval_results": results}})


def do_job(job):
    job_id = job["_id"]
    job_type = job.get("job_type")
    # Same discriminator as robotaxi: IN_PROGRESS at pickup is crash recovery,
    # NOT_STARTED with paused_at_step set is an operator resume.
    resume = job_type == "TRAIN" and (
        job.get("status") == "IN_PROGRESS" or job.get("paused_at_step") is not None)
    fields = {"status": "IN_PROGRESS", "ended_at": None, "eval_error": None}
    if not resume:
        fields.update(percent_complete=0, started_at=now())
    update_job(job_id, **fields)
    print(f"[duck-worker] {job_type} job {job_id} ({'resume' if resume else 'fresh'})", flush=True)

    log = JobLog(job_id)
    update_job(job_id, log_path=str(log.path))
    try:
        if job_type == "TRAIN":
            run_train(job, resume, log)
        elif job_type == "EXPORT":
            run_export(job, log)
        elif job_type == "EVAL":
            run_eval(job, log)
        else:
            raise launch.JobSpecError(f"unsupported job_type {job_type!r} for {ROBOT}")
    except Exception as exc:  # noqa: BLE001 - record it and keep draining the queue
        log.write(traceback.format_exc())
        update_job(job_id, status="FAILED", ended_at=now(),
                   eval_error=f"{type(exc).__name__}: {exc}"[:4000])
        return
    finally:
        log.close()

    status = job_status(job_id)
    if status == "PAUSED":
        return
    update_job(job_id, ended_at=now())
    if status == "IN_PROGRESS":
        update_job(job_id, status="DONE")


def main():
    job_queue.start_heartbeat(db, ROBOT)
    print(f"[duck-worker] polling for {ROBOT} jobs", flush=True)
    waiting_reason = None
    while True:
        if is_queue_paused():
            time.sleep(POLL_S)
            continue
        allowed, reason = job_queue.may_claim(db, ROBOT)
        if not allowed:
            if reason != waiting_reason:
                print(f"[duck-worker] waiting for shared queue: {reason}", flush=True)
                waiting_reason = reason
            time.sleep(POLL_S)
            continue
        waiting_reason = None
        job = next_job()
        if job is None:
            time.sleep(POLL_S)
            continue
        do_job(job)


if __name__ == "__main__":
    main()
