"""Chains a curriculum run's stages as jobs on the shared queue.

Each stage is one TRAIN job; a gated stage is followed by an EVAL job of the
TRAIN job's checkpoint, and the gate is checked against its ``eval_results``.
The runner advances, retries or stops the run from what it finds in Mongo, so
it can be restarted at any point: every job it queues carries a
``curriculum_key`` and is looked up before being queued again.

Run documents (``curriculum_runs``)::

    status        RUNNING | CANCEL_REQUESTED | PASSED | FAILED_GATE | FAILED | CANCELLED
    stage_index   current stage; attempt counts retries of it (0 = first try)
    generation    manual retries of the run (the dashboard's Retry), part of job keys
    pending       {key, role: train | eval, job_id, train_job_id, stage_index, attempt}
    results       one entry per finished attempt, with metrics and gate details
"""
import datetime

from . import model

ACTIVE_JOB = ("NOT_STARTED", "IN_PROGRESS", "PAUSE_REQUESTED", "PAUSED")
ACTIVE_RUN = ("RUNNING", "CANCEL_REQUESTED")
FINAL_RUN = ("PASSED", "FAILED_GATE", "FAILED", "CANCELLED")
MAX_STEPS_PER_TICK = 8
FRONT_OFFSET = datetime.timedelta(milliseconds=1)


def job_key(run_id, stage_index, attempt, role, generation=0):
    """Generation counts manual retries of a run, so a retried stage gets new jobs."""
    key = f"{run_id}:{stage_index}:{attempt}:{role}"
    return f"{key}:g{generation}" if generation else key


def _strip_prefix(robot, task):
    task = str(task or "")
    return task[len(robot) + 1:] if task.startswith(f"{robot}:") else task


class Runner:
    """``store`` is MongoStore or a test fake; ``robots`` maps id -> manifest."""

    def __init__(self, store, robots, log=print):
        self.store = store
        self.robots = robots
        self.log = log

    def tick(self):
        changed = 0
        for run in self.store.active_runs():
            for _ in range(MAX_STEPS_PER_TICK):
                try:
                    progressed = self.step(run)
                except Exception as exc:  # noqa: BLE001 - one bad run must not stall the others
                    self.log(f"[curriculum] run {run['_id']}: {type(exc).__name__}: {exc}")
                    self._finish(run, "FAILED", error=f"runner error: {type(exc).__name__}: {exc}")
                    progressed = True
                if not progressed:
                    break
                changed += 1
                run = self.store.get_run(run["_id"])
                if run is None or run.get("status") not in ACTIVE_RUN:
                    break
        return changed

    def step(self, run):
        """Advance ``run`` by one transition; True when something changed."""
        status = run.get("status")
        if status == "CANCEL_REQUESTED":
            pending = run.get("pending")
            if pending and pending.get("job_id"):
                self.store.set_job_status(pending["job_id"], "CANCELLED", only_from=ACTIVE_JOB)
            self._finish(run, "CANCELLED")
            return True
        if status != "RUNNING":
            return False
        robot = self.robots.get(run.get("robot"))
        if robot is None or not model.caps(robot).get("enabled"):
            self._finish(run, "FAILED", error=f"curricula aren't enabled for {run.get('robot')}")
            return True

        pending = run.get("pending")
        if not pending:
            return self._queue_train(run, robot)
        job = self.store.get_job(pending["job_id"])
        if job is None:
            self._finish(run, "FAILED", error=f"{pending['role']} job {pending['job_id']} was deleted")
            return True
        job_status = job.get("status")
        if job_status in ACTIVE_JOB:
            return False
        if job_status == "CANCELLED":
            self._finish(run, "CANCELLED", error=f"{pending['role']} job {pending['job_id']} was cancelled")
            return True
        if job_status == "FAILED":
            self._finish(run, "FAILED", error=f"{pending['role']} job {pending['job_id']} failed: "
                                             f"{job.get('eval_error') or 'see its log'}")
            return True
        if job_status != "DONE":
            return False

        stage = run["stages"][run["stage_index"]]
        if pending["role"] == "train":
            if stage.get("gate"):
                return self._queue_eval(run, robot, job)
            self._record(run, job, None, True, [])
            return True
        train_job = self.store.get_job(pending["train_job_id"]) or {}
        passed, details = model.gate_result(stage["gate"], job.get("eval_results") or {})
        self._record(run, train_job, job, passed, details)
        return True

    # ---- transitions -----------------------------------------------------

    def _record(self, run, train_job, eval_job, passed, details):
        stage_index, attempt = run["stage_index"], run.get("attempt", 0)
        stage = run["stages"][stage_index]
        cap = model.caps(self.robots[run["robot"]])
        result = {
            "stage_index": stage_index, "stage_id": stage["id"], "stage_name": stage["name"],
            "attempt": attempt,
            "train_job_id": str(train_job.get("_id")) if train_job.get("_id") is not None else None,
            "eval_job_id": str(eval_job["_id"]) if eval_job else None,
            "checkpoint": train_job.get(cap.get("checkpoint_field")) if cap.get("checkpoint_field") else None,
            "metrics": (eval_job or {}).get("eval_results"),
            "gate": details, "passed": passed, "at": self.store.now(),
        }
        fields = {"pending": None}
        if passed:
            if stage_index + 1 >= len(run["stages"]):
                fields.update(status="PASSED", ended_at=self.store.now())
            else:
                fields.update(stage_index=stage_index + 1, attempt=0)
        elif attempt < stage["on_fail"]["retries"]:
            fields.update(attempt=attempt + 1)
        else:
            fields.update(status="FAILED_GATE", ended_at=self.store.now(),
                          error=f"{stage['name']} missed its gate after {attempt + 1} attempt(s)")
        self.store.update_run(run["_id"], fields, push_result=result)
        self.log(f"[curriculum] run {run['_id']} stage {stage_index + 1} attempt {attempt + 1}: "
                 f"{'passed' if passed else 'missed gate'}")

    def _finish(self, run, status, error=None):
        fields = {"status": status, "pending": None, "ended_at": self.store.now()}
        if error:
            fields["error"] = error
        self.store.update_run(run["_id"], fields)
        self.log(f"[curriculum] run {run['_id']} -> {status}" + (f": {error}" if error else ""))

    def _queue(self, run, role, doc, extra_pending):
        key = job_key(run["_id"], run["stage_index"], run.get("attempt", 0), role, run.get("generation", 0))
        existing = self.store.find_job_by_key(key)
        job_id = str(existing["_id"]) if existing else self.store.insert_job({
            **doc,
            "status": "NOT_STARTED",
            "create_date": self._create_date(run),
            "curriculum_key": key,
            "curriculum_run_id": str(run["_id"]),
            "curriculum_id": str(run.get("curriculum_id") or ""),
            "curriculum_name": run.get("curriculum_name"),
            "curriculum_stage_index": run["stage_index"],
            "curriculum_stage_id": run["stages"][run["stage_index"]]["id"],
            "curriculum_attempt": run.get("attempt", 0),
            "curriculum_role": role,
        })
        self.store.update_run(run["_id"], {"pending": {
            "key": key, "role": role, "job_id": job_id, "stage_index": run["stage_index"],
            "attempt": run.get("attempt", 0), **extra_pending}})
        return True

    def _create_date(self, run):
        now = self.store.now()
        if (run.get("options") or {}).get("priority", "front") != "front":
            return now
        head = self.store.queue_head_date()
        return min(now, head - FRONT_OFFSET) if head else now

    def _job_name(self, run, stage, attempt, suffix=""):
        name = f"{run.get('curriculum_name') or 'curriculum'} · {stage['name']}"
        if attempt:
            name += f" (retry {attempt})"
        return name + suffix

    def _source(self, run, robot, stage, attempt):
        """``(source_job_id, checkpoint, carry)`` this attempt starts from."""
        cap = model.caps(robot)
        if attempt > 0:
            failed = [r for r in run.get("results") or []
                      if r["stage_index"] == run["stage_index"] and r["attempt"] == attempt - 1]
            if not failed or not failed[-1].get("checkpoint"):
                raise RuntimeError("retry has no checkpoint from the previous attempt")
            return failed[-1]["train_job_id"], failed[-1]["checkpoint"], list(cap.get("carry") or [])
        carry = list(stage.get("carry") or [])
        init = stage["init"]
        if not carry or init == model.INIT_NONE:
            return None, None, []
        if init == model.INIT_PREVIOUS:
            done = [r for r in run.get("results") or []
                    if r["stage_index"] == run["stage_index"] - 1 and r["passed"]]
            if not done or not done[-1].get("checkpoint"):
                raise RuntimeError("the previous stage left no checkpoint to start from")
            return done[-1]["train_job_id"], done[-1]["checkpoint"], carry
        model_id = init[len(model.INIT_MODEL):]
        row = self.store.get_model(model_id)
        if not row or not row.get("location"):
            raise RuntimeError(f"init model {model_id} not found")
        return row.get("job_id"), row["location"], carry

    def _queue_train(self, run, robot):
        stage_index, attempt = run["stage_index"], run.get("attempt", 0)
        stage = run["stages"][stage_index]
        cap = model.caps(robot)
        if robot.get("engine") != "mjlab_rsl_rl":
            raise RuntimeError(f"{robot['id']} ({robot.get('engine')}) can't run curriculum stages yet")
        gym = self.store.get_gym(stage["gym"])
        if gym is None:
            raise RuntimeError(f"gym {stage['gym']} not found")
        source_job_id, checkpoint, carry = self._source(run, robot, stage, attempt)
        budget = stage["on_fail"]["extra_budget"] if attempt else stage["budget"]
        doc = {
            "robot": robot["id"], "job_type": "TRAIN",
            "name": self._job_name(run, stage, attempt),
            "gym_id": stage["gym"],
            "task_id": _strip_prefix(robot["id"], gym.get("task_id") or stage["gym"]),
            "gym_name": gym.get("name") or "",
            cap["budget_field"]: budget,
            "video": True, "progress_video": True,
            "stage_settings": dict(stage.get("settings") or {}),
            **dict(stage.get("settings") or {}),
            "handoff": {"source_job_id": source_job_id, "checkpoint": checkpoint, "carry": carry},
        }
        for key in ("reward_design_id", "experiment_design_id"):
            if stage.get(key):
                doc[key] = stage[key]
        if checkpoint:
            doc["init_checkpoint"] = checkpoint
        return self._queue(run, "train", doc, {})

    def _queue_eval(self, run, robot, train_job):
        cap = model.caps(robot)
        checkpoint = train_job.get(cap.get("checkpoint_field"))
        if not checkpoint:
            raise RuntimeError(f"train job {train_job['_id']} finished without a checkpoint")
        stage = run["stages"][run["stage_index"]]
        doc = {
            "robot": robot["id"], "job_type": "EVAL",
            "name": self._job_name(run, stage, run.get("attempt", 0), " · gate eval"),
            "task_id": train_job.get("task_id"),
            "source_job_id": str(train_job["_id"]),
            "checkpoint": checkpoint,
        }
        return self._queue(run, "eval", doc, {"train_job_id": str(train_job["_id"])})


def new_run(curriculum_doc, curriculum, start_stage, priority, now):
    """A RUNNING run document for a validated ``curriculum``."""
    return {
        "curriculum_id": str(curriculum_doc["_id"]),
        "curriculum_name": curriculum["name"],
        "curriculum_version": curriculum_doc.get("version", 1),
        "robot": curriculum["robot"],
        "stages": model.snapshot(curriculum),
        "options": {"start_stage": start_stage, "priority": priority},
        "status": "RUNNING",
        "stage_index": start_stage,
        "attempt": 0,
        "generation": 0,
        "pending": None,
        "results": [],
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
