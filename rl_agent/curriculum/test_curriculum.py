"""Curriculum validation and runner tests (no Mongo, no GPU)."""
import copy
import datetime
import itertools
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import robots_registry  # noqa: E402
from curriculum import model  # noqa: E402
from curriculum.runner import Runner, job_key, new_run  # noqa: E402

T0 = datetime.datetime(2026, 11, 17, tzinfo=datetime.timezone.utc)
SUPPORTS = {c: False for c in robots_registry.REQUIRED_SUPPORTS}


def manifest(rid, engine="mjlab_rsl_rl", **curriculum):
    return robots_registry.validate_manifest({
        "id": rid, "display_name": rid, "morphology": "x", "engine": engine,
        "checkpoint_format": "pt", "supports": SUPPORTS, "curriculum": curriculum})


BENI = manifest("openbeni", enabled=True, budget_field="max_iterations",
                checkpoint_field="openbeni_checkpoint",
                carry=["weights", "optimizer", "entropy", "normalizer"], carry_atomic=True,
                gate_metrics=["fall_rate", "mean_return"])
CAR = manifest("roboracer", engine="tf_agents_unity", enabled=True, budget_field="num_iterations",
               checkpoint_field="learner_dir", carry=["weights", "optimizer", "entropy", "replay"])
OFF = manifest("microduck", enabled=False, budget_field="max_iterations")
ROBOTS = {r["id"]: r for r in (BENI, CAR, OFF)}

CONTEXT = {
    "gyms": {"openbeni:Beni-Velocity-Flat": {"robot": "openbeni"},
             "openbeni:Beni-Jump-Flat": {"robot": "openbeni"},
             "microduck:Walk": {"robot": "microduck"},
             "openbeni:Old": {"robot": "openbeni", "archived": True}},
    "reward_designs": {"rd-beni": {"robot": "openbeni"}, "rd-car": {}},
    "experiment_designs": {"ed-beni": {"robot": "openbeni"}},
    "models": {"m1": {"robot": "openbeni", "location": "/logs/m1.pt"}},
}


def beni_curriculum(**overrides):
    doc = {
        "name": "Beni low-level",
        "stages": [
            {"name": "Balance", "gym": "openbeni:Beni-Velocity-Flat", "budget": 100,
             "reward_design_id": "rd-beni",
             "gate": {"conditions": [{"metric": "fall_rate", "op": "<", "value": 0.05}]}},
            {"name": "Jump", "gym": "openbeni:Beni-Jump-Flat",
             "gate": {"conditions": [{"metric": "mean_return", "op": ">=", "value": 10}]},
             "on_fail": {"retries": 1, "extra_budget": 50}},
        ],
    }
    doc.update(overrides)
    return doc


def problems(doc, robot=BENI, context=CONTEXT):
    with pytest.raises(model.CurriculumError) as exc:
        model.validate(doc, robot, context)
    return " | ".join(exc.value.problems)


# ---- validation ------------------------------------------------------------

def test_valid_curriculum_resolves_inheritance_and_default_carry():
    cur = model.validate(beni_curriculum(), BENI, CONTEXT)
    s1, s2 = cur["stages"]
    assert [s1["id"], s2["id"]] == ["balance", "jump"]
    assert s1["init"] == "none" and s1["carry"] == []
    assert s2["init"] == "previous"
    assert s2["budget"] == 100 and s2["reward_design_id"] == "rd-beni"
    assert s2["carry"] == ["weights", "optimizer", "entropy", "normalizer"]
    assert s2["on_fail"] == {"retries": 1, "extra_budget": 50}
    assert s1["on_fail"] == {"retries": 0, "extra_budget": 100}


def test_explicit_null_design_overrides_inheritance():
    doc = beni_curriculum()
    doc["stages"][1]["reward_design_id"] = None
    assert model.validate(doc, BENI, CONTEXT)["stages"][1]["reward_design_id"] is None


def test_disabled_robot_and_missing_fields():
    msg = problems({"name": "x", "stages": [{"gym": "microduck:Walk", "budget": 5}]}, robot=OFF)
    assert "aren't enabled for microduck" in msg
    msg = problems({"stages": [{"budget": 0}]})
    assert "name is required" in msg and "gym is required" in msg and "budget must be" in msg


def test_reference_checks():
    doc = beni_curriculum()
    doc["stages"][0]["gym"] = "microduck:Walk"
    doc["stages"][0]["reward_design_id"] = "rd-car"
    doc["stages"][1]["gym"] = "openbeni:Old"
    msg = problems(doc)
    assert "is a microduck gym" in msg
    assert "is a roboracer design" in msg
    assert "is archived" in msg


def test_first_stage_cannot_start_from_previous():
    doc = beni_curriculum()
    doc["stages"][0]["init"] = "previous"
    assert "first stage has no previous stage" in problems(doc)


def test_atomic_carry_is_all_or_nothing():
    doc = beni_curriculum()
    doc["stages"][1]["carry"] = ["weights"]
    assert "carry all of them or none" in problems(doc)
    doc["stages"][1]["carry"] = []
    assert model.validate(doc, BENI, CONTEXT)["stages"][1]["carry"] == []


def test_carry_needs_an_init_and_a_supporting_trainer():
    doc = beni_curriculum()
    doc["stages"][1]["init"] = "none"
    doc["stages"][1]["carry"] = ["weights", "optimizer", "entropy", "normalizer"]
    assert "needs init 'previous'" in problems(doc)
    doc["stages"][1]["carry"] = ["replay"]
    doc["stages"][1]["init"] = "previous"
    assert "can't restore 'replay'" in problems(doc)


def car_doc(second):
    return {"name": "track", "stages": [
        {"name": "Easy", "gym": "g", "budget": 10, "reward_design_id": "r"}, second]}


def test_replay_carry_rules():
    ok = model.validate(car_doc({"name": "Hard"}), CAR)["stages"][1]
    assert ok["carry"] == ["weights", "optimizer", "entropy", "replay"]
    changed_gym = model.validate(car_doc({"name": "Cam", "gym": "g2"}), CAR)["stages"][1]
    assert "replay" not in changed_gym["carry"]
    assert "same gym" in problems(car_doc({"name": "Cam", "gym": "g2", "carry": ["weights", "replay"]}), CAR, {})
    assert "stale" in problems(car_doc({"name": "R", "reward_design_id": "r2", "carry": ["replay"]}), CAR, {})
    assert "needs init 'previous'" in problems(car_doc({"name": "M", "init": "model:m1", "carry": ["replay"]}), CAR, {})


def test_gate_validation():
    doc = beni_curriculum()
    doc["stages"][0]["gate"] = {"conditions": [{"metric": "goals", "op": "=>", "value": "9"}], "consecutive": 3}
    msg = problems(doc)
    assert "isn't reported" in msg and "op '=>'" in msg and "must be a number" in msg
    assert "consecutive > 1 needs early stop" in msg


def test_retries_need_a_gate():
    doc = beni_curriculum()
    doc["stages"][0]["gate"] = None
    doc["stages"][0]["on_fail"] = {"retries": 2}
    assert "retries needs a gate" in problems(doc)


def test_unknown_stage_settings_and_fields():
    doc = beni_curriculum()
    doc["stages"][0]["settings"] = {"cmd_lin_vel_x_max": 0.5}
    doc["stages"][0]["colour"] = "red"
    msg = problems(doc)
    assert "no stage setting 'cmd_lin_vel_x_max'" in msg and "unknown fields ['colour']" in msg


def test_start_stage():
    cur = model.validate(beni_curriculum(), BENI, CONTEXT)
    assert model.start_stage(cur, None) == 0
    assert model.start_stage(cur, "balance") == 0
    with pytest.raises(model.CurriculumError, match="previous stage's checkpoint"):
        model.start_stage(cur, 1)
    doc = beni_curriculum()
    doc["stages"][1]["init"] = "model:m1"
    assert model.start_stage(model.validate(doc, BENI, CONTEXT), "jump") == 1
    with pytest.raises(model.CurriculumError):
        model.start_stage(cur, 5)


def test_gate_result():
    gate = {"conditions": [{"metric": "fall_rate", "op": "<", "value": 0.05},
                           {"metric": "mean_return", "op": ">=", "value": 10}], "consecutive": 1}
    assert model.gate_result(gate, {"fall_rate": 0.01, "mean_return": 12})[0] is True
    passed, details = model.gate_result(gate, {"fall_rate": 0.01})
    assert passed is False and details[1]["actual"] is None
    assert model.gate_result(None, {}) == (True, [])


# ---- runner ------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        self.runs, self.jobs = {}, {}
        self.gyms = {"openbeni:Beni-Velocity-Flat": {"_id": "openbeni:Beni-Velocity-Flat", "robot": "openbeni",
                                                     "task_id": "Beni-Velocity-Flat", "name": "Velocity"},
                     "openbeni:Beni-Jump-Flat": {"_id": "openbeni:Beni-Jump-Flat", "robot": "openbeni",
                                                 "task_id": "Beni-Jump-Flat", "name": "Jump"}}
        self.models = {"m1": {"_id": "m1", "location": "/logs/m1.pt", "job_id": "j0"}}
        self.ids = itertools.count(1)
        self.clock = T0

    def now(self):
        self.clock += datetime.timedelta(seconds=1)
        return self.clock

    def active_runs(self):
        return [copy.deepcopy(r) for r in self.runs.values() if r["status"] in ("RUNNING", "CANCEL_REQUESTED")]

    def get_run(self, run_id):
        return copy.deepcopy(self.runs.get(run_id))

    def update_run(self, run_id, fields, push_result=None):
        self.runs[run_id].update(copy.deepcopy(fields))
        if push_result is not None:
            self.runs[run_id]["results"].append(copy.deepcopy(push_result))

    def get_job(self, job_id):
        return copy.deepcopy(self.jobs.get(str(job_id)))

    def find_job_by_key(self, key):
        return next((j for j in self.jobs.values() if j.get("curriculum_key") == key), None)

    def insert_job(self, doc):
        jid = f"j{next(self.ids)}"
        self.jobs[jid] = {**copy.deepcopy(doc), "_id": jid}
        return jid

    def set_job_status(self, job_id, status, only_from=()):
        job = self.jobs.get(str(job_id))
        if job and (not only_from or job["status"] in only_from):
            job["status"] = status

    def queue_head_date(self):
        dates = [j["create_date"] for j in self.jobs.values() if j["status"] == "NOT_STARTED"]
        return min(dates) if dates else None

    def get_gym(self, gym_id):
        return self.gyms.get(gym_id)

    def get_model(self, model_id):
        return self.models.get(model_id)


def start(store, doc=None, start_stage=0, priority="front"):
    cur = model.validate(doc or beni_curriculum(), BENI, CONTEXT)
    run = new_run({"_id": "c1", "version": 3}, cur, start_stage, priority, store.now())
    run["_id"] = "r1"
    store.runs["r1"] = run
    return Runner(store, ROBOTS, log=lambda m: None)


def pending_job(store):
    return store.jobs[store.runs["r1"]["pending"]["job_id"]]


def finish(store, job, **fields):
    job.update(status="DONE", **fields)


def test_happy_path_chains_train_eval_and_warm_starts():
    store = FakeStore()
    runner = start(store)
    runner.tick()
    t1 = pending_job(store)
    assert t1["job_type"] == "TRAIN" and t1["task_id"] == "Beni-Velocity-Flat"
    assert t1["max_iterations"] == 100 and t1["reward_design_id"] == "rd-beni"
    assert "init_checkpoint" not in t1 and t1["handoff"]["carry"] == []
    assert t1["curriculum_key"] == job_key("r1", 0, 0, "train")

    runner.tick()
    assert pending_job(store) is t1, "nothing happens while the job runs"

    finish(store, t1, openbeni_checkpoint="/logs/run1/model_99.pt")
    runner.tick()
    e1 = pending_job(store)
    assert e1["job_type"] == "EVAL" and e1["checkpoint"] == "/logs/run1/model_99.pt"
    assert e1["source_job_id"] == t1["_id"]

    finish(store, e1, eval_results={"fall_rate": 0.01, "mean_return": 3})
    runner.tick()
    t2 = pending_job(store)
    assert store.runs["r1"]["stage_index"] == 1
    assert t2["task_id"] == "Beni-Jump-Flat" and t2["init_checkpoint"] == "/logs/run1/model_99.pt"
    assert t2["handoff"] == {"source_job_id": t1["_id"], "checkpoint": "/logs/run1/model_99.pt",
                             "carry": ["weights", "optimizer", "entropy", "normalizer"]}

    finish(store, t2, openbeni_checkpoint="/logs/run2/model_99.pt")
    runner.tick()
    finish(store, pending_job(store), eval_results={"fall_rate": 0.02, "mean_return": 12})
    runner.tick()
    run = store.runs["r1"]
    assert run["status"] == "PASSED" and run["pending"] is None
    assert [r["passed"] for r in run["results"]] == [True, True]
    assert run["results"][1]["gate"][0]["actual"] == 12


def test_missed_gate_retries_from_the_failed_attempt_then_stops():
    store = FakeStore()
    runner = start(store, start_stage=0)
    runner.tick()
    finish(store, pending_job(store), openbeni_checkpoint="/a.pt")
    runner.tick()
    finish(store, pending_job(store), eval_results={"fall_rate": 0.0})
    runner.tick()
    finish(store, pending_job(store), openbeni_checkpoint="/b.pt")
    runner.tick()
    finish(store, pending_job(store), eval_results={"mean_return": 5})
    runner.tick()
    retry = pending_job(store)
    assert store.runs["r1"]["attempt"] == 1
    assert retry["max_iterations"] == 50 and retry["init_checkpoint"] == "/b.pt"
    assert retry["name"].endswith("(retry 1)")
    finish(store, retry, openbeni_checkpoint="/c.pt")
    runner.tick()
    finish(store, pending_job(store), eval_results={"mean_return": 6})
    runner.tick()
    run = store.runs["r1"]
    assert run["status"] == "FAILED_GATE"
    assert "after 2 attempt(s)" in run["error"]


def test_restart_does_not_queue_twice():
    store = FakeStore()
    runner = start(store)
    runner.tick()
    first = store.runs["r1"]["pending"]["job_id"]
    store.runs["r1"]["pending"] = None  # crash between insert and run update
    runner.tick()
    assert store.runs["r1"]["pending"]["job_id"] == first
    assert len(store.jobs) == 1


def test_failed_and_cancelled_jobs_end_the_run():
    store = FakeStore()
    runner = start(store)
    runner.tick()
    pending_job(store).update(status="FAILED", eval_error="boom")
    runner.tick()
    assert store.runs["r1"]["status"] == "FAILED" and "boom" in store.runs["r1"]["error"]

    store = FakeStore()
    runner = start(store)
    runner.tick()
    pending_job(store)["status"] = "CANCELLED"
    runner.tick()
    assert store.runs["r1"]["status"] == "CANCELLED"


def test_cancel_request_cancels_the_pending_job():
    store = FakeStore()
    runner = start(store)
    runner.tick()
    job = pending_job(store)
    job["status"] = "IN_PROGRESS"
    store.runs["r1"]["status"] = "CANCEL_REQUESTED"
    runner.tick()
    assert job["status"] == "CANCELLED"
    assert store.runs["r1"]["status"] == "CANCELLED"


def test_front_priority_jumps_the_queue_and_normal_appends():
    store = FakeStore()
    store.jobs["other"] = {"_id": "other", "status": "NOT_STARTED", "create_date": T0 - datetime.timedelta(hours=1)}
    start(store).tick()
    assert pending_job(store)["create_date"] < store.jobs["other"]["create_date"]

    store = FakeStore()
    store.jobs["other"] = {"_id": "other", "status": "NOT_STARTED", "create_date": T0 - datetime.timedelta(hours=1)}
    start(store, priority="normal").tick()
    assert pending_job(store)["create_date"] > store.jobs["other"]["create_date"]


def test_start_from_model_and_ungated_last_stage():
    store = FakeStore()
    doc = beni_curriculum()
    doc["stages"][1]["init"] = "model:m1"
    doc["stages"][1]["gate"] = None
    doc["stages"][1]["on_fail"] = {}
    runner = start(store, doc, start_stage=1)
    runner.tick()
    job = pending_job(store)
    assert job["init_checkpoint"] == "/logs/m1.pt" and job["handoff"]["source_job_id"] == "j0"
    finish(store, job, openbeni_checkpoint="/x.pt")
    runner.tick()
    assert store.runs["r1"]["status"] == "PASSED"
    assert store.runs["r1"]["results"][0]["eval_job_id"] is None


def test_robot_without_curricula_fails_the_run():
    store = FakeStore()
    runner = start(store)
    runner.robots = {"openbeni": OFF}
    runner.tick()
    assert store.runs["r1"]["status"] == "FAILED"
