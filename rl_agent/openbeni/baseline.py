"""Registers open-beni's shipped checkpoint as an OpenBeni model.

``assets/models/policy.pt`` is the upstream jump-stage policy (15,000 PPO
updates, GRU-512). It is the reference that the plan's EVAL baseline and
the curriculum gate thresholds are measured against, and a warm start for
``Beni-Jump-Flat`` jobs.
"""
import os
import pathlib

ROBOT = "openbeni"
TASK_ID = "Beni-Jump-Flat"
ITERATIONS = 15000
CHECKPOINT = pathlib.Path(os.environ.get("OPENBENI_RL_DIR", "/openbeni_rl")) / "assets/models/policy.pt"


def model_doc(checkpoint=CHECKPOINT):
    return {
        "location": str(checkpoint),
        "task_id": TASK_ID,
        "training_iterations": ITERATIONS,
        "model_type": "PPO",
        "notes": "open-beni upstream pretrained policy (jump stage, 15k updates); baseline",
    }


def seed(db, now, checkpoint=CHECKPOINT):
    """Upsert the baseline model record. Returns the locations registered.

    No job produced it, so it has no job_id (the Models tab offers clips and
    VIDEO jobs only for models that have one).
    """
    if not pathlib.Path(checkpoint).is_file():
        return []
    doc = model_doc(checkpoint)
    db.models.update_one(
        {"robot": ROBOT, "location": doc["location"]},
        {"$set": {**doc, "updated_at": now},
         "$setOnInsert": {"robot": ROBOT, "create_date": now}},
        upsert=True)
    return [doc["location"]]
