"""Registers open-beni tasks as ``gyms`` documents.

One gym per registered mjlab task, keyed ``openbeni:<task_id>``. Car and
microduck gyms are left alone.
"""
import datetime

ROBOT = "openbeni"
ID_PREFIX = f"{ROBOT}:"
SKILL_LABELS = {
    "Velocity": "Locomotion + recovery",
    "Jump": "Recovery-gated jump",
}


def gym_id(task_id):
    return ID_PREFIX + task_id


def parse_task_id(task_id):
    """``{skill, terrain}`` for a Beni task id, else None."""
    if not isinstance(task_id, str) or not task_id.startswith("Beni-"):
        return None
    parts = task_id.split("-")
    # Beni-Velocity-Flat / Beni-Jump-Flat
    if len(parts) < 3:
        return None
    return {"skill": parts[1], "terrain": parts[2].lower()}


def gym_doc(task):
    """Gym fields for one task as listed by list_tasks.py, or None if not Beni."""
    parsed = parse_task_id(task["task_id"])
    if parsed is None:
        return None
    skill = SKILL_LABELS.get(parsed["skill"], parsed["skill"])
    return {
        "robot": ROBOT,
        "task_id": task["task_id"],
        "name": f"OpenBeni · {skill} · {parsed['terrain']}",
        "engine": "mjlab",
        "skill": parsed["skill"],
        "terrain": parsed["terrain"],
        "episode_s": task.get("episode_s"),
        "reward_terms": task.get("reward_terms") or [],
        "reward_defaults": task.get("reward_defaults") or [],
        "file_path": "",
        "description": f"mjlab task {task['task_id']} (open-beni)",
        "managed_by": "beni-worker",
    }


def sync(db, tasks, now=None):
    """Upsert a gym per Beni task; archive managed gyms whose task is gone."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    seen = []
    for task in tasks:
        doc = gym_doc(task)
        if doc is None:
            continue
        _id = gym_id(doc["task_id"])
        seen.append(_id)
        db.gyms.update_one(
            {"_id": _id},
            {"$set": {**doc, "updated_at": now},
             "$setOnInsert": {"created_at": now, "archived": False}},
            upsert=True)
    gone = db.gyms.update_many(
        {"robot": ROBOT, "managed_by": "beni-worker", "_id": {"$nin": seen}, "archived": {"$ne": True}},
        {"$set": {"archived": True, "updated_at": now}})
    return len(seen), gone.modified_count


def task_for_job(db, job):
    """The task an openbeni job runs: its own ``task_id``, else its gym's."""
    if job.get("task_id"):
        return job["task_id"]
    gid = job.get("gym_id")
    if not gid:
        return None
    gym = db.gyms.find_one({"_id": str(gid)}, {"robot": 1, "task_id": 1})
    if gym and gym.get("robot") == ROBOT:
        return gym.get("task_id")
    return None
