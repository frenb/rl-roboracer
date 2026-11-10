"""Registers the microduck tasks as ``gyms`` documents.

One gym per registered mjlab task, keyed ``microduck:<task_id>`` so the
duck worker can re-sync at every start without duplicating rows. Car gyms
(Unity builds added from the Gyms tab) have ObjectId keys and no ``robot``
field; nothing here touches them.

Task ids follow ``Mjlab-<Skill>-<Flat|Rough|Swizzle>[-Backlash]-MicroDuck[-Rollers]``.
Terrain comes from the task's config rather than its name: RollerSlope is
named Flat but trains on generated slopes.

Pure functions apart from ``sync`` so the parsing runs in CPU-only CI.
"""
import datetime
import re

ROBOT = "microduck"
ID_PREFIX = f"{ROBOT}:"
_TASK_RE = re.compile(
    r"^Mjlab-(?P<skill>[A-Za-z]+)-(?P<variant>[A-Za-z]+)(?P<backlash>-Backlash)?"
    r"-MicroDuck(?P<rollers>-Rollers)?$")
# Skill names as the fork spells them -> what the dashboard shows.
SKILL_LABELS = {
    "Velocity": "Walk (velocity tracking)",
    "VelStand": "Walk and stand",
    "StandUp": "Stand up",
    "SitStand": "Sit and stand",
    "GroundPick": "Pick from the ground",
    "BallKick": "Kick a ball",
    "Spin": "Spin",
    "Roulade": "Roll",
    "RollerCrouch": "Roller crouch",
    "RollerSlope": "Roller slope",
    "RollerStandUp": "Roller stand up",
}


def gym_id(task_id):
    return ID_PREFIX + task_id


def parse_task_id(task_id):
    """``{skill, variant, backlash, rollers}`` for a microduck task id, else None."""
    m = _TASK_RE.match(task_id)
    if not m:
        return None
    return {
        "skill": m["skill"],
        "variant": m["variant"].lower(),
        "backlash": bool(m["backlash"]),
        "rollers": bool(m["rollers"]),
    }


def terrain_of(variant, terrain_type):
    """``flat`` for a plane; otherwise the generated terrain's flavour."""
    if terrain_type != "generator":
        return "flat"
    return "rough" if variant == "rough" else "slope"


def gym_doc(task):
    """Gym fields for one task as listed by list_tasks.py, or None if not a duck task."""
    parsed = parse_task_id(task["task_id"])
    if parsed is None:
        return None
    terrain = terrain_of(parsed["variant"], task.get("terrain_type"))
    skill = SKILL_LABELS.get(parsed["skill"], parsed["skill"])
    extras = [terrain]
    if parsed["variant"] == "swizzle":
        extras.append("swizzle")
    if parsed["backlash"]:
        extras.append("backlash")
    if parsed["rollers"]:
        extras.append("rollers")
    return {
        "robot": ROBOT,
        "task_id": task["task_id"],
        "name": f"MicroDuck · {skill} · {', '.join(extras)}",
        "skill": parsed["skill"],
        "terrain": terrain,
        "backlash": parsed["backlash"],
        "rollers": parsed["rollers"],
        "episode_s": task.get("episode_s"),
        "reward_terms": task.get("reward_terms") or [],
        "file_path": "",
        "description": f"mjlab task {task['task_id']}",
        "managed_by": "duck-worker",
    }


def sync(db, tasks, now=None):
    """Upsert a gym per duck task; archive managed gyms whose task is gone.

    Returns ``(upserted, archived)`` counts. An operator's archive of a gym
    whose task still exists is kept: only fields derived from the task are
    rewritten.
    """
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
        {"robot": ROBOT, "managed_by": "duck-worker", "_id": {"$nin": seen}, "archived": {"$ne": True}},
        {"$set": {"archived": True, "updated_at": now}})
    return len(seen), gone.modified_count


def task_for_job(db, job):
    """The task a duck job runs: its own ``task_id``, else its gym's."""
    if job.get("task_id"):
        return job["task_id"]
    gid = job.get("gym_id")
    if not gid:
        return None
    gym = db.gyms.find_one({"_id": str(gid)}, {"robot": 1, "task_id": 1})
    if gym and gym.get("robot") == ROBOT:
        return gym.get("task_id")
    return None
