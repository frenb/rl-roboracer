"""Cross-robot ordering for the shared ``jobs`` queue.

Every robot (``roboracer``, ``microduck``) has its own worker polling the
same ``jobs`` collection. They share one GPU, so the queue is drained
strictly first-come-first-served across robots: a worker only claims work
when the oldest queued job is its own and no other robot's job is running.

A job keeps its place until it is DONE, FAILED, CANCELLED or PAUSED, even
while its robot's worker is down: the queue waits for that worker rather
than run later jobs first. Workers publish a heartbeat to ``db.workers`` so
the wait reason can say the blocking robot's worker is offline.

Jobs without a ``robot`` field predate multi-robot support and belong to
``roboracer``.
"""
import datetime
import socket
import threading

from bson import ObjectId

DEFAULT_ROBOT = "roboracer"
HEARTBEAT_INTERVAL_S = 30
HEARTBEAT_TTL_S = 120


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def robot_values(robot):
    """Values of ``jobs.robot`` that belong to ``robot``."""
    return [None, DEFAULT_ROBOT] if robot == DEFAULT_ROBOT else [robot]


def robot_filter(robot):
    return {"robot": {"$in": robot_values(robot)}}


def job_robot(job):
    return job.get("robot") or DEFAULT_ROBOT


def beat(db, robot):
    db.workers.update_one(
        {"_id": robot},
        {"$set": {"last_seen": _now(), "host": socket.gethostname()}},
        upsert=True)


def start_heartbeat(db, robot, interval_s=HEARTBEAT_INTERVAL_S):
    """Beat from a daemon thread so long training runs stay visible."""
    beat(db, robot)

    def _loop():
        while True:
            threading.Event().wait(interval_s)
            try:
                beat(db, robot)
            except Exception as exc:  # noqa: BLE001 - Mongo blips must not kill the thread
                print(f"job_queue heartbeat failed: {exc}", flush=True)

    t = threading.Thread(target=_loop, name=f"{robot}-heartbeat", daemon=True)
    t.start()
    return t


_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def created_at(job):
    """When ``job`` was queued, comparable across how it was written.

    ``create_date`` is a BSON date (duck worker), an ISO string (car trainer
    follow-ups) or missing (dashboard and older jobs). Mongo sorts those
    types apart (missing < string < date) rather than by time, so FIFO
    order is computed here, falling back to the ObjectId's timestamp.
    """
    value = job.get("create_date")
    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            value = None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    oid = job.get("_id")
    if isinstance(oid, str) and ObjectId.is_valid(oid):
        oid = ObjectId(oid)
    return oid.generation_time if isinstance(oid, ObjectId) else _EPOCH


def oldest(db, query, projection=None):
    """The earliest-queued job matching ``query`` by ``created_at``, or None."""
    fields = {"create_date": 1, **(projection or {})}
    return min(db.jobs.find(query, fields), key=created_at, default=None)


def live_robots(db, now=None):
    cutoff = (now or _now()) - datetime.timedelta(seconds=HEARTBEAT_TTL_S)
    return {w["_id"] for w in db.workers.find({"last_seen": {"$gte": cutoff}}, {"_id": 1})}


def may_claim(db, robot, now=None):
    """Return ``(allowed, reason)`` for ``robot``'s worker picking up work.

    Allowed when the robot already has a job IN_PROGRESS (resume after a
    restart), or when no other robot's job is running and the oldest
    NOT_STARTED job is this robot's.
    """
    mine = robot_filter(robot)
    running = db.jobs.find_one({"status": "IN_PROGRESS", "robot": {"$nin": robot_values(robot)}},
                               {"robot": 1})
    if running is not None:
        return False, _blocked(db, running, "is running", now)

    if db.jobs.find_one({"status": "IN_PROGRESS", **mine}, {"_id": 1}):
        return True, "own job in progress"

    head = oldest(db, {"status": "NOT_STARTED"}, {"robot": 1})
    if head is None or job_robot(head) == robot:
        return True, "head of queue"
    return False, _blocked(db, head, "is ahead in the queue", now)


def _blocked(db, job, what, now):
    owner = job_robot(job)
    offline = "" if owner in live_robots(db, now) else f" ({owner} worker is offline)"
    return f"{owner} job {job['_id']} {what}{offline}"
