"""curriculum-runner service: ticks every active curriculum run.

    python -u -m curriculum.service      (cwd = rl_agent, as in the container)
"""
import os
import pathlib
import signal
import sys
import threading

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from pymongo import MongoClient  # noqa: E402

import robots_registry  # noqa: E402
from curriculum.runner import Runner  # noqa: E402
from curriculum.store import MongoStore  # noqa: E402

POLL_S = float(os.environ.get("CURRICULUM_POLL_S", "10"))
SERVICE_ID = "curriculum-runner"


def main():
    db = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))[
        os.environ.get("DATABASE_NAME", "robotaxi")]
    store = MongoStore(db)
    robots = robots_registry.load_manifests()
    enabled = sorted(r for r, m in robots.items() if m["curriculum"]["enabled"])
    print(f"[curriculum] runner up; curricula enabled for {enabled or 'no robots'}; "
          f"polling every {POLL_S:g}s", flush=True)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    runner = Runner(store, robots, log=lambda m: print(m, flush=True))
    while not stopping.is_set():
        try:
            runner.tick()
            db.services.update_one({"_id": SERVICE_ID}, {"$set": {"last_seen": store.now()}}, upsert=True)
        except Exception as exc:  # noqa: BLE001 - Mongo blips must not kill the loop
            print(f"[curriculum] tick failed: {type(exc).__name__}: {exc}", flush=True)
        stopping.wait(POLL_S)


if __name__ == "__main__":
    main()
