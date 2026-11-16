"""Run the MadScientist test files one at a time against a fresh database.

The tests share the ``robotaxi`` database and some enforce per-day quotas
(e.g. max proposals per day), so running them in a single process lets one
file's fixtures trip another file's guards. Each file gets a dropped and
re-seeded database instead.

Usage: MONGO_URL=mongodb://root:example@localhost:27017/ python scripts/ci/run_madscientist_tests.py
"""
import os
import pathlib
import subprocess
import sys

from pymongo import MongoClient

REPO = pathlib.Path(__file__).resolve().parents[2]
TEST_DIR = REPO / "rl_agent" / "madscientist"


def main() -> int:
    url = os.environ.get("MONGO_URL", "mongodb://root:example@localhost:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    env = {**os.environ, "MONGO_URL": url, "PYTHONPATH": str(REPO)}
    failed = []
    for test in sorted(TEST_DIR.glob("test_*.py")):
        client.drop_database("robotaxi")
        subprocess.run([sys.executable, "-m", "rl_agent.madscientist.seed"],
                       cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)
        print(f"=== {test.name}", flush=True)
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test)],
                            cwd=REPO, env=env).returncode
        if rc == 5:  # no pytest functions: script-style test with its own main
            module = f"rl_agent.madscientist.{test.stem}"
            rc = subprocess.run([sys.executable, "-m", module],
                                cwd=REPO, env=env).returncode
        if rc != 0:
            failed.append(test.name)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All MadScientist test files passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
