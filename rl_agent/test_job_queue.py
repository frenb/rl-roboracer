"""Tests for cross-robot FIFO ordering in job_queue.

Needs a Mongo server (MONGO_URL); uses a throwaway database, never ``robotaxi``.
"""
import datetime
import os
import pathlib
import sys

import pytest
from pymongo import MongoClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import job_queue  # noqa: E402

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
NOW = T0 + datetime.timedelta(hours=1)


@pytest.fixture
def db():
    client = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@localhost:27017/"),
                         serverSelectionTimeoutMS=5000)
    database = client["job_queue_test"]
    client.drop_database(database.name)
    yield database
    client.drop_database(database.name)


def add_job(db, minute, status="NOT_STARTED", robot=None):
    doc = {"status": status, "create_date": T0 + datetime.timedelta(minutes=minute)}
    if robot is not None:
        doc["robot"] = robot
    return db.jobs.insert_one(doc).inserted_id


def alive(db, *robots, age_s=0):
    for r in robots:
        db.workers.update_one(
            {"_id": r},
            {"$set": {"last_seen": NOW - datetime.timedelta(seconds=age_s)}},
            upsert=True)


def claim(db, robot):
    return job_queue.may_claim(db, robot, now=NOW)[0]


def test_empty_queue_allows_both(db):
    alive(db, "roboracer", "microduck")
    assert claim(db, "roboracer")
    assert claim(db, "microduck")


def test_oldest_job_wins(db):
    alive(db, "roboracer", "microduck")
    add_job(db, 1, robot="microduck")
    add_job(db, 2)
    assert claim(db, "microduck")
    assert not claim(db, "roboracer")


def test_legacy_job_without_robot_is_roboracer(db):
    alive(db, "roboracer", "microduck")
    add_job(db, 1)
    add_job(db, 2, robot="microduck")
    assert claim(db, "roboracer")
    assert not claim(db, "microduck")


def test_running_job_blocks_other_robot_even_if_its_job_is_older(db):
    alive(db, "roboracer", "microduck")
    add_job(db, 1, robot="microduck")
    add_job(db, 2, status="IN_PROGRESS", robot="roboracer")
    assert claim(db, "roboracer")
    assert not claim(db, "microduck")


def test_dead_worker_does_not_block(db):
    alive(db, "roboracer")
    alive(db, "microduck", age_s=job_queue.HEARTBEAT_TTL_S + 1)
    add_job(db, 1, robot="microduck")
    add_job(db, 2, status="IN_PROGRESS", robot="microduck")
    add_job(db, 3)
    assert claim(db, "roboracer")


def test_unknown_worker_does_not_block(db):
    alive(db, "roboracer")
    add_job(db, 1, robot="microduck")
    add_job(db, 2)
    assert claim(db, "roboracer")


def test_paused_jobs_do_not_hold_the_queue(db):
    alive(db, "roboracer", "microduck")
    add_job(db, 1, status="PAUSED", robot="microduck")
    add_job(db, 2)
    assert claim(db, "roboracer")


def test_beat_marks_worker_live(db):
    job_queue.beat(db, "microduck")
    assert "microduck" in job_queue.live_robots(db)
