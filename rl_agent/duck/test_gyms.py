"""Tests for the microduck gym registry (duck/gyms.py).

The sync tests need a Mongo server (MONGO_URL) and use a throwaway database,
never ``robotaxi``.
"""
import datetime
import os
import pathlib
import sys

import pytest
from pymongo import MongoClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from duck import gyms  # noqa: E402

T0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def task(task_id, terrain_type="plane"):
    return {"task_id": task_id, "terrain_type": terrain_type, "episode_s": 20.0,
            "reward_terms": ["track_linear_velocity"]}


@pytest.mark.parametrize("task_id, expected", [
    ("Mjlab-Velocity-Flat-MicroDuck",
     {"skill": "Velocity", "variant": "flat", "backlash": False, "rollers": False}),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck",
     {"skill": "Velocity", "variant": "rough", "backlash": True, "rollers": False}),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers",
     {"skill": "Velocity", "variant": "flat", "backlash": True, "rollers": True}),
    ("Mjlab-Velocity-Swizzle-MicroDuck",
     {"skill": "Velocity", "variant": "swizzle", "backlash": False, "rollers": False}),
])
def test_parse_task_id(task_id, expected):
    assert gyms.parse_task_id(task_id) == expected


@pytest.mark.parametrize("task_id", ["Mjlab-Velocity-Flat-Unitree-G1", "Velocity-Flat-MicroDuck", ""])
def test_non_duck_tasks_are_skipped(task_id):
    assert gyms.parse_task_id(task_id) is None
    assert gyms.gym_doc(task(task_id)) is None


def test_terrain_comes_from_config_not_name():
    # RollerSlope is named Flat but trains on generated slopes.
    assert gyms.gym_doc(task("Mjlab-RollerSlope-Flat-MicroDuck", "generator"))["terrain"] == "slope"
    assert gyms.gym_doc(task("Mjlab-Velocity-Rough-MicroDuck", "generator"))["terrain"] == "rough"
    assert gyms.gym_doc(task("Mjlab-Velocity-Flat-MicroDuck", "plane"))["terrain"] == "flat"


def test_gym_doc_name_and_fields():
    doc = gyms.gym_doc(task("Mjlab-Velocity-Flat-Backlash-MicroDuck"))
    assert doc["name"] == "MicroDuck · Walk (velocity tracking) · flat, backlash"
    assert doc["robot"] == "microduck"
    assert doc["file_path"] == ""
    assert doc["reward_terms"] == ["track_linear_velocity"]


@pytest.fixture
def db():
    client = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@localhost:27017/"),
                         serverSelectionTimeoutMS=5000)
    database = client["duck_gyms_test"]
    client.drop_database(database.name)
    yield database
    client.drop_database(database.name)


def test_sync_is_idempotent_and_leaves_car_gyms_alone(db):
    car = db.gyms.insert_one({"name": "Donut", "file_path": r"C:\gym.exe"}).inserted_id
    tasks = [task("Mjlab-Velocity-Flat-MicroDuck"), task("Mjlab-Spin-Flat-MicroDuck"),
             task("Mjlab-Velocity-Flat-Unitree-G1")]
    assert gyms.sync(db, tasks, now=T0) == (2, 0)
    assert gyms.sync(db, tasks, now=T0) == (2, 0)
    assert db.gyms.count_documents({"robot": "microduck"}) == 2
    assert db.gyms.find_one({"_id": car}) == {"_id": car, "name": "Donut", "file_path": r"C:\gym.exe"}


def test_sync_archives_vanished_tasks_and_keeps_operator_archives(db):
    gyms.sync(db, [task("Mjlab-Velocity-Flat-MicroDuck"), task("Mjlab-Spin-Flat-MicroDuck")], now=T0)
    db.gyms.update_one({"_id": "microduck:Mjlab-Velocity-Flat-MicroDuck"}, {"$set": {"archived": True}})

    assert gyms.sync(db, [task("Mjlab-Velocity-Flat-MicroDuck")], now=T0) == (1, 1)
    assert db.gyms.find_one({"_id": "microduck:Mjlab-Spin-Flat-MicroDuck"})["archived"] is True
    assert db.gyms.find_one({"_id": "microduck:Mjlab-Velocity-Flat-MicroDuck"})["archived"] is True


def test_task_for_job(db):
    gyms.sync(db, [task("Mjlab-Velocity-Rough-MicroDuck", "generator")], now=T0)
    car = db.gyms.insert_one({"name": "Donut", "file_path": r"C:\gym.exe"}).inserted_id

    assert gyms.task_for_job(db, {"task_id": "Mjlab-Spin-Flat-MicroDuck"}) == "Mjlab-Spin-Flat-MicroDuck"
    assert gyms.task_for_job(db, {"gym_id": "microduck:Mjlab-Velocity-Rough-MicroDuck"}) \
        == "Mjlab-Velocity-Rough-MicroDuck"
    assert gyms.task_for_job(db, {"gym_id": str(car)}) is None
    assert gyms.task_for_job(db, {}) is None
