"""Mongo access for the curriculum runner (the only part that needs pymongo)."""
import datetime

from bson import ObjectId

import job_queue


def _id_filter(value):
    value = str(value)
    return {"_id": {"$in": [ObjectId(value), value]}} if ObjectId.is_valid(value) else {"_id": value}


class MongoStore:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def now():
        return datetime.datetime.now(datetime.timezone.utc)

    def active_runs(self):
        return list(self.db.curriculum_runs.find({"status": {"$in": ["RUNNING", "CANCEL_REQUESTED"]}}))

    def get_run(self, run_id):
        return self.db.curriculum_runs.find_one(_id_filter(run_id))

    def update_run(self, run_id, fields, push_result=None):
        update = {"$set": {**fields, "updated_at": self.now()}}
        if push_result is not None:
            update["$push"] = {"results": push_result}
        self.db.curriculum_runs.update_one(_id_filter(run_id), update)

    def get_job(self, job_id):
        return self.db.jobs.find_one(_id_filter(job_id))

    def find_job_by_key(self, key):
        return self.db.jobs.find_one({"curriculum_key": key}, {"_id": 1})

    def insert_job(self, doc):
        return str(self.db.jobs.insert_one(doc).inserted_id)

    def set_job_status(self, job_id, status, only_from=()):
        query = _id_filter(job_id)
        if only_from:
            query["status"] = {"$in": list(only_from)}
        self.db.jobs.update_one(query, {"$set": {"status": status}})

    def queue_head_date(self):
        head = job_queue.oldest(self.db, {"status": "NOT_STARTED"})
        return job_queue.created_at(head) if head else None

    def get_gym(self, gym_id):
        return self.db.gyms.find_one({"_id": str(gym_id)})

    def get_model(self, model_id):
        return self.db.models.find_one(_id_filter(model_id))
