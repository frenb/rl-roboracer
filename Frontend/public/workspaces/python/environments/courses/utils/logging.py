from pymongo import MongoClient
import os

client = MongoClient('mongo', 
    username='root',
    password='example')
db = client[os.environ['DATABASE_NAME']]

def log_reward(job_id, type, score, diff=None, extra_data=None, step_costs=[], position_history=[], stat_array=[]):
    dilimeter = ","
    step_costs_valid = None if len(step_costs) == 0 else dilimeter.join([str(i) for i in step_costs])
    position_history_valid = None if len(position_history) == 0 else dilimeter.join([str(i) for i in position_history])
    stat_array_valid = None if len(stat_array) == 0 else dilimeter.join([str(i) for i in stat_array])
    new_log = {
        "job_id": job_id,
        "type": type,
        "score": score,
        "diff": diff,
        "extra_data": extra_data,
        "step_costs": step_costs_valid,
        "position_history": position_history_valid,
        "stat_array": stat_array_valid
    }
    db.logs.insert_one(new_log)

def log_blob(blob):
    db.logs.insert_one(blob) 