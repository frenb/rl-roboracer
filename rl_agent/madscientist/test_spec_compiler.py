"""Tests for the spec compiler loop, with a scripted LLM and a scratch database.

Run in the madscientist container:
    docker exec rl-roboracer-madscientist-1 python -m unittest rl_agent.madscientist.test_spec_compiler -v
"""
import datetime
import json
import os
import unittest

from pymongo import MongoClient

from rl_agent.madscientist import spec_compiler
from rl_agent.madscientist.spec_compiler import specs

CAR_CODE = '''def reward_standard(data, data_arr, step_costs, *, defaults, course):
    return 0.01 * data["car"]["speed"] - 0.02 * abs(data["car"]["dist_from_traj"])


def reward_failure(data, data_arr, step_costs, position_history, *, defaults, course):
    return -1.0
'''
SPEC = specs.TEMPLATE.replace("robots: [roboracer, microduck]", "robots: [roboracer]")


def answer(code, spec_text=SPEC, drop=0):
    spec = specs.parse(spec_text)
    rationale = [{"section": s, "bullet": b, "implementation": "x", "terms": []}
                 for s, b in spec_compiler.targets.bullets(spec)]
    return f"```python\n{code}```\n\n```json\n{json.dumps({'rationale': rationale[drop:], 'unsupported': []})}\n```"


class ScriptedLlm:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, client, system, messages, max_tokens):
        self.calls.append(messages)
        return {"text": self.texts.pop(0), "input_tokens": 1000, "output_tokens": 500}


class SpecCompilerTest(unittest.TestCase):
    def setUp(self):
        self.client = MongoClient(os.environ.get("MONGO_URL", "mongodb://root:example@mongo:27017/"))
        self.client.drop_database("spec_compiler_test")
        self.db = self.client.spec_compiler_test

    def tearDown(self):
        self.client.drop_database("spec_compiler_test")
        self.client.close()

    def queue(self, text=SPEC, robot="roboracer"):
        return self.db.design_specs.insert_one({
            "name": "t", "text": text,
            "compile": {robot: {"state": "queued", "requested_at": datetime.datetime.now()}}}).inserted_id

    def run_once(self, llm, budget=100.0):
        return spec_compiler.run_once(self.db, None, monthly_budget_usd=budget, call=llm)

    def compile_state(self, _id, robot="roboracer"):
        return self.db.design_specs.find_one({"_id": _id})["compile"][robot]

    def test_good_answer_becomes_pending(self):
        _id = self.queue()
        llm = ScriptedLlm(answer(CAR_CODE))
        self.assertTrue(self.run_once(llm))
        c = self.compile_state(_id)
        self.assertEqual(c["state"], "pending")
        self.assertEqual(c["result"]["code"], CAR_CODE)
        self.assertEqual(c["result"]["inputs_sha"], specs.parse(SPEC).inputs_sha("roboracer"))
        self.assertEqual(self.db.spec_compile_runs.count_documents({}), 1)
        self.assertFalse(self.run_once(llm))

    def test_problems_go_back_to_the_llm(self):
        _id = self.queue()
        llm = ScriptedLlm(answer(CAR_CODE.replace("-1.0", "5.0"), drop=1), answer(CAR_CODE))
        self.run_once(llm)
        c = self.compile_state(_id)
        self.assertEqual(c["state"], "pending")
        self.assertEqual(len(c["result"]["rounds"]), 2)
        feedback = llm.calls[1][-1]["content"]
        self.assertIn("neither implemented nor listed", feedback)
        self.assertIn("crashing", feedback)

    def test_gives_up_after_max_rounds(self):
        _id = self.queue()
        self.run_once(ScriptedLlm(*["no code here"] * spec_compiler.MAX_ROUNDS))
        c = self.compile_state(_id)
        self.assertEqual(c["state"], "failed")
        self.assertIn("python", c["result"]["problems"][0])
        self.assertEqual(self.db.spec_compile_runs.count_documents({}), spec_compiler.MAX_ROUNDS)

    def test_spent_budget_skips_the_llm(self):
        self.db.spec_compile_runs.insert_one({"cost_usd": 5.0, "created_at": datetime.datetime.now(datetime.timezone.utc)})
        _id = self.queue()
        llm = ScriptedLlm()
        self.run_once(llm, budget=5.0)
        self.assertEqual(self.compile_state(_id)["state"], "failed")
        self.assertIn("budget", self.compile_state(_id)["error"])
        self.assertEqual(llm.calls, [])

    def test_duck_without_gym_fails_clearly(self):
        _id = self.queue(specs.TEMPLATE, robot="microduck")
        self.run_once(ScriptedLlm())
        self.assertIn("no microduck gym", self.compile_state(_id, "microduck")["error"])

    def test_interrupted_compiles_are_requeued(self):
        _id = self.db.design_specs.insert_one({"text": SPEC, "compile": {"roboracer": {"state": "running"}}}).inserted_id
        spec_compiler.requeue_interrupted(self.db)
        self.assertEqual(self.compile_state(_id)["state"], "queued")


if __name__ == "__main__":
    unittest.main()
