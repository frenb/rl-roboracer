import json
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare  # noqa: E402
import spec as specs  # noqa: E402
import targets  # noqa: E402

CAR_CODE = '''\
def reward_standard(data, data_arr, step_costs, *, defaults, course):
    return 0.01 * data["car"]["speed"]


def reward_failure(data, data_arr, step_costs, position_history, *, defaults, course):
    return -1.0
'''
DUCK_CODE = 'TERMS = {"track_linear_velocity": {"weight": 3.0}, "action_rate_l2": {"weight": -0.2}}\n'
GYM = {"name": "Walk", "reward_defaults": [
    {"name": "track_linear_velocity", "func": "track_lin_vel", "weight": 2.0, "params": {}},
    {"name": "action_rate_l2", "func": "action_rate_l2", "weight": -0.1, "params": {}},
    {"name": "upright", "func": "flat_orientation", "weight": 1.0, "params": {}},
]}


class FormatTest(unittest.TestCase):
    def test_template_parses_with_every_robot_missing(self):
        s = specs.parse(specs.TEMPLATE)
        self.assertEqual(s.robots, ["roboracer", "microduck"])
        self.assertEqual(s.statuses(), {"roboracer": "missing", "microduck": "missing"})
        self.assertEqual(specs.warnings(s), [])

    def test_compile_then_prose_edit_makes_it_stale(self):
        text = specs.set_compiled(specs.TEMPLATE, "roboracer", CAR_CODE, model="m")
        s = specs.parse(text)
        self.assertEqual(s.status("roboracer"), "fresh")
        self.assertEqual(s.compiled["roboracer"].code, CAR_CODE)
        self.assertEqual(s.prose, specs.parse(specs.TEMPLATE).prose)
        edited = text.replace("This matters most.", "This matters a lot.")
        self.assertEqual(specs.parse(edited).status("roboracer"), "stale")

    def test_whitespace_only_prose_change_stays_fresh(self):
        text = specs.set_compiled(specs.TEMPLATE, "roboracer", CAR_CODE)
        self.assertEqual(specs.parse(text.replace("## Goal\n", "## Goal   \n")).status("roboracer"), "fresh")

    def test_duck_task_change_only_stales_the_duck(self):
        text = specs.set_compiled(specs.TEMPLATE, "roboracer", CAR_CODE)
        text = specs.set_compiled(text, "microduck", DUCK_CODE)
        moved = text.replace("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Velocity-Rough-MicroDuck", 1)
        self.assertEqual(specs.parse(moved).statuses(), {"roboracer": "fresh", "microduck": "stale"})

    def test_hand_edit_is_detected_and_repin_keeps_it(self):
        text = specs.set_compiled(specs.TEMPLATE, "roboracer", CAR_CODE)
        hand = text.replace("0.01 *", "0.02 *")
        self.assertEqual(specs.parse(hand).status("roboracer"), "edited")
        self.assertEqual(specs.edited_robots(text, hand), ["roboracer"])
        self.assertEqual(specs.edited_robots(hand, hand), [])
        prose_changed = hand.replace("This matters most.", "Most important.")
        self.assertEqual(specs.parse(prose_changed).status("roboracer"), "stale")
        repinned = specs.parse(specs.repin(prose_changed, "roboracer"))
        self.assertEqual(repinned.status("roboracer"), "edited")
        self.assertIn("0.02 *", repinned.compiled["roboracer"].code)

    def test_recompiling_replaces_the_section_in_place(self):
        text = specs.set_compiled(specs.TEMPLATE, "microduck", DUCK_CODE)
        text = specs.set_compiled(text, "roboracer", CAR_CODE)
        text = specs.set_compiled(text, "microduck", "TERMS = {}\n")
        self.assertEqual(text.count("## Compiled: microduck"), 1)
        self.assertLess(text.index("## Compiled: roboracer"), text.index("## Compiled: microduck"))
        self.assertEqual(specs.parse(text).compiled["microduck"].code, "TERMS = {}\n")

    def test_bad_documents(self):
        for text, msg in [
            ("# no frontmatter\n## Goal\nx\n", "frontmatter"),
            ("---\nname: a\nrobots: [tank]\n---\n## Goal\nx\n", "unknown robots"),
            ("---\nname: a\nrobots: [roboracer]\n---\n## Encourage\n- x\n", "Goal"),
            ("---\nname: a\nkind: experiment\nrobots: [roboracer]\n---\n## Goal\nx\n", "kind"),
            ("---\nname: a\nrobots: [roboracer]\n---\n## Goal\nx\n## Compiled: roboracer\nno code\n", "code block"),
        ]:
            with self.subTest(msg=msg), self.assertRaisesRegex(specs.SpecError, msg):
                specs.parse(text)


class GateTest(unittest.TestCase):
    def test_unusable_reason(self):
        gate = specs.unusable_reason
        self.assertIsNone(gate({"name": "plain"}, strict=True))
        self.assertIsNone(gate({"spec_id": "s", "spec_status": "fresh"}, strict=True))
        self.assertIn("prose changed", gate({"spec_id": "s", "spec_status": "stale"}, strict=False))
        self.assertIsNone(gate({"spec_id": "s", "spec_status": "edited"}, strict=False))
        self.assertIn("strict mode", gate({"spec_id": "s", "spec_status": "edited"}, strict=True))


class TargetsTest(unittest.TestCase):
    spec = specs.parse(specs.TEMPLATE)

    def rationale(self, terms_by_section):
        return [{"section": s, "bullet": b, "terms": terms_by_section.get(s, [])}
                for s, b in targets.bullets(self.spec)]

    def test_bullets(self):
        self.assertEqual([s for s, _ in targets.bullets(self.spec)],
                         ["encourage", "encourage", "discourage", "never"])

    def test_coverage_names_missing_bullets(self):
        rationale = self.rationale({})[1:]
        problems = targets.check_coverage(self.spec, rationale, [])
        self.assertEqual(len(problems), 1)
        self.assertIn("Progress toward the objective", problems[0])
        unsupported = [{"section": "encourage", "bullet": targets.bullets(self.spec)[0][1], "reason": "n/a"}]
        self.assertEqual(targets.check_coverage(self.spec, rationale, unsupported), [])

    @unittest.skipIf(targets.reward_designs.np is None, "needs numpy")
    def test_car_checks(self):
        self.assertEqual(targets.car_checks(self.spec, CAR_CODE), [])
        rewards_crash = CAR_CODE.replace("return -1.0", "return 5.0")
        problems = targets.car_checks(self.spec, rewards_crash)
        self.assertTrue(any("crashing" in p for p in problems), problems)
        broken = CAR_CODE.replace('data["car"]["speed"]', 'data["car"]["nope"]')
        self.assertIn("raised", targets.car_checks(self.spec, broken)[0])

    def test_duck_sign_checks(self):
        good = self.rationale({"encourage": ["track_linear_velocity"], "discourage": ["action_rate_l2"],
                               "never": ["action_rate_l2"]})
        self.assertEqual(targets.duck_checks(self.spec, DUCK_CODE, good, GYM), [])
        flipped = DUCK_CODE.replace("-0.2", "0.2")
        self.assertTrue(any("must be negative" in p for p in targets.duck_checks(self.spec, flipped, good, GYM)))
        dropped = 'TERMS = {"track_linear_velocity": None}\n'
        self.assertTrue(any("drops it" in p for p in targets.duck_checks(self.spec, dropped, good, GYM)))
        unknown = self.rationale({"encourage": ["nope"], "discourage": ["action_rate_l2"],
                                  "never": ["action_rate_l2"]})
        self.assertTrue(any("doesn't have" in p for p in targets.duck_checks(self.spec, DUCK_CODE, unknown, GYM)))
        episode_end = self.rationale({"encourage": ["track_linear_velocity"], "discourage": ["action_rate_l2"],
                                      "never": [targets.EPISODE_END]})
        self.assertEqual(targets.duck_checks(self.spec, DUCK_CODE, episode_end, GYM), [])

    def test_validate_lints_first(self):
        problems = targets.validate("microduck", self.spec, 'TERMS = {"nope": None}\n', gym=GYM, library=[])
        self.assertEqual(len(problems), 1)
        self.assertIn("drops a default term", problems[0])
        self.assertTrue(targets.validate("roboracer", self.spec, "x = 1\n"))

    def test_prompts_describe_each_target(self):
        car = targets.target_prompt("roboracer")
        self.assertIn("def reward_failure", car)
        self.assertIn("dist_from_traj", car)
        duck = targets.target_prompt("microduck", gym=GYM, task_id="T",
                                     library=[{"name": "joint_vel_l2", "params": ["asset_cfg"], "doc": "Penalise."}])
        self.assertIn("track_linear_velocity: func=track_lin_vel weight=2.0", duck)
        self.assertIn("joint_vel_l2(asset_cfg): Penalise.", duck)


class CompareTest(unittest.TestCase):
    @unittest.skipIf(targets.reward_designs.np is None, "needs numpy")
    def test_car(self):
        self.assertEqual(compare.compare("roboracer", CAR_CODE, CAR_CODE)["verdict"], "same")
        nudged = CAR_CODE.replace("0.01 *", "0.0105 *")
        self.assertEqual(compare.compare("roboracer", CAR_CODE, nudged)["verdict"], "close")
        harsher = CAR_CODE.replace("return -1.0", "return -10.0")
        self.assertEqual(compare.compare("roboracer", CAR_CODE, harsher)["verdict"], "differs")

    def test_duck(self):
        self.assertEqual(compare.compare("microduck", DUCK_CODE, DUCK_CODE, GYM)["verdict"], "same")
        nudged = DUCK_CODE.replace("3.0", "3.1")
        self.assertEqual(compare.compare("microduck", DUCK_CODE, nudged, GYM)["verdict"], "close")
        r = compare.compare("microduck", DUCK_CODE, 'TERMS = {"upright": None}\n', GYM)
        self.assertEqual(r["verdict"], "differs")
        self.assertTrue(any(d.startswith("upright: dropped") for d in r["term_diffs"]), r)


class CliTest(unittest.TestCase):
    def run_cli(self, req):
        out = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("cli.py"))],
                             input=json.dumps(req), capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_round_trip(self):
        t = self.run_cli({"op": "template"})["text"]
        t = self.run_cli({"op": "set_compiled", "text": t, "robot": "microduck", "code": DUCK_CODE})["text"]
        s = self.run_cli({"op": "parse", "text": t})
        self.assertTrue(s["ok"])
        self.assertEqual(s["statuses"], {"roboracer": "missing", "microduck": "fresh"})
        self.assertEqual(self.run_cli({"op": "lint", "robot": "roboracer", "code": "x = 1"})["problems"][0][:12],
                         "Reward desig")
        self.assertFalse(self.run_cli({"op": "parse", "text": "nope"})["ok"])


if __name__ == "__main__":
    unittest.main()
