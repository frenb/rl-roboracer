"""CPU-only gym sync helpers for OpenBeni."""
import unittest

from openbeni import gyms


class GymsTest(unittest.TestCase):
    def test_parse_and_doc(self):
        self.assertEqual(gyms.parse_task_id("Beni-Velocity-Flat")["skill"], "Velocity")
        self.assertEqual(gyms.parse_task_id("Beni-Jump-Flat")["skill"], "Jump")
        self.assertIsNone(gyms.parse_task_id("Mjlab-Velocity-Flat-MicroDuck"))
        doc = gyms.gym_doc({
            "task_id": "Beni-Velocity-Flat",
            "episode_s": 20.0,
            "reward_terms": ["upright"],
            "reward_defaults": [{"name": "upright", "func": "upright", "weight": 1.0, "params": {}}],
        })
        self.assertEqual(doc["robot"], "openbeni")
        self.assertEqual(doc["name"], "OpenBeni · Locomotion + recovery · flat")
        self.assertEqual(gyms.gym_id("Beni-Velocity-Flat"), "openbeni:Beni-Velocity-Flat")


if __name__ == "__main__":
    unittest.main()
