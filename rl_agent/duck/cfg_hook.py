"""Env-cfg hook the duck worker hands to microduck_rl's trainer.

The worker sets ``MICRODUCK_CFG_HOOK=<this file>:apply`` and
``MICRODUCK_REWARD_DESIGN=<json>`` on TRAIN jobs that pick a reward design;
the fork then passes every env cfg it loads through ``apply``.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reward_design  # noqa: E402

DESIGN_ENV = "MICRODUCK_REWARD_DESIGN"
_compiled = None


def _design():
    global _compiled
    if _compiled is None:
        doc = json.loads(pathlib.Path(os.environ[DESIGN_ENV]).read_text())
        name = doc.get("name") or str(doc.get("_id"))
        _compiled = reward_design.compile_design(name, doc["code"]), doc
    return _compiled


def apply(task_id, env_cfg, play=False):
    if not os.environ.get(DESIGN_ENV):
        return
    design, doc = _design()
    plan = reward_design.apply(design, env_cfg)
    if play:
        return
    summary = ", ".join(f"{kind} {sorted(names)}" for kind, names in plan.items() if names)
    print(f"[reward_design] {task_id}: applied {design.name!r} v{doc.get('version')}: "
          f"{summary or 'no changes'}", flush=True)
    for name, term in env_cfg.rewards.items():
        if term is not None:
            print(f"[reward_design]   {name:<24} weight {term.weight:g}", flush=True)
