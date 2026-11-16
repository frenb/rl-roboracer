"""Cfg hooks the duck worker hands to microduck_rl.

``apply`` is installed as ``MICRODUCK_CFG_HOOK`` on TRAIN jobs: every env cfg the fork loads
gets the reward design in ``MICRODUCK_REWARD_DESIGN`` and the environment fields of the
experiment design in ``MICRODUCK_EXPERIMENT_DESIGN`` (each a JSON file, each optional).

``apply_agent`` is installed as ``MICRODUCK_RL_CFG_HOOK`` on every job that trains or loads a
checkpoint from a run with an experiment design, so the runner is built with that run's
PPO settings and network.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment_design  # noqa: E402
import reward_design  # noqa: E402

DESIGN_ENV = "MICRODUCK_REWARD_DESIGN"
EXPERIMENT_ENV = "MICRODUCK_EXPERIMENT_DESIGN"
_compiled = None
_experiment = None


def _design():
    global _compiled
    if _compiled is None:
        doc = json.loads(pathlib.Path(os.environ[DESIGN_ENV]).read_text())
        name = doc.get("name") or str(doc.get("_id"))
        _compiled = reward_design.compile_design(name, doc["code"]), doc
    return _compiled


def _experiment_design():
    global _experiment
    if _experiment is None:
        doc = json.loads(pathlib.Path(os.environ[EXPERIMENT_ENV]).read_text())
        _experiment = experiment_design.validate(doc["fields"]), doc
    return _experiment


def _apply_reward_design(task_id, env_cfg, play):
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


def apply(task_id, env_cfg, play=False):
    if os.environ.get(DESIGN_ENV):
        _apply_reward_design(task_id, env_cfg, play)
    if os.environ.get(EXPERIMENT_ENV) and not play:
        fields, doc = _experiment_design()
        changed = experiment_design.apply_env(fields, env_cfg)
        print(f"[experiment_design] {task_id}: env from {doc.get('name')!r} v{doc.get('version')}: "
              f"{'; '.join(changed) or 'no changes'}", flush=True)


def apply_agent(task_id, agent_cfg):
    if not os.environ.get(EXPERIMENT_ENV):
        return
    fields, doc = _experiment_design()
    changed = experiment_design.apply_agent(fields, agent_cfg)
    print(f"[experiment_design] {task_id}: agent from {doc.get('name')!r} v{doc.get('version')}: "
          f"{', '.join(changed) or 'no changes'}", flush=True)
