"""Reward designs for the microduck: a delta on the task's default reward terms.

Delegates compilation, linting, and runtime env patching to
``mjlab_core.reward_design``.
"""
import pathlib
import sys

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from mjlab_core import reward_design as _core  # noqa: E402
from mjlab_core.reward_design import (  # noqa: E402,F401
    Design, EnvView, RewardDesignError, SPEC_KEYS, SandboxedTerm, TERM_PREFIX,
    compile_design, lint, plan_changes,
)

ROBOT = "microduck"
LIBRARY_MODULES = (
    "mjlab.envs.mdp.rewards",
    "mjlab.tasks.velocity.mdp.rewards",
    "mjlab_microduck.tasks.mdp",
)

PASSTHROUGH_DESIGN_ID = "microduck-passthrough"
PASSTHROUGH_DESIGN_NAME = "MicroDuck task default (passthrough)"
PASSTHROUGH_DESIGN_DESCRIPTION = (
    "No-op duck reward design: every task trains on its default reward "
    "terms. A baseline for validating designs and a template for new ones."
)
PASSTHROUGH_DESIGN_CODE = '''\
"""No-op duck reward design: every task trains on its default reward terms.

Start new designs from here. TERMS maps a reward term name to
  None                              drop the task's default term
  {"weight": w, "params": {...}}    reweight / re-parameterise a default term
  {"func": "<library fn>", "weight": w, "params": {...}}   add a library term
  {"weight": w, "params": {...}}    add term_<name> defined below
A curriculum that ramps a term's weight or params is removed when this design
drops the term or sets what the curriculum ramps; the job log lists them.

Custom terms receive a read-only env view and return Tensor[num_envs]:

    def term_lean_forward(env, target):
        pitch = -env.robot.projected_gravity_b[:, 0]
        return torch.exp(-((pitch - target) ** 2) / 0.01)

env offers: num_envs, device, step_dt, episode_length_buf, robot (joint_pos,
joint_vel, root_link_lin_vel_b, root_link_ang_vel_b, projected_gravity_b, ...),
action, prev_action, command(name), sensor(name), joint_ids(regex),
body_ids(regex).
"""

TERMS = {}
'''
PASSTHROUGH_SEED_VERSION = 2


def seed(db, functions, now):
    return _core.seed(
        db, ROBOT, functions,
        PASSTHROUGH_DESIGN_ID, PASSTHROUGH_DESIGN_NAME,
        PASSTHROUGH_DESIGN_DESCRIPTION, PASSTHROUGH_DESIGN_CODE,
        PASSTHROUGH_SEED_VERSION, now,
    )


def function_catalog():
    return _core.function_catalog(LIBRARY_MODULES)


def apply(design, env_cfg, catalog=None, term_cfg_cls=None):
    return _core.apply(
        design, env_cfg, catalog=catalog,
        library_modules=LIBRARY_MODULES, term_cfg_cls=term_cfg_cls,
    )
