"""Reward designs for OpenBeni: a delta on the task's default reward terms."""
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

ROBOT = "openbeni"
LIBRARY_MODULES = (
    "mjlab.envs.mdp.rewards",
    "mjlab.tasks.velocity.mdp.rewards",
    "beni_rl.mdp",
    "beni_rl.recovery",
    "beni_rl.jump",
)

PASSTHROUGH_DESIGN_ID = "openbeni-passthrough"
PASSTHROUGH_DESIGN_NAME = "OpenBeni task default (passthrough)"
PASSTHROUGH_DESIGN_DESCRIPTION = (
    "No-op OpenBeni reward design: every task trains on its default reward "
    "terms. A baseline for validating designs and a template for new ones."
)
PASSTHROUGH_DESIGN_CODE = '''\
"""No-op OpenBeni reward design: every task trains on its default reward terms.

Start new designs from here. TERMS maps a reward term name to
  None                              drop the task's default term
  {"weight": w, "params": {...}}    reweight / re-parameterise a default term
  {"func": "<library fn>", "weight": w, "params": {...}}   add a library term
  {"weight": w, "params": {...}}    add term_<name> defined below

Custom terms receive a read-only env view and return Tensor[num_envs].
"""

TERMS = {}
'''
PASSTHROUGH_SEED_VERSION = 1


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
