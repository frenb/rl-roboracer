import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duck import reward_design  # noqa: E402
from reward_designs import RewardDesignError  # noqa: E402


@dataclasses.dataclass
class EntityCfg:
    name: str
    joint_names: tuple = ()


@dataclasses.dataclass
class TermCfg:
    func: object
    weight: float
    params: dict = dataclasses.field(default_factory=dict)


def upright(env, std, asset_cfg=EntityCfg("robot")):
    return None


def foot_slip(env, command_threshold=0.01):
    return None


def joint_vel_l2(env, asset_cfg=EntityCfg("robot")):
    return None


def needs_target(env, target):
    return None


CATALOG = {"upright": upright, "foot_slip": foot_slip, "joint_vel_l2": joint_vel_l2,
           "needs_target": needs_target}


@dataclasses.dataclass
class EnvCfg:
    rewards: dict
    curriculum: dict = dataclasses.field(default_factory=dict)


def env_cfg():
    return EnvCfg(rewards={
        "upright": TermCfg(upright, 2.0, {"std": 0.2, "asset_cfg": EntityCfg("robot")}),
        "foot_slip": TermCfg(foot_slip, -0.1, {"command_threshold": 0.01}),
        "head_pose_bias": TermCfg(foot_slip, 0.0, {}),
        "unused": None,
    }, curriculum={
        "bias_ramp": TermCfg(None, 0, {"reward_name": "head_pose_bias",
                                       "weight_stages": [{"step": 0, "weight": 0.5}]}),
        "upright_std": TermCfg(None, 0, {"reward_name": "upright",
                                         "std_stages": [{"step": 0, "std": 0.5}]}),
        "upright_mixed": TermCfg(None, 0, {"reward_name": "upright", "stages": [
            {"step": 0, "weight": 1.0}, {"step": 9, "params": {"std": 0.1}}]}),
        "terrain": TermCfg(None, 0, {}),
    })


def apply(code, cfg=None):
    cfg = cfg or env_cfg()
    plan = reward_design.apply(reward_design.compile_design("t", code), cfg,
                               catalog=CATALOG, term_cfg_cls=TermCfg)
    return cfg, plan


def test_passthrough_changes_nothing():
    cfg, plan = apply(reward_design.PASSTHROUGH_DESIGN_CODE)
    assert cfg == env_cfg()
    assert plan == {"drop": [], "override": [], "add": [], "removed_curricula": []}


def test_dropping_a_scheduled_term_removes_its_curriculum():
    cfg, plan = apply('TERMS = {"head_pose_bias": None}')
    assert plan["removed_curricula"] == ["bias_ramp"]
    assert set(cfg.curriculum) == {"upright_std", "upright_mixed", "terrain"}


def test_override_removes_only_curricula_scheduling_what_it_sets():
    _, plan = apply('TERMS = {"upright": {"weight": 3}}')
    assert plan["removed_curricula"] == ["upright_mixed"]
    _, plan = apply('TERMS = {"upright": {"params": {"std": 0.3}}}')
    assert plan["removed_curricula"] == ["upright_std", "upright_mixed"]
    _, plan = apply('TERMS = {"upright": {"params": {"asset_cfg": {"name": "robot"}}}}')
    assert plan["removed_curricula"] == []


def test_override_weight_and_params_keeps_other_params():
    cfg, plan = apply('TERMS = {"upright": {"weight": 3, "params": {"std": 0.5}}}')
    term = cfg.rewards["upright"]
    assert (term.weight, term.params["std"]) == (3.0, 0.5)
    assert term.params["asset_cfg"] == EntityCfg("robot")
    assert plan["override"] == ["upright"]


def test_dict_param_updates_a_dataclass_param():
    cfg, _ = apply('TERMS = {"upright": {"params": {"asset_cfg": {"joint_names": ["knee"]}}}}')
    assert cfg.rewards["upright"].params["asset_cfg"] == EntityCfg("robot", ["knee"])


def test_none_drops_a_default():
    cfg, plan = apply('TERMS = {"head_pose_bias": None}')
    assert "head_pose_bias" not in cfg.rewards
    assert plan["drop"] == ["head_pose_bias"]


def test_add_library_term_fills_entity_defaults():
    cfg, _ = apply('TERMS = {"joint_vel": {"func": "joint_vel_l2", "weight": -1e-4}}')
    term = cfg.rewards["joint_vel"]
    assert term.func is joint_vel_l2
    assert term.weight == -1e-4
    assert term.params == {"asset_cfg": EntityCfg("robot")}


def test_add_custom_term_wraps_it():
    code = (
        'TERMS = {"lean": {"weight": 0.5, "params": {"target": 0.1}}}\n'
        "def term_lean(env, target):\n"
        "    return target\n"
    )
    cfg, plan = apply(code)
    term = cfg.rewards["lean"]
    assert isinstance(term.func, reward_design.SandboxedTerm)
    assert term.params == {"target": 0.1}
    assert plan["add"] == ["lean"]


def test_custom_term_survives_deepcopy():
    import copy
    code = 'TERMS = {"lean": {"weight": 1}}\ndef term_lean(env):\n    return 0\n'
    cfg, _ = apply(code)
    assert copy.deepcopy(cfg).rewards["lean"].func is cfg.rewards["lean"].func


@pytest.mark.parametrize("code, message", [
    ("x = 1", "must define TERMS"),
    ("TERMS = {'upright': 3}", "must be None"),
    ("TERMS = {'upright': {'wieght': 3}}", "unknown keys"),
    ("TERMS = {'upright': {'weight': float('nan')}}", "finite"),
    ("TERMS = {'upright': {'params': {'std': {1, 2}}}}", "not allowed"),
    ("TERMS = {}\ndef term_orphan(env):\n    return 0", "never uses"),
    ("import os\nTERMS = {}", "raised during module load"),
    ("TERMS = {", "syntax error"),
])
def test_compile_rejects(code, message):
    with pytest.raises(RewardDesignError, match=message):
        reward_design.compile_design("t", code)


@pytest.mark.parametrize("code, message", [
    ("TERMS = {'nope': None}", "has no 'nope'"),
    ("TERMS = {'upright': {'func': 'joint_vel_l2'}}", "can't be replaced"),
    ("TERMS = {'extra': {'func': 'joint_vel_l2'}}", "needs a weight"),
    ("TERMS = {'extra': {'weight': 1}}", "needs \"func\""),
    ("TERMS = {'extra': {'func': 'made_up', 'weight': 1}}", "no library reward"),
    ("TERMS = {'upright': {'params': {'sigma': 1}}}", "takes no sigma"),
    ("TERMS = {'extra': {'func': 'needs_target', 'weight': 1}}", "missing target"),
])
def test_apply_rejects(code, message):
    with pytest.raises(RewardDesignError, match=message):
        apply(code)


def test_lint_never_raises_and_checks_references():
    assert reward_design.lint("TERMS = {'upright': {'weight': 1}}", ["upright"])["ok"]
    result = reward_design.lint("TERMS = {'gone': None}", ["upright"])
    assert not result["ok"] and "has no 'gone'" in result["error"]
    assert not reward_design.lint("TERMS = {")["ok"]
    assert not reward_design.lint(
        "TERMS = {'x': {'func': 'nope', 'weight': 1}}", ["upright"], ["joint_vel_l2"])["ok"]


# --- runtime: needs torch ------------------------------------------------------

class FakeRobot:
    joint_names = ("hip", "knee")
    body_names = ("base",)

    def __init__(self, torch, n):
        self.data = type("Data", (), {})()
        self.data.joint_pos = torch.zeros(n, 2)

    def find_joints(self, pattern):
        return [i for i, j in enumerate(self.joint_names) if pattern in j], []


class FakeEnv:
    def __init__(self, torch, n=4):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.robot = FakeRobot(torch, n)
        self.scene = {"robot": self.robot}
        self.episode_length_buf = torch.zeros(n)


def term(code_body, name="t"):
    namespace = {}
    exec(f"import torch\ndef term_{name}(env, **p):\n{code_body}", namespace)  # noqa: S102
    return reward_design.SandboxedTerm("design", name, namespace[f"term_{name}"])


def test_term_returns_scrubbed_float_rewards():
    torch = pytest.importorskip("torch")
    env = FakeEnv(torch)
    t = term("    return torch.tensor([1, float('nan'), float('inf'), -2])")
    assert t(env).tolist() == [1.0, 0.0, 0.0, -2.0]


def test_term_failing_first_call_fails_fast():
    torch = pytest.importorskip("torch")
    with pytest.raises(RewardDesignError, match="first call"):
        term("    return torch.zeros(3)")(FakeEnv(torch))


def test_term_failing_later_contributes_zero(capsys):
    torch = pytest.importorskip("torch")
    env = FakeEnv(torch)
    t = term("    raise ValueError('rare corner case')")
    t._view = reward_design.EnvView(env)
    t.calls = 5
    assert t(env).tolist() == [0.0] * 4
    assert "contributing 0" in capsys.readouterr().out


def test_view_is_read_only_and_copies_tensors():
    torch = pytest.importorskip("torch")
    env = FakeEnv(torch)
    view = reward_design.EnvView(env)
    joint_pos = view.robot.joint_pos
    joint_pos += 1
    assert env.robot.data.joint_pos.sum() == 0
    with pytest.raises(AttributeError):
        view.robot.joint_pos = 1
    with pytest.raises(AttributeError):
        view.num_envs = 2
    with pytest.raises(AttributeError, match="private"):
        view.robot._secret  # noqa: B018
    assert view.joint_ids("knee") == [1]
