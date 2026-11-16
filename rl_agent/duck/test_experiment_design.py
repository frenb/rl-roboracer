import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from duck import experiment_design as ed  # noqa: E402
from duck import launch  # noqa: E402


def agent_cfg():
    return NS(num_steps_per_env=24,
              actor=NS(hidden_dims=[512, 256, 128], activation="elu",
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0}),
              critic=NS(hidden_dims=[512, 256, 128], activation="elu"),
              algorithm=NS(learning_rate=1e-3, schedule="adaptive", entropy_coef=0.01, gamma=0.99))


def env_cfg():
    events = {
        "reset_base": NS(mode="reset", params={"pose_range": {}}),
        "push_robot": NS(mode="interval", interval_range_s=(3.0, 6.0),
                         params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}),
        "foot_friction": NS(mode="startup", params={"operation": "abs", "ranges": (0.7, 1.3)}),
        "encoder_bias": NS(mode="startup", params={"bias_range": (-0.01, 0.01)}),
        "randomize_com": NS(mode="reset", params={"ranges": (-0.003, 0.003)}),
        "set_ground_state": NS(mode="reset", params={"mix": 0.5}),
    }
    curriculum = {
        "com_range": NS(params={"event_name": "randomize_com"}),
        "push_magnitude": NS(params={"event_name": "push_robot"}),
        "action_rate_weight": NS(params={"reward_name": "action_rate_l2"}),
    }
    return NS(episode_length_s=20.0, events=events, curriculum=curriculum)


def test_unset_fields_keep_the_task_defaults():
    assert ed.validate({"name": "x", "learning_rate": None}) == {}
    cfg = agent_cfg()
    assert ed.apply_agent({}, cfg) == []
    assert cfg.algorithm.learning_rate == 1e-3


@pytest.mark.parametrize("doc, message", [
    ({"learning_rate": 2.0}, "outside"),
    ({"learning_rate": "fast"}, "number"),
    ({"num_envs": 12.5}, "whole number"),
    ({"num_envs": True}, "number"),
    ({"lr_schedule": "cosine"}, "one of"),
    ({"pushes": "no"}, "true or false"),
    ({"actor_hidden_dims": [256, 0]}, "layer widths"),
    ({"actor_hidden_dims": []}, "layer widths"),
    ({"push_interval_s": [5, 2]}, "min <= max"),
    ({"pushes": False, "push_velocity": 0.5}, "would do nothing"),
])
def test_bad_values_are_rejected(doc, message):
    with pytest.raises(ed.ExperimentDesignError, match=message):
        ed.validate(doc)


def test_agent_fields_reach_nested_cfgs():
    cfg = agent_cfg()
    fields = ed.validate({"learning_rate": 3e-4, "lr_schedule": "fixed", "init_noise_std": 0.5,
                          "num_steps_per_env": 48, "actor_hidden_dims": [256, 128],
                          "activation": "relu", "num_envs": 1024})
    ed.apply_agent(fields, cfg)
    assert cfg.algorithm.learning_rate == 3e-4
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.actor.distribution_cfg["init_std"] == 0.5
    assert cfg.num_steps_per_env == 48
    assert cfg.actor.hidden_dims == [256, 128]
    assert cfg.critic.hidden_dims == [512, 256, 128]
    assert cfg.actor.activation == cfg.critic.activation == "relu"


def test_turning_randomisation_off_keeps_setup_events_and_drops_their_curricula():
    cfg = env_cfg()
    changed = ed.apply_env({"domain_randomization": False}, cfg)
    assert set(cfg.events) == {"reset_base", "push_robot", "set_ground_state"}
    assert "com_range" not in cfg.curriculum
    assert "push_magnitude" in cfg.curriculum
    assert "randomize_com" in changed[0]


def test_push_edits_remove_the_push_curriculum():
    cfg = env_cfg()
    ed.apply_env({"push_interval_s": [1.0, 2.0], "push_velocity": 0.5}, cfg)
    push = cfg.events["push_robot"]
    assert push.interval_range_s == (1.0, 2.0)
    assert push.params["velocity_range"] == {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    assert "push_magnitude" not in cfg.curriculum


def test_pushes_off_and_curricula_off():
    cfg = env_cfg()
    ed.apply_env({"pushes": False, "curricula": False, "episode_length_s": 10.0}, cfg)
    assert "push_robot" not in cfg.events
    assert cfg.curriculum == {}
    assert cfg.episode_length_s == 10.0


def test_push_fields_on_a_task_without_pushes_fail():
    cfg = env_cfg()
    del cfg.events["push_robot"]
    with pytest.raises(ed.ExperimentDesignError, match="no push_robot"):
        ed.apply_env({"push_velocity": 0.5}, cfg)


def test_the_jobs_own_settings_win_over_the_design():
    job = {"num_envs": 512, "max_iterations": None,
           "experiment_design_fields": {"num_envs": 4096, "max_iterations": 3000, "num_steps_per_env": 32}}
    merged = ed.effective_job(job)
    assert merged["num_envs"] == 512
    assert merged["max_iterations"] == 3000
    assert launch.steps_per_iteration(merged) == 32
    assert "num_steps_per_env" not in job


def test_warm_start_needs_the_same_network():
    ed.check_warm_start({"learning_rate": 1e-4}, {})
    ed.check_warm_start({"actor_hidden_dims": [64]}, {"actor_hidden_dims": [64], "gamma": 0.9})
    with pytest.raises(ed.ExperimentDesignError, match="cannot warm-start"):
        ed.check_warm_start({"actor_hidden_dims": [64]}, {})


def test_design_env_splits_training_from_loading():
    train = launch.design_env("rd.json", "ed.json")
    assert train["MICRODUCK_CFG_HOOK"].endswith("cfg_hook.py:apply")
    assert train["MICRODUCK_RL_CFG_HOOK"].endswith("cfg_hook.py:apply_agent")
    load = launch.design_env(experiment_design_file="ed.json", train=False)
    assert set(load) == {"MICRODUCK_RL_CFG_HOOK", "MICRODUCK_EXPERIMENT_DESIGN"}
    assert launch.design_env() == {}


def test_schema_lists_every_field_once_with_no_hardcoded_defaults():
    entries = ed.schema_for_endpoint()
    names = [e["name"] for e in entries if e["kind"] == "field"]
    assert names == ed.field_keys()
    assert all(e["default"] is None for e in entries if e["kind"] == "field")
    assert next(e for e in entries if e.get("name") == "activation")["choices"] == ed.ACTIVATIONS
