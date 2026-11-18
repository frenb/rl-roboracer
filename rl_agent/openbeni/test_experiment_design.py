"""CPU-only tests for OpenBeni experiment designs and the GRU-512 deploy lock."""
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openbeni import experiment_design as ed  # noqa: E402
from openbeni import launch  # noqa: E402


def rnn_model(**overrides):
    model = NS(class_name="MLPModel", rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=2,
               hidden_dims=[512, 256, 128], activation="elu")
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def test_rnn_shape_is_not_a_design_field():
    assert not {"rnn_type", "rnn_hidden_dim", "rnn_num_layers"} & set(ed.field_keys())
    assert ed.validate({"rnn_hidden_dim": 256}) == {}


def test_deploy_network_is_forced_to_single_layer_gru_512():
    cfg = NS(actor=rnn_model(), critic=rnn_model())
    ed.enforce_deploy_network(cfg)
    for side in (cfg.actor, cfg.critic):
        assert side.class_name == "RNNModel"
        assert (side.rnn_type, side.rnn_hidden_dim, side.rnn_num_layers) == ("gru", 512, 1)
        assert side.hidden_dims == [512, 256, 128]


def test_mlp_widths_and_ppo_fields_still_apply():
    cfg = NS(num_steps_per_env=24, actor=rnn_model(distribution_cfg={"init_std": 1.0}),
             critic=rnn_model(), algorithm=NS(learning_rate=1e-3, schedule="adaptive"))
    fields = ed.validate({"learning_rate": 3e-4, "actor_hidden_dims": [256, 128],
                          "activation": "relu", "init_noise_std": 0.5})
    ed.apply_agent(fields, cfg)
    assert cfg.algorithm.learning_rate == 3e-4
    assert cfg.actor.hidden_dims == [256, 128]
    assert cfg.critic.hidden_dims == [512, 256, 128]
    assert cfg.actor.activation == "relu"
    assert cfg.actor.distribution_cfg["init_std"] == 0.5


@pytest.mark.parametrize("doc, message", [
    ({"activation": "swish"}, "one of"),
    ({"learning_rate": 2.0}, "outside"),
    ({"num_envs": 0}, "outside"),
    ({"actor_hidden_dims": []}, "layer widths"),
])
def test_bad_values_are_rejected(doc, message):
    with pytest.raises(ed.ExperimentDesignError, match=message):
        ed.validate(doc)


def test_warm_start_needs_the_same_mlp():
    ed.check_warm_start({"learning_rate": 1e-4}, {})
    with pytest.raises(ed.ExperimentDesignError, match="cannot warm-start"):
        ed.check_warm_start({"actor_hidden_dims": [64]}, {})


def test_the_jobs_own_settings_win_over_the_design():
    job = {"num_envs": 512, "experiment_design_fields": {"num_envs": 4096, "max_iterations": 3000,
                                                         "num_steps_per_env": 32}}
    merged = ed.effective_job(job)
    assert merged["num_envs"] == 512
    assert merged["max_iterations"] == 3000
    assert launch.steps_per_iteration(merged) == 32


def test_schema_lists_every_field_once():
    entries = ed.schema_for_endpoint()
    names = [e["name"] for e in entries if e["kind"] == "field"]
    assert names == ed.field_keys()
    assert next(e for e in entries if e.get("name") == "activation")["choices"] == ed.ACTIVATIONS
