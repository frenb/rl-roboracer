"""Experiment designs for OpenBeni: PPO + env overrides with GRU-512 locked."""
import pathlib
import sys

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from mjlab_core import experiment_design as _core  # noqa: E402
from mjlab_core.experiment_design import (  # noqa: E402,F401
    ExperimentDesignError, apply_env, field as _field,
)

ROBOT = "openbeni"
REGISTRY_ID = f"experiment_design:{ROBOT}"
DEFAULT_DESIGN_ID = "openbeni-experiment-default"
ACTIVATIONS = ["elu", "relu", "selu", "tanh", "gelu", "silu", "mish"]
# MLP widths may vary in experiments, but the RNN deploy contract is fixed.
NETWORK_FIELDS = ("actor_hidden_dims", "critic_hidden_dims", "activation")
DEPLOY_RNN = {"rnn_type": "gru", "rnn_hidden_dim": 512, "rnn_num_layers": 1}

SCHEMA = {
    "_section_ppo": "PPO",
    "learning_rate": _field("float", "agent", "Adam learning rate. Beni default: 1e-3.",
                            min=1e-6, max=1.0, path="algorithm.learning_rate"),
    "lr_schedule": _field("enum", "agent", "adaptive or fixed learning-rate schedule.",
                          choices=["adaptive", "fixed"], path="algorithm.schedule"),
    "desired_kl": _field("float", "agent", "Target KL for adaptive schedule. Beni default: 0.01.",
                         min=1e-4, max=1.0, path="algorithm.desired_kl"),
    "entropy_coef": _field("float", "agent", "Entropy bonus. Beni default: 0.01.",
                           min=0.0, max=1.0, path="algorithm.entropy_coef"),
    "gamma": _field("float", "agent", "Discount factor. Beni default: 0.99.",
                    min=0.0, max=1.0, path="algorithm.gamma"),
    "lam": _field("float", "agent", "GAE lambda. Beni default: 0.95.",
                  min=0.0, max=1.0, path="algorithm.lam"),
    "clip_param": _field("float", "agent", "PPO ratio clip. Beni default: 0.2.",
                         min=0.01, max=1.0, path="algorithm.clip_param"),
    "num_learning_epochs": _field("int", "agent", "Passes per batch. Beni default: 5.",
                                  min=1, max=50, path="algorithm.num_learning_epochs"),
    "num_mini_batches": _field("int", "agent", "Mini-batches per pass. Beni default: 4.",
                               min=1, max=256, path="algorithm.num_mini_batches"),
    "value_loss_coef": _field("float", "agent", "Critic loss weight. Beni default: 1.0.",
                              min=0.0, max=10.0, path="algorithm.value_loss_coef"),
    "max_grad_norm": _field("float", "agent", "Gradient clip. Beni default: 1.0.",
                            min=0.01, max=100.0, path="algorithm.max_grad_norm"),
    "init_noise_std": _field("float", "agent", "Initial action noise std. Beni default: 1.0.",
                             min=0.01, max=5.0, path="actor.distribution_cfg.init_std"),

    "_section_rollout": "Rollout and budget",
    "num_envs": _field("int", "job", "Parallel envs. Overridden by the job. Default 4096.",
                       min=1, max=16384),
    "num_steps_per_env": _field("int", "agent", "Steps per env per iteration. Beni default: 24.",
                                min=4, max=256, path="num_steps_per_env"),
    "max_iterations": _field("int", "job", "PPO iterations. Overridden by the job. Default 1000.",
                             min=1, max=1_000_000),
    "save_interval": _field("int", "job", "Checkpoint interval. Overridden by the job. Default 100.",
                            min=1, max=100_000),
    "seed": _field("int", "job", "PPO seed. Overridden by the job. Task default 42.",
                   min=0, max=2**31 - 1),

    "_section_network": "Network (MLP widths; GRU-512 is locked for deploy)",
    "actor_hidden_dims": _field("json", "agent", "Actor MLP widths after the GRU. Beni default: "
                                "[512, 256, 128].", path="actor.hidden_dims"),
    "critic_hidden_dims": _field("json", "agent", "Critic MLP widths. Beni default: [512, 256, 128].",
                                 path="critic.hidden_dims"),
    "activation": _field("enum", "agent", "Activation for both MLPs. Beni default: elu.",
                         choices=ACTIVATIONS, path="actor.activation"),

    "_section_env": "Environment",
    "episode_length_s": _field("float", "env", "Episode timeout in seconds.",
                               min=1.0, max=600.0),
}


def field_keys():
    return _core.field_keys(SCHEMA)


def schema_for_endpoint():
    return _core.schema_for_endpoint(SCHEMA)


def validate(doc):
    return _core.validate(doc, SCHEMA)


def job_settings(fields):
    return _core.job_settings(fields, SCHEMA)


def effective_job(job):
    return _core.effective_job(job, SCHEMA)


def network(fields):
    return _core.network(fields, NETWORK_FIELDS)


def apply_agent(fields, agent_cfg):
    return _core.apply_agent(fields, agent_cfg, SCHEMA)


def enforce_deploy_network(agent_cfg):
    """Force the browser/MNN deploy contract: single-layer GRU-512 RNNModel."""
    for side in ("actor", "critic"):
        model = getattr(agent_cfg, side, None)
        if model is None:
            continue
        for key, value in DEPLOY_RNN.items():
            if hasattr(model, key):
                setattr(model, key, value)
        if hasattr(model, "class_name"):
            model.class_name = "RNNModel"


def check_warm_start(fields, source_fields):
    return _core.check_warm_start(fields, source_fields, NETWORK_FIELDS)


def seed(db, schema_updated_at):
    return _core.seed(
        db, REGISTRY_ID, ROBOT, SCHEMA, DEFAULT_DESIGN_ID,
        "OpenBeni task defaults",
        "Sets nothing: every value comes from the open-beni task. Clone it to vary "
        "PPO or the environment. The GRU-512 actor is locked for ONNX/MNN deploy.",
        "beni-worker", schema_updated_at,
    )
