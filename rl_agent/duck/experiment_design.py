"""Experiment designs for the MicroDuck: PPO, network, and environment overrides.

Delegates validation, schema conversion, and cfg patching to
``mjlab_core.experiment_design``.
"""
import pathlib
import sys

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from mjlab_core import experiment_design as _core  # noqa: E402
from mjlab_core.experiment_design import (  # noqa: E402,F401
    ExperimentDesignError, apply_env, field as _field,
)

ROBOT = "microduck"
REGISTRY_ID = f"experiment_design:{ROBOT}"
DEFAULT_DESIGN_ID = "microduck-experiment-default"
ACTIVATIONS = ["elu", "relu", "selu", "tanh", "gelu", "silu", "mish"]
NETWORK_FIELDS = ("actor_hidden_dims", "critic_hidden_dims", "activation")

SCHEMA = {
    "_section_ppo": "PPO",
    "learning_rate": _field("float", "agent", "Adam learning rate. With the adaptive schedule this is "
                            "only the starting value. Walk task: 1e-3.", min=1e-6, max=1.0,
                            path="algorithm.learning_rate"),
    "lr_schedule": _field("enum", "agent", "adaptive raises or lowers the learning rate each update "
                          "to keep the policy change near desired_kl; fixed keeps learning_rate.",
                          choices=["adaptive", "fixed"], path="algorithm.schedule"),
    "desired_kl": _field("float", "agent", "Target KL divergence per update for the adaptive "
                         "schedule. Walk task: 0.01.", min=1e-4, max=1.0, path="algorithm.desired_kl"),
    "entropy_coef": _field("float", "agent", "Bonus for keeping the action distribution wide; higher "
                           "explores longer. Walk task: 0.01.", min=0.0, max=1.0,
                           path="algorithm.entropy_coef"),
    "gamma": _field("float", "agent", "Discount factor. 0.99 at 50 Hz looks ~2 s ahead.",
                    min=0.0, max=1.0, path="algorithm.gamma"),
    "lam": _field("float", "agent", "GAE lambda: bias/variance trade-off of the advantage estimate. "
                  "Walk task: 0.95.", min=0.0, max=1.0, path="algorithm.lam"),
    "clip_param": _field("float", "agent", "PPO ratio clip. Walk task: 0.2.", min=0.01, max=1.0,
                         path="algorithm.clip_param"),
    "num_learning_epochs": _field("int", "agent", "Passes over each batch of experience. Walk task: 5.",
                                  min=1, max=50, path="algorithm.num_learning_epochs"),
    "num_mini_batches": _field("int", "agent", "Mini-batches each pass is split into. Walk task: 4.",
                               min=1, max=256, path="algorithm.num_mini_batches"),
    "value_loss_coef": _field("float", "agent", "Weight of the critic loss. Walk task: 1.0.",
                              min=0.0, max=10.0, path="algorithm.value_loss_coef"),
    "max_grad_norm": _field("float", "agent", "Gradient clipping norm. Walk task: 1.0.",
                            min=0.01, max=100.0, path="algorithm.max_grad_norm"),
    "init_noise_std": _field("float", "agent", "Starting standard deviation of the action noise "
                             "(radians of joint target). Walk task: 1.0.", min=0.01, max=5.0,
                             path="actor.distribution_cfg.init_std"),

    "_section_rollout": "Rollout and budget",
    "num_envs": _field("int", "job", "Robots simulated in parallel. Overridden by the job's own value. "
                       "Default 2048.", min=1, max=16384),
    "num_steps_per_env": _field("int", "agent", "Steps each robot collects per iteration; batch size "
                                "is num_envs x this. Walk task: 24.", min=4, max=256,
                                path="num_steps_per_env"),
    "max_iterations": _field("int", "job", "PPO iterations. Overridden by the job's own value. "
                             "Default 1000.", min=1, max=1_000_000),
    "save_interval": _field("int", "job", "Iterations between checkpoints. Overridden by the job's own "
                            "value. Default 50.", min=1, max=100_000),
    "seed": _field("int", "job", "PPO seed. Overridden by the job's own value. Task default 42.",
                   min=0, max=2**31 - 1),

    "_section_network": "Network (changes checkpoint shapes)",
    "actor_hidden_dims": _field("json", "agent", "Actor MLP layer widths, e.g. [256, 128]. Walk task: "
                                "[512, 256, 128]. A checkpoint can only warm-start a run with the "
                                "same network.", path="actor.hidden_dims"),
    "critic_hidden_dims": _field("json", "agent", "Critic MLP layer widths. Walk task: "
                                 "[512, 256, 128].", path="critic.hidden_dims"),
    "activation": _field("enum", "agent", "Activation for both MLPs. Walk task: elu.",
                         choices=ACTIVATIONS, path="actor.activation"),

    "_section_env": "Environment",
    "episode_length_s": _field("float", "env", "Seconds before an episode times out. Walk task: 20.",
                               min=1.0, max=600.0),
    "pushes": _field("bool", "env", "Random shoves during episodes (push_robot). Off trains on calm "
                     "ground only."),
    "push_interval_s": _field("json", "env", "[min, max] seconds between shoves. Walk task: [3, 6]."),
    "push_velocity": _field("float", "env", "Largest shove, in m/s added to the body along x and y. "
                            "Walk task: 0.3.", min=0.0, max=3.0),
    "domain_randomization": _field("bool", "env", "Off removes every startup/reset randomisation "
                                   "(friction, mass, centre of mass, armature, encoder bias) so all "
                                   "robots are the nominal duck. Usually hurts sim-to-real."),
    "curricula": _field("bool", "env", "Off removes the task's curricula; every scheduled value stays "
                        "at its first stage for the whole run."),
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


def check_warm_start(fields, source_fields):
    return _core.check_warm_start(fields, source_fields, NETWORK_FIELDS)


def seed(db, schema_updated_at):
    return _core.seed(
        db, REGISTRY_ID, ROBOT, SCHEMA, DEFAULT_DESIGN_ID,
        "MicroDuck task defaults",
        "Sets nothing: every value comes from the mjlab task. Clone it to vary "
        "PPO, the network or the environment.",
        "duck-worker", schema_updated_at,
    )
