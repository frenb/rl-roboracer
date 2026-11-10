"""Experiment designs for the MicroDuck: how a TRAIN job learns, not what it is rewarded for.

A design is a document in ``experiment_designs`` with ``robot: "microduck"`` and any of the
fields in ``SCHEMA`` set at top level. An unset field keeps the task's own value, so the
canonical "MicroDuck task defaults" design sets nothing. Fields land in one of four places:

    job     num_envs, max_iterations, save_interval, seed - the job's own field wins if set
    agent   PPO and network settings, applied to the rsl_rl cfg by ``cfg_hook.apply_agent``
    env     episode length, pushes, domain randomisation, curricula, by ``cfg_hook.apply``

Network fields change the checkpoint's tensor shapes, so the worker applies a model's design
to every later job that loads it (EXPORT, EVAL, VIDEO, the live viewer), and refuses to
warm-start across different networks.

Pure Python apart from the cfg objects handed in, so it unit-tests without mjlab.
"""
import copy
import re

ROBOT = "microduck"
REGISTRY_ID = f"experiment_design:{ROBOT}"
DEFAULT_DESIGN_ID = "microduck-experiment-default"
ACTIVATIONS = ["elu", "relu", "selu", "tanh", "gelu", "silu", "mish"]
# Reset/startup events that put the robot in its initial state rather than perturb it.
_NOT_RANDOMIZATION = {"reset_base", "reset_robot_joints", "reset_action_history",
                      "expand_bam_friction_fields"}


def _field(type_, target, doc, *, min=None, max=None, choices=None, path=None):
    return {"type": type_, "target": target, "doc": doc, "min": min, "max": max,
            "choices": choices, "path": path}


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

NETWORK_FIELDS = ("actor_hidden_dims", "critic_hidden_dims", "activation")


class ExperimentDesignError(ValueError):
    """A design document that cannot be applied."""


def field_keys():
    return [k for k in SCHEMA if not k.startswith("_")]


def schema_for_endpoint():
    """The schema as the dashboard renders it: ordered section and field entries."""
    out = []
    for key, spec in SCHEMA.items():
        if key.startswith("_section_"):
            out.append({"kind": "section", "label": spec})
            continue
        out.append({"kind": "field", "name": key, "type": spec["type"], "default": None,
                    "min": spec["min"], "max": spec["max"], "choices": spec["choices"],
                    "doc": spec["doc"], "paper_ref": None, "kwarg": spec["target"]})
    return out


def _number(name, spec, raw):
    if isinstance(raw, bool):
        raise ExperimentDesignError(f"{name} must be a number, got {raw!r}")
    if spec["type"] == "int":
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        if not isinstance(raw, int):
            raise ExperimentDesignError(f"{name} must be a whole number, got {raw!r}")
    elif not isinstance(raw, (int, float)):
        raise ExperimentDesignError(f"{name} must be a number, got {raw!r}")
    else:
        raw = float(raw)
    if spec["min"] is not None and raw < spec["min"] or spec["max"] is not None and raw > spec["max"]:
        raise ExperimentDesignError(f"{name}={raw} is outside [{spec['min']}, {spec['max']}]")
    return raw


def _dims(name, raw):
    if (not isinstance(raw, list) or not raw or len(raw) > 8
            or not all(isinstance(w, int) and not isinstance(w, bool) and 1 <= w <= 4096 for w in raw)):
        raise ExperimentDesignError(f"{name} must be a list of 1-8 layer widths in [1, 4096], got {raw!r}")
    return list(raw)


def validate(doc):
    """``{field: value}`` for every field ``doc`` sets, type-checked; raises on anything wrong.

    Strict where the car's designs clamp: a duck run takes minutes to start, so a bad value
    should fail the job immediately rather than train something nobody asked for.
    """
    fields = {}
    for name in field_keys():
        raw = (doc or {}).get(name)
        if raw is None:
            continue
        spec = SCHEMA[name]
        if spec["type"] in ("int", "float"):
            value = _number(name, spec, raw)
        elif spec["type"] == "bool":
            if not isinstance(raw, bool):
                raise ExperimentDesignError(f"{name} must be true or false, got {raw!r}")
            value = raw
        elif spec["type"] == "enum":
            if raw not in spec["choices"]:
                raise ExperimentDesignError(f"{name} must be one of {spec['choices']}, got {raw!r}")
            value = raw
        elif name.endswith("_hidden_dims"):
            value = _dims(name, raw)
        elif name == "push_interval_s":
            if (not isinstance(raw, list) or len(raw) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
                    or not 0 < raw[0] <= raw[1] <= 600):
                raise ExperimentDesignError(f"push_interval_s must be [min, max] seconds with "
                                            f"0 < min <= max <= 600, got {raw!r}")
            value = [float(v) for v in raw]
        else:
            raise ExperimentDesignError(f"{name}: unsupported type {spec['type']}")
        fields[name] = value
    if fields.get("pushes") is False and ({"push_interval_s", "push_velocity"} & set(fields)):
        raise ExperimentDesignError("pushes is off, so push_interval_s / push_velocity would do nothing")
    return fields


def job_settings(fields):
    """The design's job-level fields (num_envs, max_iterations, ...)."""
    return {k: v for k, v in fields.items() if SCHEMA[k]["target"] == "job"}


def effective_job(job):
    """``job`` with its experiment design's job-level settings filled in where the job has none,
    plus ``num_steps_per_env``. Not written back, so a restarted job re-reads its design."""
    fields = job.get("experiment_design_fields") or {}
    merged = {**job, **{k: v for k, v in job_settings(fields).items() if job.get(k) is None}}
    if "num_steps_per_env" in fields:
        merged["num_steps_per_env"] = fields["num_steps_per_env"]
    return merged


def network(fields):
    return {k: fields[k] for k in NETWORK_FIELDS if k in fields}


def _set_path(cfg, path, value):
    *parents, leaf = path.split(".")
    for part in parents:
        cfg = cfg[part] if isinstance(cfg, dict) else getattr(cfg, part)
    if isinstance(cfg, dict):
        cfg[leaf] = value
    else:
        if not hasattr(cfg, leaf):
            raise ExperimentDesignError(f"agent cfg has no {path!r}")
        setattr(cfg, leaf, value)


def apply_agent(fields, agent_cfg):
    """Apply the PPO and network fields to an rsl_rl runner cfg in place; returns what changed."""
    changed = []
    for name, value in fields.items():
        spec = SCHEMA[name]
        if spec["target"] != "agent":
            continue
        _set_path(agent_cfg, spec["path"], copy.deepcopy(value))
        if name == "activation":
            _set_path(agent_cfg, "critic.activation", value)
        changed.append(f"{name}={value}")
    return changed


def _is_randomization(name, term):
    if name in _NOT_RANDOMIZATION or getattr(term, "mode", None) == "interval":
        return False
    params = getattr(term, "params", None) or {}
    return name.startswith("randomize") or "operation" in params or re.search(r"_bias$", name) is not None


def _unschedule_events(env_cfg, names):
    """Remove curricula that drive the named events: they would raise on a deleted event and
    overwrite an edited one."""
    curriculum = getattr(env_cfg, "curriculum", None) or {}
    found = sorted(c for c, term in curriculum.items()
                   if term is not None and (getattr(term, "params", None) or {}).get("event_name") in names)
    for name in found:
        del curriculum[name]
    return found


def _drop_events(env_cfg, names):
    for name in names:
        del env_cfg.events[name]
    return _unschedule_events(env_cfg, names)


def apply_env(fields, env_cfg):
    """Apply the environment fields to an mjlab env cfg in place; returns what changed."""
    changed = []
    events = env_cfg.events if isinstance(getattr(env_cfg, "events", None), dict) else {}
    if "episode_length_s" in fields:
        env_cfg.episode_length_s = fields["episode_length_s"]
        changed.append(f"episode_length_s={fields['episode_length_s']}")
    push = events.get("push_robot")
    push_fields = {"pushes", "push_interval_s", "push_velocity"} & set(fields)
    if push_fields and push is None:
        raise ExperimentDesignError(f"{sorted(push_fields)} set, but this task has no push_robot event")
    if fields.get("pushes") is False:
        orphaned = _drop_events(env_cfg, ["push_robot"])
        changed.append("pushes off" + (f" (and curricula {', '.join(orphaned)})" if orphaned else ""))
    if "push_interval_s" in fields:
        push.interval_range_s = tuple(fields["push_interval_s"])
        changed.append(f"push_interval_s={fields['push_interval_s']}")
    if "push_velocity" in fields:
        v = fields["push_velocity"]
        push.params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
        changed.append(f"push_velocity={v}")
    if {"push_interval_s", "push_velocity"} & set(fields):
        orphaned = _unschedule_events(env_cfg, ["push_robot"])
        if orphaned:
            changed.append(f"removed push curricula {', '.join(orphaned)}")
    if fields.get("domain_randomization") is False:
        removed = sorted(n for n, t in events.items() if t is not None and _is_randomization(n, t))
        orphaned = _drop_events(env_cfg, removed)
        changed.append(f"domain randomisation off (removed {', '.join(removed) or 'nothing'}"
                       + (f"; curricula {', '.join(orphaned)}" if orphaned else "") + ")")
    if fields.get("curricula") is False:
        removed = sorted(getattr(env_cfg, "curriculum", None) or {})
        env_cfg.curriculum = {}
        changed.append(f"curricula off (removed {', '.join(removed) or 'nothing'})")
    return changed


def check_warm_start(fields, source_fields):
    """Raise unless a checkpoint trained with ``source_fields`` loads into a run with ``fields``."""
    mine, theirs = network(fields), network(source_fields or {})
    if mine != theirs:
        raise ExperimentDesignError(
            f"cannot warm-start: the checkpoint was trained with {theirs or 'the task default network'}, "
            f"this design uses {mine or 'the task default network'}")


def seed(db, schema_updated_at):
    """Publish the schema for the dashboard and make sure the canonical design exists."""
    db.schema_registry.update_one(
        {"_id": REGISTRY_ID},
        {"$set": {"fields": schema_for_endpoint(), "robot": ROBOT, "updated_at": schema_updated_at}},
        upsert=True)
    db.experiment_designs.update_one(
        {"_id": DEFAULT_DESIGN_ID},
        {"$setOnInsert": {
            "robot": ROBOT, "name": "MicroDuck task defaults", "version": 1, "archived": False,
            "description": "Sets nothing: every value comes from the mjlab task. Clone it to vary "
                           "PPO, the network or the environment.",
            "author": "duck-worker", "created_at": schema_updated_at, "updated_at": schema_updated_at,
        }},
        upsert=True)
