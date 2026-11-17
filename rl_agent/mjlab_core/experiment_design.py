"""Shared experiment design validator, schema publisher, and cfg-applier for mjlab robots."""
import copy
import re

NOT_RANDOMIZATION = {"reset_base", "reset_robot_joints", "reset_action_history",
                     "expand_bam_friction_fields"}
DEFAULT_NETWORK_FIELDS = ("actor_hidden_dims", "critic_hidden_dims", "activation")


def field(type_, target, doc, *, min=None, max=None, choices=None, path=None):
    return {"type": type_, "target": target, "doc": doc, "min": min, "max": max,
            "choices": choices, "path": path}


class ExperimentDesignError(ValueError):
    """A design document that cannot be applied."""


def field_keys(schema):
    return [k for k in schema if not k.startswith("_")]


def schema_for_endpoint(schema):
    out = []
    for key, spec in schema.items():
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


def validate(doc, schema):
    """``{field: value}`` for every field ``doc`` sets, type-checked against ``schema``."""
    fields = {}
    for name in field_keys(schema):
        raw = (doc or {}).get(name)
        if raw is None:
            continue
        spec = schema[name]
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


def job_settings(fields, schema):
    return {k: v for k, v in fields.items() if schema[k]["target"] == "job"}


def effective_job(job, schema):
    fields = job.get("experiment_design_fields") or {}
    merged = {**job, **{k: v for k, v in job_settings(fields, schema).items() if job.get(k) is None}}
    if "num_steps_per_env" in fields:
        merged["num_steps_per_env"] = fields["num_steps_per_env"]
    return merged


def network(fields, network_fields=DEFAULT_NETWORK_FIELDS):
    return {k: fields[k] for k in network_fields if k in fields}


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


def apply_agent(fields, agent_cfg, schema):
    changed = []
    for name, value in fields.items():
        spec = schema[name]
        if spec["target"] != "agent":
            continue
        _set_path(agent_cfg, spec["path"], copy.deepcopy(value))
        if name == "activation":
            _set_path(agent_cfg, "critic.activation", value)
        changed.append(f"{name}={value}")
    return changed


def _is_randomization(name, term):
    if name in NOT_RANDOMIZATION or getattr(term, "mode", None) == "interval":
        return False
    params = getattr(term, "params", None) or {}
    return name.startswith("randomize") or "operation" in params or re.search(r"_bias$", name) is not None


def _unschedule_events(env_cfg, names):
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


def check_warm_start(fields, source_fields, network_fields=DEFAULT_NETWORK_FIELDS):
    mine, theirs = network(fields, network_fields), network(source_fields or {}, network_fields)
    if mine != theirs:
        raise ExperimentDesignError(
            f"cannot warm-start: the checkpoint was trained with {theirs or 'the task default network'}, "
            f"this design uses {mine or 'the task default network'}")


def seed(db, registry_id, robot, schema, default_id, default_name, default_description, author, schema_updated_at):
    db.schema_registry.update_one(
        {"_id": registry_id},
        {"$set": {"fields": schema_for_endpoint(schema), "robot": robot, "updated_at": schema_updated_at}},
        upsert=True)
    db.experiment_designs.update_one(
        {"_id": default_id},
        {"$setOnInsert": {
            "robot": robot, "name": default_name, "version": 1, "archived": False,
            "description": default_description,
            "author": author, "created_at": schema_updated_at, "updated_at": schema_updated_at,
        }},
        upsert=True)
