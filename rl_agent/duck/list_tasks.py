"""Print the registered microduck tasks and reward library as JSON.

Run by the duck worker in a subprocess (it needs mjlab, which pulls in torch
and warp) to fill the ``gyms`` and ``reward_term_catalog`` collections.
"""
import dataclasses
import inspect
import json
import math
import sys

import mjlab.tasks  # noqa: F401 - registers mjlab's and the fork's tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg

import reward_design


def jsonable(value):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
                if not f.name.startswith("_")}
    return repr(value)


def func_name(func):
    return getattr(func, "__name__", type(func).__name__)


def describe(task_id):
    cfg = load_env_cfg(task_id)
    terrain = getattr(cfg.scene, "terrain", None)
    rewards = cfg.rewards if isinstance(cfg.rewards, dict) else vars(cfg.rewards)
    terms = {name: term for name, term in rewards.items() if term is not None}
    return {
        "task_id": task_id,
        "terrain_type": getattr(terrain, "terrain_type", None),
        "episode_s": cfg.episode_length_s,
        "reward_terms": sorted(terms),
        "reward_defaults": [
            {"name": name, "func": func_name(term.func), "weight": term.weight,
             "params": jsonable(term.params or {})}
            for name, term in sorted(terms.items())],
    }


def describe_function(name, func):
    target = func.__call__ if inspect.isclass(func) else func
    params = []
    try:
        sig = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        sig = []
    if sig and sig[0].name == "self":
        sig = sig[1:]
    for p in sig[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        required = p.default is p.empty
        params.append({"name": p.name, "required": required,
                       "default": None if required else jsonable(p.default)})
    doc = inspect.getdoc(func) or ""
    return {"name": name, "module": func.__module__, "params": params,
            "doc": doc.split("\n\n")[0].replace("\n", " ")}


def main():
    tasks = [describe(t) for t in sorted(list_tasks()) if "MicroDuck" in t]
    used = {d["func"] for t in tasks for d in t["reward_defaults"]}
    functions = [
        describe_function(name, func)
        for name, func in sorted(reward_design.function_catalog().items())
        if func.__module__ != "mjlab_microduck.tasks.mdp" or name in used]
    json.dump({"tasks": tasks, "functions": functions}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
