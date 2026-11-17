"""Shared TERMS-based reward design compiler, linter, and runtime for mjlab tasks.

Robot adapters (``duck/reward_design.py``, future ``openbeni/reward_design.py``)
supply their ``robot`` id, ``LIBRARY_MODULES``, and passthrough seed text, and
delegate compilation, validation, and in-place ``env_cfg.rewards`` patching here.
"""
import dataclasses
import importlib
import inspect
import math
import pathlib
import sys
import traceback

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)
from reward_designs import REWARD_DESIGN_NAMESPACE_BUILTINS, RewardDesignError  # noqa: E402

TERM_PREFIX = "term_"
SPEC_KEYS = {"weight", "params", "func"}


def seed(db, robot, functions, passthrough_id, passthrough_name,
         passthrough_description, passthrough_code, seed_version, now):
    """Store a robot's reward library and passthrough design; idempotent."""
    db.reward_term_catalog.update_one(
        {"_id": robot},
        {"$set": {"robot": robot, "functions": functions, "updated_at": now}},
        upsert=True)
    existing = db.reward_designs.find_one({"_id": passthrough_id}, {"version": 1}) or {}
    if existing.get("version", 0) >= seed_version:
        return
    db.reward_designs.update_one(
        {"_id": passthrough_id},
        {"$set": {
            "robot": robot,
            "name": passthrough_name,
            "description": passthrough_description,
            "code": passthrough_code,
            "version": seed_version,
            "author": "system",
            "archived": False,
            "updated_at": now,
        }, "$setOnInsert": {"created_at": now}},
        upsert=True)


@dataclasses.dataclass
class Design:
    name: str
    terms: dict
    funcs: dict


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_value(where, value):
    if value is None or isinstance(value, (bool, str)) or _is_number(value):
        if _is_number(value) and not math.isfinite(value):
            raise RewardDesignError(f"{where}: {value!r} is not finite")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_value(f"{where}[{i}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RewardDesignError(f"{where}: keys must be strings, got {key!r}")
            _check_value(f"{where}.{key}", item)
        return
    raise RewardDesignError(
        f"{where}: {type(value).__name__} values are not allowed; use numbers, "
        "strings, booleans, None, lists and dicts")


def _check_spec(name, spec, funcs):
    if spec is None:
        return
    if not isinstance(spec, dict):
        raise RewardDesignError(
            f"TERMS[{name!r}] must be None (drop the term) or a dict, got {type(spec).__name__}")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise RewardDesignError(
            f"TERMS[{name!r}]: unknown keys {sorted(unknown)}; allowed: {sorted(SPEC_KEYS)}")
    if "weight" in spec and not (_is_number(spec["weight"]) and math.isfinite(spec["weight"])):
        raise RewardDesignError(f"TERMS[{name!r}].weight must be a finite number, got {spec['weight']!r}")
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise RewardDesignError(f"TERMS[{name!r}].params must be a dict")
    _check_value(f"TERMS[{name!r}].params", params)
    func = spec.get("func")
    if func is not None:
        if not isinstance(func, str) or not func:
            raise RewardDesignError(f"TERMS[{name!r}].func must be a function name string")
        if func.startswith(TERM_PREFIX) and func not in funcs:
            raise RewardDesignError(f"TERMS[{name!r}].func: the design defines no {func}()")


def compile_design(name, code):
    """Compile design source into a ``Design``; raises ``RewardDesignError``."""
    namespace = {"__builtins__": REWARD_DESIGN_NAMESPACE_BUILTINS, "math": math}
    try:
        import torch
        namespace["torch"] = torch
    except ImportError:  # linting on a host without torch
        pass
    try:
        compiled = compile(code, f"<reward_design:{name}>", "exec")
    except SyntaxError as e:
        raise RewardDesignError(f"Reward design {name!r} has a syntax error: {e}") from e
    try:
        exec(compiled, namespace)  # noqa: S102 - restricted namespace, see module docstring
    except Exception as e:
        raise RewardDesignError(
            f"Reward design {name!r} raised during module load: {type(e).__name__}: {e}") from e

    terms = namespace.get("TERMS")
    if not isinstance(terms, dict):
        raise RewardDesignError(
            f"Reward design {name!r} must define TERMS as a dict "
            "(TERMS = {} trains on the task's defaults)")
    funcs = {k: v for k, v in namespace.items() if k.startswith(TERM_PREFIX) and callable(v)}
    for key, spec in terms.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise RewardDesignError(f"TERMS key {key!r} must be an identifier-like string")
        _check_spec(key, spec, funcs)
    referenced = {(s or {}).get("func") or TERM_PREFIX + k for k, s in terms.items()}
    unused = sorted(set(funcs) - referenced)
    if unused:
        raise RewardDesignError(
            f"Reward design {name!r} defines {', '.join(unused)} but TERMS never uses "
            f"{'it' if len(unused) == 1 else 'them'}; add e.g. "
            f"TERMS[{unused[0][len(TERM_PREFIX):]!r}] = {{'weight': 1.0}}")
    return Design(name=name, terms=terms, funcs=funcs)


def lint(code, default_terms=None, library=None):
    """Check a design without torch or mjlab. Never raises."""
    try:
        design = compile_design("<lint>", code)
        if default_terms is not None:
            plan_changes(design, set(default_terms), set(library) if library is not None else None)
    except RewardDesignError as e:
        return {"ok": False, "error": str(e), "terms": [], "funcs": []}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "terms": [], "funcs": []}
    return {"ok": True, "error": None, "terms": sorted(design.terms), "funcs": sorted(design.funcs)}


def plan_changes(design, default_terms, library=None):
    """Classify each TERMS entry as drop / override / add; raises on bad references."""
    plan = {"drop": [], "override": [], "add": []}
    for name, spec in design.terms.items():
        if spec is None:
            if name not in default_terms:
                raise RewardDesignError(
                    f"TERMS[{name!r}] = None drops a default term, but the task has no "
                    f"{name!r}. Defaults: {', '.join(sorted(default_terms))}")
            plan["drop"].append(name)
        elif name in default_terms:
            if "func" in spec:
                raise RewardDesignError(
                    f"TERMS[{name!r}]: a default term's function can't be replaced; drop it "
                    "(None) and add your version under a new name")
            plan["override"].append(name)
        else:
            if "weight" not in spec:
                raise RewardDesignError(f"TERMS[{name!r}] adds a term and needs a weight")
            func = spec.get("func") or TERM_PREFIX + name
            if func.startswith(TERM_PREFIX):
                if func not in design.funcs:
                    raise RewardDesignError(
                        f"TERMS[{name!r}] is not a default term of this task, so it needs "
                        f"\"func\" (a library reward) or a {func}() in the design. "
                        f"Defaults: {', '.join(sorted(default_terms))}")
            elif library is not None and func not in library:
                raise RewardDesignError(f"TERMS[{name!r}].func: no library reward named {func!r}")
            plan["add"].append(name)
    return plan


def function_catalog(library_modules):
    """Library reward functions by name, from ``library_modules``."""
    catalog = {}
    for module_name in library_modules:
        module = importlib.import_module(module_name)
        for attr, obj in vars(module).items():
            if attr.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", None) != module_name:
                continue
            catalog[attr] = obj
    return catalog


def _signature(func):
    target = func.__call__ if inspect.isclass(func) else func
    try:
        return inspect.signature(target)
    except (TypeError, ValueError):
        return None


def _accepted_params(func):
    sig = _signature(func)
    if sig is None:
        return None
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name: p for p in params[1:]}


def _convert(name, func, current, new):
    accepted = _accepted_params(func)
    out = dict(current)
    if accepted is not None:
        unknown = set(new) - set(accepted)
        if unknown:
            raise RewardDesignError(
                f"TERMS[{name!r}].params: {getattr(func, '__name__', func)} takes no "
                f"{', '.join(sorted(unknown))}; it takes {', '.join(accepted) or 'nothing'}")
        for key, param in accepted.items():
            default = param.default
            if key not in out and dataclasses.is_dataclass(default) and not isinstance(default, type):
                out[key] = default
        if not inspect.isclass(func):
            missing = [k for k, p in accepted.items() if p.default is p.empty
                       and p.kind is not p.VAR_POSITIONAL and k not in out and k not in new]
            if missing:
                raise RewardDesignError(
                    f"TERMS[{name!r}].params is missing {', '.join(missing)} "
                    f"(required by {getattr(func, '__name__', func)})")
    for key, value in new.items():
        base = out.get(key)
        if isinstance(value, dict) and dataclasses.is_dataclass(base) and not isinstance(base, type):
            try:
                out[key] = dataclasses.replace(base, **value)
            except TypeError as e:
                raise RewardDesignError(f"TERMS[{name!r}].params.{key}: {e}") from e
        else:
            out[key] = value
    return out


def _scheduled(curriculum_term):
    params = getattr(curriculum_term, "params", None) or {}
    reward = params.get("reward_name")
    if not isinstance(reward, str):
        return None, set()
    fields = set()
    if "weight_stages" in params:
        fields.add("weight")
    if "std_stages" in params:
        fields.add("std")
    for stage in params.get("stages") or []:
        if isinstance(stage, dict):
            if "weight" in stage:
                fields.add("weight")
            fields.update((stage.get("params") or {}).keys())
    return reward, fields


def _unschedule(env_cfg, design, plan):
    curriculum = getattr(env_cfg, "curriculum", None)
    if not isinstance(curriculum, dict):
        return []
    removed = []
    for cname, cterm in list(curriculum.items()):
        reward, fields = _scheduled(cterm)
        if reward in plan["drop"]:
            hit = True
        elif reward in plan["override"]:
            spec = design.terms[reward]
            edited = set(spec.get("params", {})) | ({"weight"} if "weight" in spec else set())
            hit = bool(fields & edited)
        else:
            hit = False
        if hit:
            del curriculum[cname]
            removed.append(cname)
    return removed


def apply(design, env_cfg, catalog=None, library_modules=(), term_cfg_cls=None):
    """Edit ``env_cfg.rewards`` in place; returns what changed by kind."""
    rewards = env_cfg.rewards
    if not isinstance(rewards, dict):
        raise RewardDesignError(f"env_cfg.rewards is a {type(rewards).__name__}, expected a dict")
    if term_cfg_cls is None:
        from mjlab.managers.reward_manager import RewardTermCfg as term_cfg_cls
    catalog = function_catalog(library_modules) if catalog is None else catalog
    defaults = {k for k, v in rewards.items() if v is not None}
    plan = plan_changes(design, defaults, set(catalog))

    for name in plan["drop"]:
        del rewards[name]
    for name in plan["override"]:
        spec, term = design.terms[name], rewards[name]
        if "params" in spec:
            term.params = _convert(name, term.func, term.params or {}, spec["params"])
        if "weight" in spec:
            term.weight = float(spec["weight"])
    for name in plan["add"]:
        spec = design.terms[name]
        func_name = spec.get("func") or TERM_PREFIX + name
        params = spec.get("params", {})
        if func_name.startswith(TERM_PREFIX):
            user_fn = design.funcs[func_name]
            _convert(name, user_fn, {}, params)
            func = SandboxedTerm(design.name, name, user_fn)
        else:
            func = catalog[func_name]
            params = _convert(name, func, {}, params)
        rewards[name] = term_cfg_cls(func=func, weight=float(spec["weight"]), params=params)
    plan["removed_curricula"] = _unschedule(env_cfg, design, plan)
    return plan


class _ReadOnly:
    """Attribute view that hides private names and returns tensor copies."""

    def __init__(self, label, target):
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name):
        label = object.__getattribute__(self, "_label")
        if name.startswith("_"):
            raise AttributeError(f"{label}.{name}: private attributes are not exposed")
        value = getattr(object.__getattribute__(self, "_target"), name)
        if callable(value):
            raise AttributeError(f"{label}.{name}: methods are not exposed to reward designs")
        clone = getattr(value, "clone", None)
        return clone() if callable(clone) else value

    def __setattr__(self, name, value):
        raise AttributeError(f"{object.__getattribute__(self, '_label')}.{name}: read-only")

    def __dir__(self):
        target = object.__getattribute__(self, "_target")
        return [n for n in dir(target) if not n.startswith("_")]


class EnvView:
    """Read-only view of an mjlab env for custom ``term_*`` reward functions."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        robot = env.scene["robot"]
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "robot", _ReadOnly("env.robot", robot.data))

    num_envs = property(lambda self: self._env.num_envs)
    device = property(lambda self: self._env.device)
    step_dt = property(lambda self: self._env.step_dt)
    episode_length_buf = property(lambda self: self._env.episode_length_buf.clone())
    action = property(lambda self: self._env.action_manager.action.clone())
    prev_action = property(lambda self: self._env.action_manager.prev_action.clone())
    joint_names = property(lambda self: tuple(self._robot.joint_names))
    body_names = property(lambda self: tuple(self._robot.body_names))

    def command(self, name):
        return self._env.command_manager.get_command(name).clone()

    def sensor(self, name):
        return _ReadOnly(f"env.sensor({name!r})", self._env.scene[name].data)

    def joint_ids(self, pattern):
        return list(self._robot.find_joints(pattern)[0])

    def body_ids(self, pattern):
        return list(self._robot.find_bodies(pattern)[0])

    def __setattr__(self, name, value):
        raise AttributeError(f"env.{name}: read-only")


class SandboxedTerm:
    """Reward-manager callable around a design's ``term_<name>`` function."""

    def __init__(self, design_name, term_name, fn):
        self.design_name = design_name
        self.term_name = term_name
        self.fn = fn
        self.calls = 0
        self.failures = 0
        self._view = None
        self.__name__ = fn.__name__

    def __deepcopy__(self, memo):
        return self

    def __call__(self, env, **params):
        import torch
        self.calls += 1
        try:
            if self._view is None or object.__getattribute__(self._view, "_env") is not env:
                self._view = EnvView(env)
            out = self.fn(self._view, **params)
            if not torch.is_tensor(out):
                raise TypeError(f"returned {type(out).__name__}, expected a torch.Tensor")
            if tuple(out.shape) != (env.num_envs,):
                raise ValueError(f"returned shape {tuple(out.shape)}, expected ({env.num_envs},)")
            return torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            if self.calls == 1:
                raise RewardDesignError(
                    f"Reward design {self.design_name!r}: {TERM_PREFIX}{self.term_name} failed "
                    f"on its first call: {type(exc).__name__}: {exc}") from exc
            self.failures += 1
            if math.log10(self.failures).is_integer():
                tb = "\n".join(traceback.format_exc().splitlines()[-3:])
                print(f"[reward_design] {TERM_PREFIX}{self.term_name} failed "
                      f"({self.failures} times so far, call {self.calls}); contributing 0. "
                      f"Last frames:\n{tb}", flush=True)
            return torch.zeros(env.num_envs, device=env.device)
