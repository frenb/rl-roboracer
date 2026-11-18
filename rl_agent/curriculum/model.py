"""Curriculum documents: validation, stage inheritance and carry rules.

Stdlib only, so it runs in the dashboard's bare ``python3`` as well as in the
runner service.

A curriculum is ``{robot, name, description, stages: [stage, ...]}``. A stage:

    id, name            stable id (generated if missing) and display name
    kind                "train" ("skill" arrives with the skill library)
    gym                 gym id; inherited from the previous stage when unset
    reward_design_id    inherited when the key is absent; null = task default
    experiment_design_id  same as reward_design_id
    budget              training budget in the robot's unit (budget_field)
    settings            robot stage settings, merged over the previous stage's
    init                "previous" | "none" | "model:<id>"
    carry               training state taken from init (robot defaults if unset)
    gate                {"conditions": [{metric, op, value}], "consecutive": 1} or null
    on_fail             {"retries": 0, "extra_budget": <budget>}
"""
import copy
import pathlib
import re
import sys

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from robots_registry import CARRY_ITEMS  # noqa: E402

KINDS = ("train",)
OPS = (">=", ">", "<=", "<", "==")
INIT_PREVIOUS = "previous"
INIT_NONE = "none"
INIT_MODEL = "model:"
INHERITED = ("gym", "reward_design_id", "experiment_design_id", "budget")
STAGE_KEYS = {"id", "name", "kind", "gym", "reward_design_id", "experiment_design_id", "budget",
              "settings", "init", "carry", "gate", "on_fail"}
MAX_STAGES = 32


class CurriculumError(ValueError):
    """A curriculum that can't be run; ``problems`` lists every reason."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40]


def _positive_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value if isinstance(value, int) and value > 0 else None


def _robot_of(row):
    return (row or {}).get("robot") or "roboracer"


def caps(robot):
    """The manifest's curriculum block (robots_registry fills its defaults)."""
    return (robot or {}).get("curriculum") or {}


def _label(i, stage):
    name = stage.get("name")
    return f"stage {i + 1}" + (f" ({name})" if name else "")


def _gate(raw, cap, where, problems):
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        problems.append(f"{where}: gate must be an object with conditions")
        return None
    conditions = raw.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        problems.append(f"{where}: gate needs at least one condition")
        return None
    allowed = cap.get("gate_metrics") or []
    out = []
    for c in conditions:
        if not isinstance(c, dict):
            problems.append(f"{where}: each gate condition is {{metric, op, value}}")
            continue
        metric, op, value = c.get("metric"), c.get("op"), c.get("value")
        if not isinstance(metric, str) or not metric:
            problems.append(f"{where}: gate condition without a metric")
        elif allowed and metric not in allowed:
            problems.append(f"{where}: gate metric {metric!r} isn't reported for this robot; "
                            f"use one of {allowed}")
        if op not in OPS:
            problems.append(f"{where}: gate op {op!r} must be one of {list(OPS)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{where}: gate value for {metric!r} must be a number")
        out.append({"metric": metric, "op": op, "value": value})
    consecutive = raw.get("consecutive", 1)
    if _positive_int(consecutive) is None:
        problems.append(f"{where}: gate consecutive must be a positive whole number")
        consecutive = 1
    elif consecutive > 1 and not cap.get("early_stop"):
        problems.append(f"{where}: gate consecutive > 1 needs early stop, which this robot's "
                        "trainer doesn't support yet (each stage is evaluated once after training)")
    return {"conditions": out, "consecutive": consecutive}


def _on_fail(raw, budget, where, problems):
    raw = raw or {}
    if not isinstance(raw, dict):
        problems.append(f"{where}: on_fail must be an object")
        raw = {}
    retries = raw.get("retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0 or retries > 10:
        problems.append(f"{where}: on_fail.retries must be a whole number from 0 to 10")
        retries = 0
    extra = raw.get("extra_budget", budget)
    if extra is not None and _positive_int(extra) is None:
        problems.append(f"{where}: on_fail.extra_budget must be a positive whole number")
        extra = budget
    return {"retries": retries, "extra_budget": extra}


def _check_refs(stage, robot_id, context, where, problems):
    gyms = context.get("gyms")
    if gyms is not None and stage.get("gym") is not None:
        gym = gyms.get(str(stage["gym"]))
        if gym is None:
            problems.append(f"{where}: gym {stage['gym']!r} not found")
        elif _robot_of(gym) != robot_id:
            problems.append(f"{where}: gym {stage['gym']!r} is a {_robot_of(gym)} gym")
        elif gym.get("archived"):
            problems.append(f"{where}: gym {stage['gym']!r} is archived")
    for key, coll in (("reward_design_id", "reward_designs"), ("experiment_design_id", "experiment_designs")):
        rows = context.get(coll)
        ref = stage.get(key)
        if rows is None or ref is None:
            continue
        row = rows.get(str(ref))
        kind = coll[:-1].replace("_", " ")
        if row is None:
            problems.append(f"{where}: {kind} {ref!r} not found")
        elif _robot_of(row) != robot_id:
            problems.append(f"{where}: {kind} {ref!r} is a {_robot_of(row)} design")
        elif row.get("archived"):
            problems.append(f"{where}: {kind} {ref!r} is archived")
    init = stage.get("init") or ""
    models = context.get("models")
    if models is not None and init.startswith(INIT_MODEL):
        model = models.get(init[len(INIT_MODEL):])
        if model is None:
            problems.append(f"{where}: init model {init[len(INIT_MODEL):]!r} not found")
        elif _robot_of(model) != robot_id:
            problems.append(f"{where}: init model is a {_robot_of(model)} model")


def default_carry(cap, init, same_gym, same_reward):
    if init == INIT_NONE:
        return []
    items = list(cap.get("carry") or [])
    if "replay" in items and not (init == INIT_PREVIOUS and same_gym and same_reward):
        items.remove("replay")
    return items


def _carry(raw, cap, init, same_gym, same_reward, where, problems):
    if raw is None:
        return default_carry(cap, init, same_gym, same_reward)
    if not isinstance(raw, list) or not all(isinstance(c, str) for c in raw):
        problems.append(f"{where}: carry must be a list of item names")
        return []
    items = list(dict.fromkeys(raw))
    supported = list(cap.get("carry") or [])
    for c in items:
        if c not in CARRY_ITEMS:
            problems.append(f"{where}: unknown carry item {c!r}; known: {list(CARRY_ITEMS)}")
        elif c not in supported:
            problems.append(f"{where}: this robot's trainer can't restore {c!r} "
                            f"(it can restore {supported or 'nothing yet'})")
    if items and init == INIT_NONE:
        problems.append(f"{where}: carry {items} needs init 'previous' or 'model:<id>', not 'none'")
    if items and cap.get("carry_atomic") and set(items) != set(supported):
        problems.append(f"{where}: this robot restores {supported} together; carry all of them or none")
    if "replay" in items:
        if init != INIT_PREVIOUS:
            problems.append(f"{where}: carry 'replay' needs init 'previous' (a saved model has no replay buffer)")
        elif not same_gym:
            problems.append(f"{where}: carry 'replay' needs the same gym as the previous stage "
                            "(the stored observations would not match)")
        elif not same_reward:
            problems.append(f"{where}: carry 'replay' needs the same reward design as the previous stage "
                            "(the stored rewards would be stale)")
    return [c for c in CARRY_ITEMS if c in items]


def validate(doc, robot, context=None):
    """Normalize a curriculum document for ``robot`` (its manifest).

    ``context`` holds id -> row maps for reference checks (``gyms``,
    ``reward_designs``, ``experiment_designs``, ``models``); a missing map
    skips that check. Returns the normalized document with fully resolved
    stages; raises CurriculumError listing every problem.
    """
    problems = []
    context = context or {}
    if not isinstance(doc, dict):
        raise CurriculumError(["curriculum must be an object"])
    robot_id = (robot or {}).get("id")
    cap = caps(robot)
    if not robot_id:
        raise CurriculumError(["unknown robot"])
    if not cap.get("enabled"):
        problems.append(f"curricula aren't enabled for {robot_id} yet")
    name = str(doc.get("name") or "").strip()
    if not name:
        problems.append("name is required")
    stages_in = doc.get("stages")
    if not isinstance(stages_in, list) or not stages_in:
        raise CurriculumError(problems + ["stages must be a non-empty list"])
    if len(stages_in) > MAX_STAGES:
        problems.append(f"at most {MAX_STAGES} stages")

    stage_settings = cap.get("stage_settings") or {}
    resolved, seen_ids, prev = [], set(), None
    for i, raw in enumerate(stages_in):
        where = _label(i, raw if isinstance(raw, dict) else {})
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object")
            continue
        unknown = sorted(set(raw) - STAGE_KEYS)
        if unknown:
            problems.append(f"{where}: unknown fields {unknown}")
        stage = {"id": str(raw.get("id") or _slug(raw.get("name") or "") or f"stage-{i + 1}"),
                 "name": str(raw.get("name") or f"Stage {i + 1}")}
        if stage["id"] in seen_ids:
            stage["id"] = f"{stage['id']}-{i + 1}"
        seen_ids.add(stage["id"])
        kind = raw.get("kind", "train")
        if kind not in KINDS:
            problems.append(f"{where}: kind {kind!r} isn't supported yet (supported: {list(KINDS)})")
        stage["kind"] = kind
        for key in INHERITED:
            if key in raw:
                stage[key] = raw[key]
            else:
                stage[key] = prev[key] if prev else None
        if stage["gym"] is None:
            problems.append(f"{where}: gym is required")
        else:
            stage["gym"] = str(stage["gym"])
        for key in ("reward_design_id", "experiment_design_id"):
            if stage[key] is not None:
                stage[key] = str(stage[key])
        if _positive_int(stage["budget"]) is None:
            problems.append(f"{where}: budget must be a positive whole number "
                            f"({cap.get('budget_field') or 'training budget'})")
        settings = raw.get("settings") or {}
        if not isinstance(settings, dict):
            problems.append(f"{where}: settings must be an object")
            settings = {}
        for key in settings:
            if key not in stage_settings:
                problems.append(f"{where}: {robot_id} has no stage setting {key!r}"
                                + (f" (it has {sorted(stage_settings)})" if stage_settings else ""))
        stage["settings"] = {**(prev["settings"] if prev else {}), **settings}

        init = raw.get("init") or (INIT_NONE if i == 0 else INIT_PREVIOUS)
        if init not in (INIT_PREVIOUS, INIT_NONE) and not (isinstance(init, str) and init.startswith(INIT_MODEL)
                                                          and len(init) > len(INIT_MODEL)):
            problems.append(f"{where}: init must be 'previous', 'none' or 'model:<id>'")
            init = INIT_NONE
        if init == INIT_PREVIOUS and i == 0:
            problems.append(f"{where}: the first stage has no previous stage; use 'none' or 'model:<id>'")
            init = INIT_NONE
        stage["init"] = init
        same_gym = prev is not None and prev["gym"] == stage["gym"]
        same_reward = prev is not None and prev["reward_design_id"] == stage["reward_design_id"]
        stage["carry"] = _carry(raw.get("carry"), cap, init, same_gym, same_reward, where, problems)
        stage["gate"] = _gate(raw.get("gate"), cap, where, problems)
        stage["on_fail"] = _on_fail(raw.get("on_fail"), stage["budget"], where, problems)
        if stage["on_fail"]["retries"] and not stage["gate"]:
            problems.append(f"{where}: on_fail.retries needs a gate (without one the stage always passes)")
        if stage["on_fail"]["retries"] and not cap.get("carry"):
            problems.append(f"{where}: retries continue from the failed attempt's checkpoint, "
                            "which this robot can't restore yet")
        _check_refs(stage, robot_id, context, where, problems)
        resolved.append(stage)
        prev = stage

    if problems:
        raise CurriculumError(problems)
    return {
        "robot": robot_id,
        "name": name,
        "description": str(doc.get("description") or ""),
        "stages": resolved,
    }


def start_stage(curriculum, value):
    """Index of the stage a run starts on; its init must not point backwards."""
    n = len(curriculum["stages"])
    if value in (None, ""):
        return 0
    if isinstance(value, str):
        ids = [s["id"] for s in curriculum["stages"]]
        if value in ids:
            return ids.index(value)
        if not value.isdigit():
            raise CurriculumError([f"start_stage {value!r} is neither a stage id nor a number"])
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < n:
        raise CurriculumError([f"start_stage must be a stage id or an index from 0 to {n - 1}"])
    if value > 0 and curriculum["stages"][value]["init"] == INIT_PREVIOUS:
        raise CurriculumError([
            f"stage {value + 1} starts from the previous stage's checkpoint; a run starting there "
            "has none. Set that stage's init to 'model:<id>' or 'none', or start earlier"])
    return value


def gate_result(gate, metrics):
    """``(passed, details)`` for ``gate`` against an evaluation's ``metrics``."""
    if not gate:
        return True, []
    details, passed = [], True
    for c in gate["conditions"]:
        actual = (metrics or {}).get(c["metric"])
        ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) and _compare(actual, c["op"], c["value"])
        details.append({**c, "actual": actual, "passed": ok})
        passed = passed and ok
    return passed, details


def _compare(a, op, b):
    return {">=": a >= b, ">": a > b, "<=": a <= b, "<": a < b, "==": a == b}[op]


def snapshot(curriculum):
    return copy.deepcopy(curriculum["stages"])
