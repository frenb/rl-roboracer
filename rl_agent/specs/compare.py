"""Compare two reward designs for the same robot by what they reward, not how they're written.

Car designs run through the same scripted episodes (fresh module state each),
and the per-step rewards and episode returns are compared. Duck designs are
compared by the reward terms they produce on a task: weight, function and
params of every term, plus the source of custom terms.

Verdicts: ``same`` (identical rewards), ``close`` (each episode's returns
differ by at most 10% of the episode's total absolute reward, and every
per-step reward has the same sign), ``differs``.
"""
import contextlib
import io
import re

import targets


# --- roboracer ---------------------------------------------------------------

class _Course:
    def __init__(self):
        self.steps_since_last_goal = 0
        self.goals_reached = 0


class _Defaults:
    def __init__(self, course):
        self._c = course

    def standard(self, data, data_arr, step_costs):
        return 0.0

    def success(self, data, data_arr, step_costs, position_history):
        return max(1, 100 - self._c.steps_since_last_goal) / 100

    def failure(self, data, data_arr, step_costs, position_history):
        return 0.0


def _segment(steps, speed, start_dist, heading=0.03, walls=25.0):
    """Steps that close ``start_dist`` to the goal at a constant rate."""
    out = []
    for i in range(steps):
        d = max(0.3, start_dist * (1 - (i + 1) / steps))
        out.append(("step", {"speed": speed, "dist_from_goal": d, "dist_from_traj": heading,
                             "forward": walls}))
    return out


EPISODES = {
    "drive, 2 goals, crash": _segment(30, 3.0, 10.0) + [("goal", {})] + _segment(25, 3.0, 9.0)
                             + [("goal", {})] + _segment(10, 3.0, 9.0)[:6] + [("crash", {"forward": 0.2})],
    "stall then crash": [("step", {"speed": 0.0, "dist_from_goal": 8.0, "dist_from_traj": 0.6})] * 40
                        + [("crash", {"speed": 0.0})],
    "fast laps, 3 goals": (_segment(12, 6.5, 10.0) + [("goal", {})]) * 3 + _segment(5, 6.5, 10.0),
    "weaving, 1 slow goal": _segment(90, 1.2, 10.0, heading=0.45, walls=4.0) + [("goal", {})]
                            + _segment(10, 1.2, 10.0, heading=0.45) + [("crash", {"forward": 0.1})],
}


def _run_car(code, events):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_car_quiet(code, events)


def _run_car_quiet(code, events):
    funcs = targets.reward_designs.load_reward_design("<compare>", code)
    course = _Course()
    kw = {"defaults": _Defaults(course), "course": course}
    arr, costs, hist = [0.0] * 32, [1.0], [(0.0, 0.0)]
    rewards = []
    for kind, fields in events:
        if kind == "step":
            course.steps_since_last_goal += 1
            data = targets._car_step(**fields)
            fn = funcs.get("reward_standard")
            r = fn(data, arr, costs, **kw) if fn else kw["defaults"].standard(data, arr, costs)
        elif kind == "goal":
            course.steps_since_last_goal += 1
            data = targets._car_step(has_reached_goal=True, dist_from_goal=0.3, **fields)
            fn = funcs.get("reward_success")
            r = fn(data, arr, costs, hist, **kw) if fn else kw["defaults"].success(data, arr, costs, hist)
            course.goals_reached += 1
            course.steps_since_last_goal = 0
        else:
            data = targets._car_step(has_crashed=True, **fields)
            fn = funcs.get("reward_failure")
            r = fn(data, arr, costs, hist, **kw) if fn else kw["defaults"].failure(data, arr, costs, hist)
        rewards.append(float(r))
    return rewards


def compare_car(original, compiled):
    episodes, verdicts = [], []
    for name, events in EPISODES.items():
        a, b = _run_car(original, events), _run_car(compiled, events)
        ra, rb = sum(a), sum(b)
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        scale = max(sum(map(abs, a)), sum(map(abs, b)), 1e-9)
        signs = all((x > 0) == (y > 0) and (x < 0) == (y < 0) for x, y in zip(a, b))
        if max_diff < 1e-6:
            v = "same"
        elif abs(ra - rb) / scale <= 0.10 and signs:
            v = "close"
        else:
            v = "differs"
        verdicts.append(v)
        episodes.append({"episode": name, "original_return": round(ra, 4), "compiled_return": round(rb, 4),
                         "max_step_diff": round(max_diff, 4), "same_signs": signs, "verdict": v})
    order = {"same": 0, "close": 1, "differs": 2}
    return {"verdict": max(verdicts, key=order.get), "episodes": episodes}


def replay_car(code, records):
    """Rewards ``code`` pays on logged steps: ``[{"kind": step|goal|crash, "car": {...},
    "steps_since_last_goal": n}]`` in order (module state carries across them)."""
    with contextlib.redirect_stdout(io.StringIO()):
        funcs = targets.reward_designs.load_reward_design("<replay>", code)
        course = _Course()
        kw = {"defaults": _Defaults(course), "course": course}
        arr, costs, hist = [0.0] * 32, [1.0], [(0.0, 0.0)]
        out = []
        for rec in records:
            course.steps_since_last_goal = rec["steps_since_last_goal"]
            data = {"car": rec["car"]}
            name = {"step": "reward_standard", "goal": "reward_success", "crash": "reward_failure"}[rec["kind"]]
            fn = funcs.get(name)
            extra = () if rec["kind"] == "step" else (hist,)
            if fn:
                r = fn(data, arr, costs, *extra, **kw)
            else:
                r = getattr(kw["defaults"], name[len("reward_"):])(data, arr, costs, *extra)
            out.append(float(r))
        return out


# --- microduck ---------------------------------------------------------------

def _norm_src(src):
    return re.sub(r"\s+", " ", re.sub(r"#.*", "", src)).strip()


def duck_terms(code, gym):
    """``{term: (func, weight, params)}`` after applying the design to the task defaults."""
    import reward_design as duck
    design = duck.compile_design("<compare>", code)
    terms = {d["name"]: (d.get("func"), d.get("weight"), d.get("params") or {})
             for d in gym.get("reward_defaults") or []}
    for name, spec in design.terms.items():
        if spec is None:
            terms.pop(name, None)
            continue
        func, weight, params = terms.get(name, (None, None, {}))
        func = spec.get("func") or func or f"term_{name}"
        weight = spec.get("weight", weight)
        params = {**params, **(spec.get("params") or {})}
        terms[name] = (func, weight, params)
    return terms, design


def _custom_sources(code, design):
    out = {}
    for name in design.funcs:
        m = re.search(rf"^def {name}\(.*?(?=^\S|\Z)", code, re.M | re.S)
        out[name] = _norm_src(m.group(0)) if m else ""
    return out


def compare_duck(original, compiled, gym):
    a, da = duck_terms(original, gym)
    b, db = duck_terms(compiled, gym)
    diffs = []
    for name in sorted(set(a) | set(b)):
        if name not in b:
            diffs.append(f"{name}: dropped in the copy (original {a[name][0]} w={a[name][1]})")
        elif name not in a:
            diffs.append(f"{name}: only in the copy ({b[name][0]} w={b[name][1]})")
        elif a[name] != b[name]:
            fa, wa, pa = a[name]
            fb, wb, pb = b[name]
            parts = []
            if fa != fb and not (fa.startswith("term_") and fb.startswith("term_")):
                parts.append(f"func {fa} -> {fb}")
            if wa != wb:
                parts.append(f"weight {wa} -> {wb}")
            if pa != pb:
                parts.append(f"params {pa} -> {pb}")
            if parts:
                diffs.append(f"{name}: " + ", ".join(parts))
    sa, sb = _custom_sources(original, da), _custom_sources(compiled, db)
    custom = list(sa.values()) != list(sb.values())
    weights_close = all(
        name in b and a[name][0] == b[name][0] and a[name][2] == b[name][2]
        and (a[name][1] == b[name][1] or (a[name][1] and b[name][1] and a[name][1] * b[name][1] > 0
                                          and abs(a[name][1] - b[name][1]) <= 0.1 * max(abs(a[name][1]), abs(b[name][1]))))
        for name in a) and set(a) == set(b)
    if not diffs and not custom:
        verdict = "same"
    elif weights_close and not custom:
        verdict = "close"
    else:
        verdict = "differs"
    return {"verdict": verdict, "term_diffs": diffs,
            "custom_terms_differ": custom, "terms": len(b)}


def compare(robot, original, compiled, gym=None):
    try:
        return compare_car(original, compiled) if robot == "roboracer" else compare_duck(original, compiled, gym or {})
    except Exception as e:  # noqa: BLE001
        return {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
