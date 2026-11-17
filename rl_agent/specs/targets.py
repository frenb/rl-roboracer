"""What a spec compiles to on each robot, and the checks compiled code must pass.

The shared vocabulary is ``SIGNALS``: what a reward can be about, in words
that fit any robot. Each target describes its native reward format, how each
signal maps onto it (or that it can't), and validates code for it:

    lint    the robot's own loader accepts the code (the same check a job runs)
    checks  behaviour the prose implies, tested without a simulator: on the
            car the three reward functions run on synthetic steps; on the duck
            each term the compiler cites for an Encourage bullet must end up
            with a positive weight and each Discourage / Never term a negative one
    cover   every Encourage / Discourage / Never bullet is either implemented
            or reported as unsupported, with a reason

Needs ``reward_designs`` and ``mjlab_core/reward_design`` importable;
numpy is optional and only used to run car code.
"""
import math
import pathlib
import re
import sys

_RL_AGENT = pathlib.Path(__file__).resolve().parents[1]
if str(_RL_AGENT) not in sys.path:
    sys.path.append(str(_RL_AGENT))

import reward_designs  # noqa: E402
from mjlab_core import reward_design as mjlab_reward_design  # noqa: E402

SIGNALS = {
    "progress": "moving toward the objective (the next goal, the commanded velocity)",
    "heading": "pointing or moving in the intended direction",
    "speed": "how fast the robot moves",
    "stability": "staying upright and balanced, not spinning or wobbling",
    "termination": "episode-ending failure: crashing, leaving the course, falling over",
    "smoothness": "small changes between consecutive actions",
    "effort": "energy, torque, joint speed or steering activity",
    "clearance": "distance kept from walls, obstacles or joint limits",
    "contact": "foot contact quality: slip, air time, swing height",
    "time": "reaching objectives quickly",
}
GRADED_SECTIONS = ("encourage", "discourage", "never")
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)\s*$", re.M)


def bullets(spec):
    """``[(section, bullet text)]`` for the sections the compiler must cover."""
    out = []
    for section in GRADED_SECTIONS:
        body = spec.section(section)
        items = _BULLET_RE.findall(body)
        if not items and body.strip():
            items = [p.strip() for p in body.split("\n\n") if p.strip()]
        out.extend((section, " ".join(item.split())) for item in items)
    return out


def check_coverage(spec, rationale, unsupported):
    """Every graded bullet is accounted for exactly where the spec put it."""
    covered = {(r.get("section", "").lower(), " ".join(str(r.get("bullet", "")).split()))
               for r in list(rationale) + list(unsupported)}
    missing = [f"{s}: {b}" for s, b in bullets(spec) if (s, b) not in covered]
    if missing:
        return ["these bullets are neither implemented nor listed as unsupported "
                "(quote them verbatim, with their section): " + "; ".join(missing)]
    return []


# --- roboracer ---------------------------------------------------------------

CAR_FIELDS = {
    "speed": "magnitude of the car's velocity, about 0 to 10",
    "dist_from_goal": "distance to the next goal (Unity units)",
    "dist_from_traj": "heading error to the next goal in degrees / 180, -1..1 (0 = pointing at it)",
    "goal_2": "angle between the car's velocity and its heading, -1..1 (sideslip)",
    "rotation_z": "the car's yaw in degrees",
    "has_crashed": "True on the step the car crashes",
    "has_reached_goal": "True on the step the car reaches its goal",
    "forward, left, right, forward_left, forward_right, forward_left_left, forward_right_right":
        "lidar distances to the nearest wall in those directions, 0..1000",
    "n_02_50 ... n_27_50, p_02_50 ... p_27_50":
        "lidar distances at 2.5-degree steps to the left (n_) and right (p_) of forward, 0..1000",
}
CAR_COURSE_ATTRS = {
    "steps_since_last_goal": "steps since the car last reached a goal",
    "goals_reached": "goals reached so far this episode",
}
CAR_SIGNALS = {
    "progress": "reward_success (fires on each goal reached); in reward_standard, the drop in "
                "dist_from_goal since the previous step (keep the previous value in a module-level dict)",
    "heading": "abs(data['car']['dist_from_traj']) near 0",
    "speed": "data['car']['speed']",
    "stability": "abs(data['car']['goal_2']) (sideslip)",
    "termination": "reward_failure (fires once, when the car crashes; the episode then ends)",
    "smoothness": "unsupported: reward functions don't see the actions",
    "effort": "unsupported: reward functions don't see the actions",
    "clearance": "min of the lidar distances",
    "contact": "unsupported: the car has no feet",
    "time": "course.steps_since_last_goal (the default success reward shrinks with it)",
}
CAR_DEFAULTS = (
    "Course defaults: reward_standard returns 0 every step; reward_success returns "
    "max(1, 100 - course.steps_since_last_goal) / 100 (0.01..1); reward_failure returns 0. "
    "An episode runs until a crash or a cap of goals. Keep per-step rewards small "
    "(|r| <= 0.1) so goal and crash rewards stay the dominant signal.")


def _car_target():
    fields = "\n".join(f"- data['car']['{k}']: {v}" if "," not in k and "..." not in k
                       else f"- {k}: {v}" for k, v in CAR_FIELDS.items())
    attrs = "\n".join(f"- course.{k}: {v}" for k, v in CAR_COURSE_ATTRS.items())
    signals = "\n".join(f"- {s}: {CAR_SIGNALS[s]}" for s in SIGNALS)
    return f"""Target: roboracer, a small car driving a closed course of goals (Unity sim, SAC).

Write a Python module defining any of these three functions (missing ones keep
the course default). Each returns a float.

def reward_standard(data, data_arr, step_costs, *, defaults, course):        # every step
def reward_success(data, data_arr, step_costs, position_history, *, defaults, course):  # goal reached
def reward_failure(data, data_arr, step_costs, position_history, *, defaults, course):  # crash

In scope: np (numpy), math, and builtins abs/min/max/sum/len/float/int/round/range/
enumerate/zip/sorted/list/dict/tuple. No imports, no I/O, no getattr/hasattr.
Module-level variables (e.g. a dict) persist across calls for the whole run, so they
can carry state between steps; reward_success starts a new goal segment and
reward_failure ends the episode, so reset per-segment state there.
defaults.standard(data, data_arr, step_costs), defaults.success(...), defaults.failure(...)
return the course default reward (same positional arguments) for "shape on top of default".

data['car'] fields:
{fields}
step_costs: list of the course's per-step path costs, step_costs[-1] is this step's.
course (read-only):
{attrs}

{CAR_DEFAULTS}

How each signal maps:
{signals}"""


class _FakeDefaults:
    def __init__(self, course):
        self._c = course

    def standard(self, data, data_arr, step_costs):
        return 0.0

    def success(self, data, data_arr, step_costs, position_history):
        return max(1, 100 - self._c.steps_since_last_goal) / 100

    def failure(self, data, data_arr, step_costs, position_history):
        return 0.0


class _FakeCourse:
    steps_since_last_goal = 40
    goals_reached = 3


def _car_step(**car):
    rays = {k: 25.0 for k in ("forward", "left", "right", "forward_left", "forward_right",
                              "forward_left_left", "forward_right_right")}
    rays.update({f"{side}_{deg // 10:02d}_{deg % 10}0": 25.0
                 for side in ("n", "p") for deg in range(25, 300, 25)})
    base = {"speed": 3.0, "dist_from_goal": 4.0, "dist_from_traj": 0.03, "goal_2": 0.02,
            "rotation_z": 90.0, "has_crashed": False, "has_reached_goal": False, "cost": 1.0}
    base.update(rays)
    base.update(car)
    return {"car": base}


def car_checks(spec, code):
    """Run the car functions on synthetic steps; ``[]`` or problems."""
    if reward_designs.np is None:
        return ["numpy is not installed here, so the car code could not be run"]
    funcs = reward_designs.load_reward_design("<spec>", code)
    course = _FakeCourse()
    kw = {"defaults": _FakeDefaults(course), "course": course}
    arr, costs, hist = [0.0] * 32, [1.0, 1.0], [(0.0, 0.0), (0.5, 0.0)]
    cruise, stall = _car_step(), _car_step(speed=0.0, dist_from_traj=0.5)
    crash = _car_step(has_crashed=True, speed=2.5, forward=0.2)
    goal = _car_step(has_reached_goal=True, dist_from_goal=0.3)

    def call(name, data, *extra):
        fn = funcs.get(name)
        if fn is None:
            default = getattr(kw["defaults"], name[len("reward_"):])
            return float(default(data, arr, costs, *extra))
        return float(fn(data, arr, costs, *extra, **kw))

    problems, values = [], {}
    cases = {"driving": ("reward_standard", cruise), "stalled": ("reward_standard", stall),
             "goal": ("reward_success", goal, hist), "crash": ("reward_failure", crash, hist)}
    for case, (name, data, *extra) in cases.items():
        try:
            v = call(name, data, *extra)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name} raised on a {case} step: {type(e).__name__}: {e}")
            continue
        if not math.isfinite(v):
            problems.append(f"{name} returned {v} on a {case} step")
        values[case] = v
    if problems:
        return problems
    if not values["crash"] < values["goal"]:
        problems.append(f"crashing ({values['crash']:.3g}) must pay less than reaching a goal "
                        f"({values['goal']:.3g})")
    if spec.section("never") and not values["crash"] < values["driving"]:
        problems.append(f"the spec has Never items, so crashing ({values['crash']:.3g}) must pay "
                        f"less than an ordinary driving step ({values['driving']:.3g})")
    return problems


def car_lint(code):
    r = reward_designs.lint_reward_design_code(code)
    return [] if r["ok"] else [r["error"]]


# --- microduck ---------------------------------------------------------------

EPISODE_END = "episode_end"
DUCK_SIGNALS = {
    "progress": "track_linear_velocity / track_angular_velocity (velocity tasks)",
    "heading": "track_angular_velocity, or a term on the base orientation",
    "speed": "the commanded velocity sets the speed; reward tracking it",
    "stability": "upright, body_ang_vel, angular_momentum",
    "termination": "the task ends episodes on falls, which forfeits the rest of the episode's "
                   f"reward; a Never item met by that alone cites the term {EPISODE_END!r}. "
                   "A library term such as is_terminated can add a penalty on top",
    "smoothness": "action_rate_l2",
    "effort": "library terms on joint torques / velocities / accelerations",
    "clearance": "dof_pos_limits, self_collisions",
    "contact": "air_time, foot_slip, foot_clearance, foot_swing_height",
    "time": "per-step terms already accrue every step of an episode",
}


def _mjlab_target(robot, task_id, gym, library, description, signals_map):
    defaults = gym.get("reward_defaults") or []
    rows = "\n".join(f"- {d['name']}: func={d.get('func')} weight={d.get('weight')} "
                     f"params={d.get('params') or {}}" for d in defaults)
    lib = "\n".join(f"- {f['name']}({', '.join(p if isinstance(p, str) else p.get('name', '?') for p in f.get('params') or [])})"
                    f": {(f.get('doc') or '').strip().splitlines()[0][:110] if f.get('doc') else ''}"
                    for f in library)
    signals = "\n".join(f"- {s}: {signals_map[s]}" for s in SIGNALS)
    return f"""Target: {robot}, {description} (trained with PPO in mjlab, thousands of
robots in parallel on the GPU). Task: {task_id} ({gym.get('name', '')}).

The reward is a weighted sum of named terms. A design is Python defining TERMS, a
delta on the task's default terms; terms it doesn't mention keep their defaults:

TERMS = {{
    "track_linear_velocity": {{"weight": 3.0}},                # reweight a default
    "foot_slip": {{"params": {{"command_threshold": 0.05}}}},   # change its params
    "head_pose_bias": None,                                     # drop a default
    "joint_vel": {{"func": "joint_vel_l2", "weight": -1e-4}},  # add a library term
    "lean_forward": {{"weight": 0.5, "params": {{"target": 0.1}}}},  # add term_lean_forward below
}}

def term_lean_forward(env, target):
    pitch = -env.robot.projected_gravity_b[:, 0]
    return torch.exp(-((pitch - target) ** 2) / 0.01)

Positive weights reward, negative weights penalise (penalty functions return
positive magnitudes). Custom term_<name>(env, **params) functions return a
Tensor[num_envs]; in scope: torch, math, env (num_envs, device, step_dt,
episode_length_buf, robot.joint_pos / joint_vel / root_link_lin_vel_b /
root_link_ang_vel_b / projected_gravity_b, action, prev_action, command(name),
sensor(name), joint_ids(regex), body_ids(regex)). Prefer default and library terms.
A default term's func cannot be replaced; drop it and add yours under a new name.

The task's default terms:
{rows}

Library functions you can add with "func":
{lib}

How each signal maps:
{signals}"""


def _duck_target(task_id, gym, library):
    return _mjlab_target("microduck", task_id, gym, library,
                         "a small bipedal robot", DUCK_SIGNALS)


def duck_effective_weights(code, gym):
    """``{term: weight or None if dropped}`` after applying the design to the defaults."""
    design = mjlab_reward_design.compile_design("<spec>", code)
    weights = {d["name"]: d.get("weight") for d in gym.get("reward_defaults") or []}
    for name, spec in design.terms.items():
        if spec is None:
            weights[name] = None
        elif "weight" in spec:
            weights[name] = spec["weight"]
    return weights


def duck_checks(spec, code, rationale, gym):
    weights = duck_effective_weights(code, gym)
    want = {"encourage": 1, "discourage": -1, "never": -1}
    problems = []
    for r in rationale:
        section = str(r.get("section", "")).lower()
        if section not in want:
            continue
        for term in r.get("terms") or []:
            if term == EPISODE_END and section == "never":
                continue
            if term not in weights:
                problems.append(f"{section} bullet {r.get('bullet')!r} cites term {term!r}, "
                                "which the compiled reward doesn't have")
            elif weights[term] is None:
                problems.append(f"{section} bullet {r.get('bullet')!r} cites {term!r}, "
                                "but the design drops it")
            elif weights[term] * want[section] <= 0:
                sign = "positive" if want[section] > 0 else "negative"
                problems.append(f"{section} bullet {r.get('bullet')!r}: {term!r} has weight "
                                f"{weights[term]}, it must be {sign}")
        if not r.get("terms"):
            problems.append(f"{section} bullet {r.get('bullet')!r}: list the reward terms that "
                            "implement it in 'terms'")
    return problems


def duck_lint(code, gym, library):
    r = mjlab_reward_design.lint(
        code, [d["name"] for d in gym.get("reward_defaults") or []] or None,
        [f["name"] for f in library] or None)
    return [] if r["ok"] else [r["error"]]


MJLAB_TARGETS = {
    "microduck": ("a small bipedal robot", DUCK_SIGNALS),
}


# --- dispatch ------------------------------------------------------------------

def target_prompt(robot, *, gym=None, library=None, task_id=None):
    if robot == "roboracer":
        return _car_target()
    desc, sigs = MJLAB_TARGETS.get(robot, ("an mjlab robot", DUCK_SIGNALS))
    return _mjlab_target(robot, task_id, gym or {}, library or [], desc, sigs)


def validate(robot, spec, code, *, rationale=(), unsupported=(), gym=None, library=None):
    """``[]`` or the problems to fix: lint first, then coverage and behaviour checks."""
    problems = car_lint(code) if robot == "roboracer" else duck_lint(code, gym or {}, library or [])
    if problems:
        return problems
    problems = check_coverage(spec, rationale, unsupported)
    if robot == "roboracer":
        problems += car_checks(spec, code)
    else:
        problems += duck_checks(spec, code, rationale, gym or {})
    return problems


def lint(robot, code, *, gym=None, library=None):
    """Loader check only, for hand-edited code."""
    return car_lint(code) if robot == "roboracer" else duck_lint(code, gym or {}, library or [])
