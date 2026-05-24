"""User-defined reward functions for robotaxi RL training.

A "reward design" is a small Python module supplied at job-submission
time that overrides the per-step reward formulas a course would
otherwise use. The user writes three optional functions (see the
contract below); this module compiles them in a constrained namespace,
wraps them in a robustness shim that handles errors and timeouts, and
monkey-patches them onto the env's active ``BaseCourse`` instance.

Why it lives here (rather than in robotaxi.py): keeps the trainer focused
on the RL loop and lets the env-side reward-shaping logic be unit-
testable without TF / tf-agents / Reverb imports.

The user-facing contract
------------------------

A reward-design module is expected to define some subset of:

    def reward_standard(data, data_arr, step_costs, *, defaults, course):
        ...

    def reward_success(data, data_arr, step_costs, position_history, *,
                       defaults, course):
        ...

    def reward_failure(data, data_arr, step_costs, position_history, *,
                       defaults, course):
        ...

Each returns a float (the reward for that situation). Missing functions
fall through to the underlying course's scalar reward methods - so a
reward design that only changes ``reward_failure`` is a one-function
module.

The keyword-only ``defaults`` argument is an object with three methods
that return what the underlying course's default reward would have been
(``defaults.standard(...)``, ``defaults.success(...)``,
``defaults.failure(...)``). Useful for "shape on top of default" patterns
like ``return defaults.standard(data, data_arr, step_costs) + 0.05 * speed``.

The keyword-only ``course`` argument is a read-only proxy that exposes
the course instance's state attributes (counters, aggregates) so reward
functions can react to course-level facts like ``course.goals_reached``,
``course.steps_since_last_goal``, etc., without being able to mutate them.

Safety
------

User-supplied code runs inside the sim-controller process via ``exec``,
in a restricted namespace that provides ``np``, ``math``, and the function
arguments but NOT ``os`` / ``sys`` / file IO / network. This is a soft
sandbox suitable for a single-user research environment, not a security
barrier; do not deploy with multi-tenant authorisation.

Every user reward call is wrapped in a try/except that converts any
runtime exception into a single configurable "penalty" reward (default
-1.0) plus a logged warning, so a buggy reward design doesn't crash the
trainer mid-run.
"""

import math
import time
import traceback

import numpy as np


REWARD_DESIGN_NAMESPACE_BUILTINS = {
    # Tightly curated set. Provides the math we expect a user to need
    # for reward shaping; deliberately leaves out builtins that would
    # let a function reach out of process (open, __import__, exec,
    # eval, globals, locals, vars, compile, etc.).
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,                   # users can debug-print to robotaxi.out
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    # Exception types so user code can raise / catch as they would
    # in normal Python without surprising NameErrors.
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
}


class RewardDesignError(ValueError):
    """Raised when a reward design fails to load or compile.

    Caught by do_job in robotaxi.py and surfaced as a clear
    ``job.eval_error`` field; do_job marks the job DONE and the trainer
    keeps running. Distinct from a runtime exception inside a successfully
    loaded design (those are caught per-call and silently degraded to a
    penalty reward, see ``_safe_call``).
    """


class _DefaultRewards:
    """Read-only handle to a course's scalar reward methods.

    Passed to every user reward function as the ``defaults`` kwarg so
    "shape on top of the default" patterns are easy:

        def reward_standard(data, data_arr, step_costs, *, defaults, course):
            return defaults.standard(data, data_arr, step_costs) + 0.05 * speed

    The methods bind to the **original** scalar methods captured BEFORE
    the user's design was installed, so there's no risk of accidental
    infinite recursion if the user's function calls
    ``defaults.standard(...)`` from within ``reward_standard``.
    """

    def __init__(self, course):
        # Capture unbound references to the course's scalar methods at
        # the time the design is installed. install_on_course() takes
        # care to construct this BEFORE patching, so these point at the
        # untouched originals.
        self._original_standard = type(course)._compute_standard_reward
        self._original_success = type(course)._compute_success_reward
        self._original_failure = type(course)._compute_failure_reward
        self._course = course

    def standard(self, data, data_arr, step_costs):
        return self._original_standard(self._course, data, data_arr, step_costs)

    def success(self, data, data_arr, step_costs, position_history):
        return self._original_success(self._course, data, data_arr, step_costs, position_history)

    def failure(self, data, data_arr, step_costs, position_history):
        return self._original_failure(self._course, data, data_arr, step_costs, position_history)


class _CourseProxy:
    """Read-only proxy over a course instance.

    Exposes the course's public attributes (counters, aggregates) so
    reward designs can read state like ``course.goals_reached`` or
    ``course.steps_since_last_goal``, but rejects attribute writes so a
    reward function can't accidentally corrupt mid-episode course state.

    Private attributes (those starting with ``_``) and methods are
    deliberately hidden from the user; reward designs should rely on
    documented course state, not implementation internals.
    """

    def __init__(self, course):
        # Use object.__setattr__ to bypass our own __setattr__.
        object.__setattr__(self, "_course", course)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"course.{name}: private attribute is not exposed to reward "
                "designs by design. Use the documented public counters / "
                "aggregates (e.g. goals_reached, steps_since_last_goal, "
                "avg_speed_last_30, etc.) instead.")
        course = object.__getattribute__(self, "_course")
        if hasattr(course, name):
            value = getattr(course, name)
            # Don't expose bound methods or callables - we don't want
            # reward designs invoking arbitrary course functions that
            # might have side effects.
            if callable(value):
                raise AttributeError(
                    f"course.{name}: callables are not exposed to reward "
                    "designs (they may mutate course state). Read the data "
                    "you need from the step args + step_costs / "
                    "position_history instead.")
            return value
        raise AttributeError(
            f"course.{name}: no such attribute on the active course "
            f"({type(course).__name__}). Available attrs: see the course "
            "class definition in rl_agent/environments/courses/.")

    def __setattr__(self, name, value):
        raise AttributeError(
            f"course.{name}: reward designs cannot mutate course state. "
            "Course state is managed by the env step loop; if you need "
            "design-local state, store it in module-level variables in "
            "your reward-design code.")


def _safe_call(user_fn, fn_name, default_value, penalty_reward, *args, **kwargs):
    """Run a single user reward call with a timeout + exception shield.

    Returns:
      The user function's return value, or ``penalty_reward`` if it
      raised. ``default_value`` is the value the underlying course
      would have returned; we prefer the penalty over silently
      falling back to default because "the user's design is broken"
      is something they should notice, not paper over.

    Note: Python doesn't have a clean per-call wall-clock timeout
    without threads; tf-agents uses asyncio elsewhere but env steps
    are sync. We measure elapsed time AFTER the call and warn if it
    exceeded a soft budget; we don't preempt. A hot-loop'd reward
    function would still slow training to a crawl, but the user
    would see warnings in the logs telling them where to look.
    """
    SOFT_TIMEOUT_SEC = 0.1
    t0 = time.perf_counter()
    try:
        result = user_fn(*args, **kwargs)
    except Exception:
        # Single-line traceback so robotaxi.out doesn't blow up.
        tb = "\n".join(traceback.format_exc().splitlines()[-3:])
        print(
            f"[reward_design] {fn_name} raised; using penalty reward "
            f"{penalty_reward}. Last frames:\n{tb}", flush=True)
        return penalty_reward
    elapsed = time.perf_counter() - t0
    if elapsed > SOFT_TIMEOUT_SEC:
        print(
            f"[reward_design] WARNING: {fn_name} took {elapsed*1000:.1f}ms "
            f"(soft budget {SOFT_TIMEOUT_SEC*1000:.0f}ms). Reward calls run "
            "every step; consider simplifying.", flush=True)
    try:
        return float(result)
    except (TypeError, ValueError):
        print(
            f"[reward_design] {fn_name} returned a non-numeric value "
            f"({result!r}); using penalty reward {penalty_reward}.",
            flush=True)
        return penalty_reward


def load_reward_design(name, code, *, penalty_reward=-1.0):
    """Compile a reward-design module and return the three function refs.

    Args:
      name: Human-friendly name, only used in error messages.
      code: The Python source of the reward design (string).
      penalty_reward: Fallback reward value used by ``_safe_call`` when
        a user function raises or returns a non-numeric value. Default
        -1.0 (a small per-step penalty), which discourages the agent
        from exploiting buggy designs without crashing the trainer.

    Returns:
      A dict with optional keys ``reward_standard``, ``reward_success``,
      ``reward_failure`` mapping to the *raw* user functions (not yet
      wrapped in ``_safe_call``). Missing functions are absent from
      the dict; install_on_course() uses the original course methods
      as the fall-through for those.

    Raises:
      RewardDesignError if the code fails to compile/exec or if none
      of the three expected function names are defined.
    """
    namespace = {
        "__builtins__": REWARD_DESIGN_NAMESPACE_BUILTINS,
        # Common scientific-Python entry points users will expect.
        "np": np,
        "numpy": np,
        "math": math,
    }
    try:
        compiled = compile(code, f"<reward_design:{name}>", "exec")
    except SyntaxError as e:
        raise RewardDesignError(
            f"Reward design {name!r} has a syntax error: {e}") from e
    try:
        exec(compiled, namespace)  # noqa: S102 - this is intentional sandboxing
    except Exception as e:
        raise RewardDesignError(
            f"Reward design {name!r} raised during module load: "
            f"{type(e).__name__}: {e}") from e

    out = {}
    for fn_name in ("reward_standard", "reward_success", "reward_failure"):
        fn = namespace.get(fn_name)
        if fn is None:
            continue
        if not callable(fn):
            raise RewardDesignError(
                f"Reward design {name!r}: '{fn_name}' is defined but is not "
                f"callable (got {type(fn).__name__}).")
        out[fn_name] = fn

    if not out:
        raise RewardDesignError(
            f"Reward design {name!r} defines none of reward_standard, "
            "reward_success, reward_failure. At least one is required - "
            "a do-nothing design is fine, but it must contain at least "
            "one of those three function names so we know it loaded.")

    return out


def install_on_course(course, design_funcs, *, penalty_reward=-1.0):
    """Monkey-patch a course's scalar reward methods with user functions.

    For each scalar method the user defined, we install a thin wrapper
    that:
      1. Builds the keyword args (``defaults``, ``course``) on each call,
      2. Invokes the user function via ``_safe_call`` for exception and
         non-numeric-return handling,
      3. Returns the resulting float.

    Methods the user didn't define are left untouched; the existing
    course implementation serves as the fall-through.

    Args:
      course: An instance of a ``BaseCourse`` subclass.
      design_funcs: Dict returned by ``load_reward_design``.
      penalty_reward: Forwarded to ``_safe_call`` for runtime errors.

    Returns:
      A handle the caller can store on the env for diagnostics
      (currently a dict ``{installed: [fn_names], ...}``); does NOT
      include a way to uninstall, since the course is re-built per
      training job.
    """
    # Capture the originals BEFORE patching so _DefaultRewards binds
    # to the untouched methods. If we built _DefaultRewards after
    # patching, defaults.standard() would call the user's function,
    # making "shape on top of default" patterns recursive.
    defaults = _DefaultRewards(course)
    proxy = _CourseProxy(course)

    installed = []
    name_to_underlying = {
        "reward_standard": "_compute_standard_reward",
        "reward_success": "_compute_success_reward",
        "reward_failure": "_compute_failure_reward",
    }
    for user_name, underlying_name in name_to_underlying.items():
        user_fn = design_funcs.get(user_name)
        if user_fn is None:
            continue

        # The wrapper signature matches what the course's existing
        # public methods call (different per scalar method). The user
        # function gets called with a normalised arg shape: positional
        # = the natural step args, keyword = (defaults, course).
        if underlying_name == "_compute_standard_reward":
            default_fn = type(course)._compute_standard_reward
            def make_wrapper(uf=user_fn, df=default_fn, n=user_name):
                def wrapper(self, data, data_arr, step_costs):
                    default_value = df(self, data, data_arr, step_costs)
                    return _safe_call(
                        uf, n, default_value, penalty_reward,
                        data, data_arr, step_costs,
                        defaults=defaults, course=proxy)
                return wrapper
            setattr(course, underlying_name, make_wrapper().__get__(course))
        elif underlying_name == "_compute_success_reward":
            default_fn = type(course)._compute_success_reward
            def make_wrapper(uf=user_fn, df=default_fn, n=user_name):
                def wrapper(self, data, data_arr, step_costs, position_history):
                    default_value = df(self, data, data_arr, step_costs, position_history)
                    return _safe_call(
                        uf, n, default_value, penalty_reward,
                        data, data_arr, step_costs, position_history,
                        defaults=defaults, course=proxy)
                return wrapper
            setattr(course, underlying_name, make_wrapper().__get__(course))
        elif underlying_name == "_compute_failure_reward":
            default_fn = type(course)._compute_failure_reward
            def make_wrapper(uf=user_fn, df=default_fn, n=user_name):
                def wrapper(self, data, data_arr, step_costs, position_history):
                    default_value = df(self, data, data_arr, step_costs, position_history)
                    return _safe_call(
                        uf, n, default_value, penalty_reward,
                        data, data_arr, step_costs, position_history,
                        defaults=defaults, course=proxy)
                return wrapper
            setattr(course, underlying_name, make_wrapper().__get__(course))

        installed.append(user_name)

    return {"installed": installed, "penalty_reward": penalty_reward}


def lint_reward_design_code(code):
    """Validate that a reward-design code string would load cleanly.

    Used by the dashboard's ``/lint_reward_design`` endpoint to catch
    syntax errors and missing-function errors before the user saves a
    design and discovers them at the next training job's start.

    Returns:
      A dict with keys:
        ok:       bool
        error:    str | None   (set when ok is False)
        funcs:    list[str]    (function names found; only set when ok is True)

    Never raises; all errors are converted to ``ok=False, error=...``.
    """
    try:
        funcs = load_reward_design("<lint>", code)
        return {"ok": True, "error": None, "funcs": sorted(funcs.keys())}
    except RewardDesignError as e:
        return {"ok": False, "error": str(e), "funcs": []}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "funcs": []}


# Canonical seed design used as a sanity-check baseline. Inserted into
# the reward_designs MongoDB collection at sim-controller startup so the
# validation procedure ("does the design system produce results
# statistically identical to direct course training?") is runnable the
# moment the feature ships. The unique _id makes the upsert idempotent
# across container restarts.
PASSTHROUGH_DESIGN_ID = "passthrough-course-default"
PASSTHROUGH_DESIGN_NAME = "Course default (passthrough)"
PASSTHROUGH_DESIGN_CODE = '''\
"""No-op reward design.

Every function forwards directly to the underlying course's default
reward via the `defaults` accessor. Training a model with this design
should produce results statistically identical to training without
any reward design selected. Use this as:

  - A sanity check when validating the reward-design system itself
    (compare on the Analysis tab against models trained with no
    reward design).
  - A clean starting template for new shaping experiments
    ("copy + modify reward_standard").
"""

def reward_standard(data, data_arr, step_costs, *, defaults, course):
    return defaults.standard(data, data_arr, step_costs)


def reward_success(data, data_arr, step_costs, position_history, *, defaults, course):
    return defaults.success(data, data_arr, step_costs, position_history)


def reward_failure(data, data_arr, step_costs, position_history, *, defaults, course):
    return defaults.failure(data, data_arr, step_costs, position_history)
'''
