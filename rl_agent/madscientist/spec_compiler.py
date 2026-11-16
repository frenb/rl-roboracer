"""Compile reward specs (rl_agent/specs) into each robot's reward design.

The dashboard queues a compile by setting ``compile.<robot>.state = "queued"``
on a ``design_specs`` document. This loop claims it, asks the LLM for the
robot's code plus a rationale mapping every Encourage / Discourage / Never
bullet to how it was implemented, and runs ``targets.validate`` on the answer.
Problems go back to the LLM for up to ``MAX_ROUNDS`` rounds. A passing answer
is stored as ``compile.<robot>.pending``; nothing is published until someone
accepts it on the dashboard, which writes the code into the spec text and the
``reward_designs`` collection.

Every LLM call is logged in ``spec_compile_runs``, whose cost counts against
Mad Scientist's monthly budget.

The LLM is reached through the Cursor SDK when ``CURSOR_API_KEY`` is set
(billed to the Cursor account, so logged at $0), else the Anthropic API;
``SPEC_COMPILER_BACKEND=cursor|anthropic`` forces one.
"""
from __future__ import annotations

import datetime
import os
import re
import sys
import tempfile
import time
import traceback

from . import researcher

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specs"))
import spec as specs  # noqa: E402
import targets  # noqa: E402

BACKEND = os.environ.get("SPEC_COMPILER_BACKEND") or (
    "cursor" if os.environ.get("CURSOR_API_KEY") else "anthropic")
SPEC_COMPILER_MODEL = os.environ.get("SPEC_COMPILER_MODEL") or (
    "claude-opus-5-5" if BACKEND == "cursor" else researcher.RESEARCHER_MODEL)
MAX_ROUNDS = int(os.environ.get("SPEC_COMPILER_MAX_ROUNDS", "3"))
MAX_TOKENS = 6000
CURSOR_ATTEMPTS = 4
DEFAULT_POLL_INTERVAL_SECONDS = 10

_PY_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
_JSON_RE = re.compile(r"```json\s*\n(.*?)```", re.S)

SYSTEM_PROMPT = """You compile robot-agnostic reward specs into one robot's native reward code.

A spec is Markdown: a Goal, then Encourage / Discourage / Never bullets, then
Success looks like (how runs are judged - use it to pick scales, don't reward it
directly), and optional "Notes for <robot>" sections (follow the ones for your
robot, ignore the others).

Implement the intent faithfully and conservatively:
- Every Encourage bullet raises reward, every Discourage bullet lowers it, every
  Never bullet lowers it sharply. Keep relative importance ("matters most",
  "small penalty") in the magnitudes.
- Change only what the spec asks for; keep the robot's defaults otherwise.
- If the robot cannot express a bullet (see "How each signal maps"), don't fake
  it: list it under "unsupported" with the reason.

Answer with exactly two fenced blocks and nothing else:

```python
<the complete reward code for the robot>
```

```json
{
  "rationale": [
    {"section": "encourage", "bullet": "<the bullet, verbatim>",
     "implementation": "<one sentence: how the code implements it>",
     "terms": ["<reward term names, microduck only>"]}
  ],
  "unsupported": [
    {"section": "discourage", "bullet": "<verbatim>", "reason": "<why this robot can't>"}
  ]
}
```

Every Encourage, Discourage and Never bullet appears exactly once, in either
rationale or unsupported, with section in lowercase.

"""


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def parse_answer(text):
    """``(code, rationale, unsupported)`` from the LLM's answer; raises ValueError."""
    import json
    py, js = _PY_RE.search(text), _JSON_RE.search(text)
    if not py:
        raise ValueError("the answer has no ```python block")
    if not js:
        raise ValueError("the answer has no ```json block")
    try:
        meta = json.loads(js.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"the ```json block is not valid JSON: {e}") from e
    rationale, unsupported = meta.get("rationale") or [], meta.get("unsupported") or []
    if not isinstance(rationale, list) or not isinstance(unsupported, list):
        raise ValueError("rationale and unsupported must be lists")
    return py.group(1), rationale, unsupported


def call_cursor(client, system, messages, *, max_tokens=MAX_TOKENS):
    """One Cursor agent prompt per round, carrying the whole conversation so far."""
    from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    turns = "\n\n".join(f"### {m['role']}\n{m['content']}" for m in messages)
    prompt = (f"{system}\n\nDo not use any tools or read any files: answer from this message alone, "
              f"as the next assistant turn of this conversation.\n\n{turns}")
    for attempt in range(CURSOR_ATTEMPTS):
        if attempt:
            time.sleep(15 * 2 ** (attempt - 1))
        with tempfile.TemporaryDirectory() as cwd:
            try:
                result = Agent.prompt(prompt, AgentOptions(
                    api_key=os.environ["CURSOR_API_KEY"], model=SPEC_COMPILER_MODEL,
                    local=LocalAgentOptions(cwd=cwd)))
            except CursorAgentError as e:
                error = RuntimeError(f"Cursor agent did not start: {e.message}")
                continue
        if result.status == "finished":
            return {"text": result.result or "", "input_tokens": 0, "output_tokens": 0}
        error = RuntimeError(f"Cursor agent run {result.id} ended {result.status}")
    raise error


def default_call():
    return call_cursor if BACKEND == "cursor" else researcher.call_anthropic_researcher


def _context(db, spec, robot):
    if robot != "microduck":
        return {}
    task_id = spec.setting(robot, "microduck_task")
    gym = db.gyms.find_one({"_id": f"microduck:{task_id}"})
    if not gym or not gym.get("reward_defaults"):
        raise ValueError(f"no microduck gym {task_id!r} with reward defaults; is the duck worker up?")
    catalog = db.reward_term_catalog.find_one({"_id": "microduck"}) or {}
    return {"gym": gym, "library": catalog.get("functions") or [], "task_id": task_id}


def compile_robot(db, client, spec, robot, *, spec_id=None, call=None):
    """Run the compile loop for one robot. Returns the result dict to store."""
    call = call or default_call()
    ctx = _context(db, spec, robot)
    system = SYSTEM_PROMPT + targets.target_prompt(robot, **ctx)
    messages = [{"role": "user", "content": f"Compile this spec for {robot}.\n\n{spec.prose}"}]
    rounds, problems, code, rationale, unsupported, cost = [], [], "", [], [], 0.0
    for n in range(1, MAX_ROUNDS + 1):
        resp = call(client, system, messages, max_tokens=MAX_TOKENS)
        round_cost = researcher._estimate_cost_usd(resp["input_tokens"], resp["output_tokens"])
        cost += round_cost
        try:
            code, rationale, unsupported = parse_answer(resp["text"])
            problems = targets.validate(robot, spec, code, rationale=rationale, unsupported=unsupported,
                                        gym=ctx.get("gym"), library=ctx.get("library"))
        except Exception as e:  # noqa: BLE001
            problems = [f"{type(e).__name__}: {e}"]
        db.spec_compile_runs.insert_one({
            "spec_id": spec_id, "robot": robot, "round": n, "model": SPEC_COMPILER_MODEL,
            "input_tokens": resp["input_tokens"], "output_tokens": resp["output_tokens"],
            "cost_usd": round_cost, "problems": problems, "created_at": _now()})
        rounds.append({"round": n, "problems": problems})
        if not problems:
            break
        messages += [{"role": "assistant", "content": resp["text"]},
                     {"role": "user", "content": "Your answer failed these checks; fix them and answer "
                                                 "again in the same format:\n- " + "\n- ".join(problems)}]
    return {
        "ok": not problems, "code": code, "rationale": rationale, "unsupported": unsupported,
        "problems": problems, "rounds": rounds, "cost_usd": round(cost, 4),
        "inputs_sha": spec.inputs_sha(robot), "model": SPEC_COMPILER_MODEL,
    }


def _claim(db):
    """Atomically take one queued (spec, robot) compile, or None."""
    for robot in specs.ROBOTS:
        doc = db.design_specs.find_one_and_update(
            {f"compile.{robot}.state": "queued"},
            {"$set": {f"compile.{robot}.state": "running", f"compile.{robot}.started_at": _now()}},
            sort=[(f"compile.{robot}.requested_at", 1)])
        if doc:
            return doc, robot
    return None


def run_once(db, client, *, monthly_budget_usd, call=None):
    """Process one queued compile. Returns True when there was one."""
    claimed = _claim(db)
    if not claimed:
        return False
    doc, robot = claimed
    key = f"compile.{robot}"
    try:
        budget = researcher.fetch_budget_status(db, monthly_budget_usd=monthly_budget_usd,
                                                max_proposals_per_day=0)
        if budget["remaining_this_month_usd"] <= 0:
            raise ValueError(f"Mad Scientist's monthly budget (${monthly_budget_usd:.0f}) is spent")
        spec = specs.parse(doc["text"])
        result = compile_robot(db, client, spec, robot, spec_id=doc["_id"], call=call)
        state = "pending" if result["ok"] else "failed"
        update = {f"{key}.state": state, f"{key}.finished_at": _now(), f"{key}.result": result,
                  f"{key}.error": None if result["ok"] else "the compiled code failed its checks"}
        print(f"spec_compiler: {doc.get('name')!r} for {robot}: {state} after "
              f"{len(result['rounds'])} round(s), ${result['cost_usd']:.3f}", flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        update = {f"{key}.state": "failed", f"{key}.finished_at": _now(),
                  f"{key}.error": f"{type(e).__name__}: {e}", f"{key}.result": None}
    db.design_specs.update_one({"_id": doc["_id"], f"{key}.state": "running"}, {"$set": update})
    return True


def requeue_interrupted(db):
    """Compiles left ``running`` by a restart go back in the queue."""
    for robot in specs.ROBOTS:
        db.design_specs.update_many({f"compile.{robot}.state": "running"},
                                    {"$set": {f"compile.{robot}.state": "queued"}})


def compile_loop(db, anthropic_client, *, monthly_budget_usd,
                 poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS, should_stop_fn=lambda: False):
    print(f"spec_compiler: polling every {poll_interval_seconds}s (backend={BACKEND}, model={SPEC_COMPILER_MODEL}, "
          f"max_rounds={MAX_ROUNDS})", flush=True)
    requeue_interrupted(db)
    while not should_stop_fn():
        try:
            busy = run_once(db, anthropic_client, monthly_budget_usd=monthly_budget_usd)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            busy = False
        if not busy:
            for _ in range(poll_interval_seconds):
                if should_stop_fn():
                    return
                time.sleep(1)
