"""Turn existing reward designs into reward specs, compile them, and check the copies.

For each hand-written design in reward_designs (originals are never touched):
  1. the LLM describes what the code rewards as spec prose (Goal / Encourage /
     Discourage / Never / Success looks like, exact constants in "Notes for <robot>"),
  2. the spec is saved through the dashboard as "<name> (spec)" with
     ``source_design: <id>`` in its frontmatter,
  3. it is compiled by the spec_compiler loop like any other spec,
  4. the compiled code is compared with the original (specs/compare.py), the
     result is stored on the spec as ``conversion``, and the compile is accepted
     (published as a new reward design) only when the copy rewards the same or
     close; otherwise it stays pending on the Reward Specs tab for review.

    python -m rl_agent.madscientist.convert_designs [--only ID ...] [--redo] [--include-archived]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import sys
import time
import urllib.request

from . import spec_compiler

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specs"))
import compare  # noqa: E402
import spec as specs  # noqa: E402
import targets  # noqa: E402

DASHBOARD = os.environ.get("DASHBOARD_URL", "http://dashboard")
COMPILE_TIMEOUT_S = 20 * 60
_MD_RE = re.compile(r"```(?:markdown|md)\s*\n(.*?)```", re.S)

DECOMPILE_PROMPT = """You turn reward code into a reward spec: Markdown prose saying what the
reward asks the robot to do, which a compiler (another LLM) later turns back into
code. The copy it compiles is checked against the original by running both, so the
prose has to carry everything that changes the reward's value.

Write the body of the spec (no frontmatter), in this shape:

# <short title>

## Goal
<one short paragraph: what the robot should learn, and the idea behind this design>

## Encourage
- <each thing that earns reward, how much it matters>

## Discourage
- <each thing with a small penalty>

## Never
- <episode-ending failures with a large penalty>

## Success looks like
- <observable behaviour of a policy that learned this>

## Notes for {robot}
<the exact recipe: every constant, formula, clip, threshold and state reset in the
code, in plain words, e.g. "each step pays 0.15 x speed on top of the course's
default step reward". Say which parts come from the course defaults.>

Rules:
- Encourage / Discourage / Never are behaviour, robot-agnostic where possible; the
  numbers live in the Notes section.
- Leave out a section that would be empty. A design that only forwards to the
  defaults has a Goal saying so and Notes saying every reward is the default.
- Ignore debug printing and logging; they don't change rewards.
- Answer with one ```markdown block and nothing else.

What the {robot} reward code can see and how it is shaped:
{target}"""


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _post(path, body):
    req = urllib.request.Request(DASHBOARD + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


def _frontmatter(design, robot):
    lines = ["---", f"name: {design.get('name') or design['_id']} (spec)", "kind: reward",
             f"robots: [{robot}]"]
    if robot == "microduck":
        lines.append(f"microduck_task: {specs.DEFAULT_MICRODUCK_TASK}")
    desc = re.sub(r"\s+", " ", design.get("description") or "").strip()
    if desc:
        lines.append(f"description: {desc[:300]}")
    lines += [f"source_design: {design['_id']}", "---", ""]
    return "\n".join(lines)


def decompile(db, client, call, design, robot):
    """Spec text for ``design``; raises if the LLM can't produce a parseable one."""
    ctx = spec_compiler._context(db, specs.Spec({"name": "x", "robots": [robot]}, "", "", {}), robot) \
        if robot == "microduck" else {}
    system = DECOMPILE_PROMPT.format(robot=robot, target=targets.target_prompt(robot, **ctx))
    messages = [{"role": "user", "content": f"Reward design {design.get('name')!r}"
                 f" ({design.get('description') or 'no description'}):\n\n```python\n{design['code']}\n```"}]
    for _ in range(3):
        resp = call(client, system, messages, max_tokens=4000)
        m = _MD_RE.search(resp["text"])
        body = (m.group(1) if m else resp["text"]).strip()
        body = re.sub(r"\A---\n.*?\n---\n", "", body, flags=re.S)
        text = _frontmatter(design, robot) + body + "\n"
        try:
            specs.parse(text)
            return text
        except specs.SpecError as e:
            messages += [{"role": "assistant", "content": resp["text"]},
                         {"role": "user", "content": f"That doesn't parse as a spec: {e}. Answer again."}]
    raise RuntimeError(f"could not write a parseable spec for {design['_id']}")


def _wait_compile(db, spec_id, robot):
    deadline = time.time() + COMPILE_TIMEOUT_S
    while time.time() < deadline:
        doc = db.design_specs.find_one({"_id": spec_id}, {f"compile.{robot}": 1})
        c = (doc.get("compile") or {}).get(robot) or {}
        if c.get("state") in ("pending", "failed", "error"):
            return c
        time.sleep(5)
    raise TimeoutError(f"compile of {spec_id} for {robot} did not finish in {COMPILE_TIMEOUT_S}s")


def convert(db, client, call, design, *, redo=False, log=print):
    robot = design.get("robot") or "roboracer"
    existing = db.design_specs.find_one({"text": {"$regex": f"\nsource_design: {re.escape(str(design['_id']))}\n"}})
    if existing and not redo:
        spec_id, text = existing["_id"], existing["text"]
        log(f"[{design['_id']}] reusing spec {spec_id}")
    else:
        text = decompile(db, client, call, design, robot)
        body = {"text": text, **({"_id": str(existing["_id"])} if existing else {})}
        spec_id = _post("/save_design_spec", body)["_id"]
        spec_id = existing["_id"] if existing else _to_id(spec_id)
        log(f"[{design['_id']}] saved spec {spec_id}")

    doc = db.design_specs.find_one({"_id": spec_id})
    state = ((doc.get("compile") or {}).get(robot) or {}).get("state")
    if redo or state not in ("pending", "accepted", "queued", "running"):
        _post("/compile_design_spec", {"_id": str(spec_id), "robots": [robot]})
        log(f"[{design['_id']}] compile queued")
    if state != "accepted" or redo:
        c = _wait_compile(db, spec_id, robot)
        result = c.get("result") or {}
        code = result.get("code") or ""
    else:
        c = {"state": "accepted"}
        result = {"ok": True}
        code = specs.parse(db.design_specs.find_one({"_id": spec_id})["text"]).compiled[robot].code

    gym = db.gyms.find_one({"_id": f"microduck:{specs.DEFAULT_MICRODUCK_TASK}"}) if robot == "microduck" else None
    cmp = compare.compare(robot, design["code"], code, gym) if code else {"verdict": "error", "error": "no compiled code"}
    conversion = {"source_design": str(design["_id"]), "robot": robot, "at": _now(),
                  "compile_ok": bool(result.get("ok")), "compile_problems": result.get("problems") or [],
                  "unsupported": result.get("unsupported") or [], "comparison": cmp}
    db.design_specs.update_one({"_id": spec_id}, {"$set": {"conversion": conversion}})

    published = c.get("state") == "accepted"
    if c.get("state") == "pending" and result.get("ok") and cmp["verdict"] in ("same", "close"):
        _post("/accept_design_spec_compile", {"_id": str(spec_id), "robot": robot})
        published = True
        if design.get("archived"):
            _post("/archive_design_spec", {"_id": str(spec_id), "archived": True})
    return {"design": design, "spec_id": spec_id, "text": db.design_specs.find_one({"_id": spec_id})["text"],
            "code": code, "conversion": conversion, "published": published}


def _to_id(raw):
    from bson import ObjectId
    try:
        return ObjectId(str(raw))
    except Exception:  # noqa: BLE001
        return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--include-archived", action="store_true")
    ap.add_argument("--report", default="")
    ap.add_argument("--workers", type=int, default=1,
                    help="designs converted at once; Cursor agent starts fail under load")
    args = ap.parse_args()

    from .main import _open_anthropic, _open_mongo
    db = _open_mongo()
    client = None if spec_compiler.BACKEND == "cursor" else _open_anthropic()
    call = spec_compiler.default_call()

    query = {"spec_id": {"$exists": False}}
    designs = list(db.reward_designs.find(query))
    if args.only:
        designs = [d for d in designs if str(d["_id"]) in args.only]
    elif not args.include_archived:
        # an archived design is skipped when a live one shares its name or code
        live_codes = {d["code"] for d in designs if not d.get("archived")}
        live_names = {d.get("name") for d in designs if not d.get("archived")}
        designs = [d for d in designs if not d.get("archived")
                   or (d.get("name") not in live_names and d["code"] not in live_codes)]
    print(f"converting {len(designs)} designs with {spec_compiler.SPEC_COMPILER_MODEL} "
          f"({spec_compiler.BACKEND})", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, db, client, call, d, redo=args.redo,
                               log=lambda m: print(m, flush=True)): d for d in designs}
        for f in concurrent.futures.as_completed(futures):
            d = futures[f]
            try:
                results.append(f.result())
                r = results[-1]
                print(f"[{d['_id']}] {r['conversion']['comparison']['verdict']}"
                      f"{' (published)' if r['published'] else ''}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{d['_id']}] FAILED: {type(e).__name__}: {e}", flush=True)
                results.append({"design": d, "error": f"{type(e).__name__}: {e}"})

    out = [{"design_id": str(r["design"]["_id"]), "name": r["design"].get("name"),
            "robot": r["design"].get("robot") or "roboracer",
            **({"error": r["error"]} if "error" in r else {
                "spec_id": str(r["spec_id"]), "published": r["published"],
                "conversion": {**r["conversion"], "at": str(r["conversion"]["at"])},
                "original_code": r["design"]["code"], "compiled_code": r["code"], "spec_text": r["text"]})}
           for r in results]
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    print(json.dumps([{k: v for k, v in o.items() if k not in ("original_code", "compiled_code", "spec_text")}
                      for o in out], indent=1, default=str))


if __name__ == "__main__":
    main()
