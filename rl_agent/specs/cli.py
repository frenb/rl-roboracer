"""stdin -> stdout JSON entry point for the dashboard's spec endpoints.

Input is ``{"op": ..., ...}``; output always has ``ok`` and, on failure,
``error``. Ops:

    parse        {text}                      -> summary of the spec
    set_compiled {text, robot, code, model}  -> {text}
    repin        {text, robot}               -> {text}
    edited       {old_text, text}            -> {robots}: compiled code changed by hand
    lint         {robot, code, gym, library} -> {problems}
    template     {}                          -> {text}
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spec as specs  # noqa: E402


def summary(spec):
    return {
        "name": spec.name,
        "kind": spec.frontmatter.get("kind"),
        "description": spec.frontmatter.get("description", ""),
        "goal": spec.section("goal"),
        "robots": spec.robots,
        "microduck_task": spec.setting("microduck", "microduck_task"),
        "openbeni_task": spec.setting("openbeni", "openbeni_task"),
        "statuses": spec.statuses(),
        "inputs_sha": {r: spec.inputs_sha(r) for r in spec.robots},
        "compiled": {r: {"code": c.code, "meta": c.meta, "edited": c.edited}
                     for r, c in spec.compiled.items()},
        "warnings": specs.warnings(spec),
    }


def run(req):
    op = req.get("op")
    if op == "parse":
        return summary(specs.parse(req["text"]))
    if op == "set_compiled":
        return {"text": specs.set_compiled(req["text"], req["robot"], req["code"],
                                           model=req.get("model", ""))}
    if op == "repin":
        return {"text": specs.repin(req["text"], req["robot"])}
    if op == "edited":
        return {"robots": specs.edited_robots(req.get("old_text") or "", req["text"])}
    if op == "lint":
        import targets
        return {"problems": targets.lint(req["robot"], req["code"], gym=req.get("gym"),
                                         library=req.get("library"))}
    if op == "template":
        return {"text": specs.TEMPLATE}
    raise specs.SpecError(f"unknown op {op!r}")


def main():
    try:
        out = {"ok": True, **run(json.loads(sys.stdin.read()))}
    except specs.SpecError as e:
        out = {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
