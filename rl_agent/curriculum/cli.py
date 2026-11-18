"""stdin -> stdout JSON entry point for the dashboard's curriculum endpoints.

Input is ``{"op": ..., ...}``; output always has ``ok`` and, on failure,
``error`` (and ``problems`` for an invalid curriculum). Ops:

    validate      {curriculum, context}       -> {curriculum}: normalized, stages resolved
    new_run       {curriculum, context, start_stage, priority}
                                              -> {run}: a RUNNING run (the caller stamps dates)
    capabilities  {}                          -> {robots: {id: curriculum block}}

``curriculum`` is the stored document (with ``_id`` and ``version`` for new_run).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import robots_registry  # noqa: E402
from curriculum import model, runner  # noqa: E402

PRIORITIES = ("front", "normal")


def robot_for(robot_id):
    robot = robots_registry.get_robot(robot_id)
    if robot is None:
        raise model.CurriculumError([f"unknown robot {robot_id!r}"])
    return robot


def run(req):
    op = req.get("op")
    if op == "validate":
        doc = req.get("curriculum") or {}
        return {"curriculum": model.validate(doc, robot_for(doc.get("robot")), req.get("context") or {})}
    if op == "new_run":
        doc = req.get("curriculum") or {}
        cur = model.validate(doc, robot_for(doc.get("robot")), req.get("context") or {})
        priority = req.get("priority") or "front"
        if priority not in PRIORITIES:
            raise model.CurriculumError([f"priority must be one of {list(PRIORITIES)}"])
        start = model.start_stage(cur, req.get("start_stage"))
        run = runner.new_run(doc, cur, start, priority, None)
        for key in ("created_at", "updated_at"):
            run.pop(key)
        return {"run": run}
    if op == "capabilities":
        return {"robots": {r["id"]: r["curriculum"] for r in robots_registry.list_robots()},
                "carry_items": list(robots_registry.CARRY_ITEMS), "ops": list(model.OPS)}
    raise model.CurriculumError([f"unknown op {op!r}"])


def main():
    try:
        out = {"ok": True, **run(json.loads(sys.stdin.read()))}
    except model.CurriculumError as e:
        out = {"ok": False, "error": str(e), "problems": e.problems}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
