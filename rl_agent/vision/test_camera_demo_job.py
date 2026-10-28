"""Phase 4: a camera DEMO is allowed, and it does not run the flat MLP BC.

    docker compose -f docker-compose.yml -f compose/scale.yml exec -e CUDA_VISIBLE_DEVICES= sim-controller \
        sh -c "cd /python_ws/src && python vision/test_camera_demo_job.py"
"""
from __future__ import annotations

import os
import sys
import traceback

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import collect_training_data as ctd


_passed = 0
_failed = 0


def _expect(label, predicate, detail=""):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception:
        ok = False
        detail = (detail + "\n" + traceback.format_exc()).strip()
    if ok:
        _passed += 1
        print("PASS  " + label, flush=True)
    else:
        _failed += 1
        extra = ("  " + detail) if detail else ""
        print("FAIL  " + label + extra, flush=True)


def main():
    for course in ("donut_camera", "donut_camera_no_rays"):
        _expect(
            course + " is not in COURSES_WITHOUT_DEMOS",
            lambda course=course: course not in ctd.COURSES_WITHOUT_DEMOS)
        _expect(
            course + " DEMO is allowed",
            lambda course=course: ctd.recording_job_refusal(course, "DEMO") is None)
        _expect(
            course + " skips flat MLP BC",
            lambda course=course: ctd.demo_runs_mlp_bc(course) is False)
        refused = ctd.recording_job_refusal(course, "BC_TRAINING_ONLY")
        _expect(
            course + " BC_TRAINING_ONLY is refused",
            lambda refused=refused: refused is not None and "flat MLP" in refused)

    for course in ("fly_donut", "fly_donut_flow"):
        _expect(
            course + " stays without demos",
            lambda course=course: course in ctd.COURSES_WITHOUT_DEMOS)
        _expect(
            course + " DEMO is refused",
            lambda course=course: ctd.recording_job_refusal(course, "DEMO") is not None)

    _expect(
        "donut DEMO still runs the flat MLP",
        lambda: ctd.recording_job_refusal("donut", "DEMO") is None
        and ctd.demo_runs_mlp_bc("donut") is True)

    print("%d passed, %d failed" % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
