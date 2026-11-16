"""Validate a duck experiment design for the dashboard: design JSON in on stdin, JSON out.

    {"ok": true, "fields": {...}} or {"ok": false, "error": "..."}

Needs only the standard library, so the dashboard container runs it with its own python3;
the duck worker runs the same ``experiment_design.validate`` when a job starts.
"""
import json
import sys

import experiment_design


def main():
    try:
        fields = experiment_design.validate(json.load(sys.stdin))
    except experiment_design.ExperimentDesignError as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout)
        return
    json.dump({"ok": True, "fields": fields}, sys.stdout)


if __name__ == "__main__":
    main()
