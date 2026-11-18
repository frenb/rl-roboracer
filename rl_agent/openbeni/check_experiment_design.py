"""Validate an OpenBeni experiment design for the dashboard: JSON in/out."""
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
