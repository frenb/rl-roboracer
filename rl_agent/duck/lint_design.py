"""Lint a duck reward design for the dashboard: JSON in on stdin, JSON out.

    {"code": "...", "default_terms": [...] | null, "library": [...] | null}

Needs only the standard library, so the dashboard container runs it with its
own python3; see ``reward_design.lint`` for what is checked.
"""
import json
import sys

import reward_design


def main():
    request = json.load(sys.stdin)
    result = reward_design.lint(request.get("code") or "", request.get("default_terms"),
                                request.get("library"))
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
