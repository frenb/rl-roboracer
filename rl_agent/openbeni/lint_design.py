"""Lint an OpenBeni reward design for the dashboard: JSON in on stdin, JSON out."""
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
