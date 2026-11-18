"""Declarative robot manifest registry (``robots/*/robot.yaml``).

Pure-Python (stdlib-only) loader so it runs identically in the trainers,
the dashboard's bare ``python3`` (no PyYAML installed), and CPU-only CI.
When PyYAML is available, manifests are also valid YAML 1.2.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

REQUIRED_FIELDS = ("id", "display_name", "morphology", "engine", "checkpoint_format", "supports")
REQUIRED_SUPPORTS = ("eval", "export", "video", "weakness_map", "demos", "unity_clients")
DEFAULT_ROBOT_ID = "roboracer"
CARRY_ITEMS = ("weights", "optimizer", "entropy", "normalizer", "replay", "jsrl_guide")
CURRICULUM_DEFAULTS = {
    "enabled": False,
    "budget_field": None,
    "checkpoint_field": None,
    "carry": [],
    "carry_atomic": False,
    "early_stop": False,
    "gate_metrics": [],
}


class RobotManifestError(ValueError):
    """A ``robot.yaml`` manifest is missing or malformed."""


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i]
    return line


def _split_inline(text: str) -> list[str]:
    items, cur, in_s, in_d, depth = [], [], False, False, 0
    for ch in text:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif not in_s and not in_d:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    items.append(part)
                cur = []
                continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(tail)
    return items


def _parse_scalar(raw: str):
    s = raw.strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        if s.startswith('"'):
            return json.loads(s)
        return s[1:-1].replace("''", "'")
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_parse_scalar(x) for x in _split_inline(inner)] if inner else []
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(s) if not s.startswith("0") or s in ("0",) else int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_simple_yaml(text: str) -> dict:
    """Parse nested ``key: value`` maps + inline lists without PyYAML."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        cleaned = _strip_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        content = cleaned.strip()
        if ":" not in content:
            raise RobotManifestError(f"unsupported YAML line: {raw_line!r}")
        key_raw, val_raw = content.split(":", 1)
        key = str(_parse_scalar(key_raw))
        val_raw = val_raw.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not val_raw:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(val_raw)
    return root


def load_yaml_text(text: str) -> dict:
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise RobotManifestError("manifest root must be a mapping")
        return loaded
    except ImportError:
        return _parse_simple_yaml(text)


def robots_dirs() -> list[Path]:
    env_dir = os.environ.get("ROBOTS_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        Path(__file__).resolve().parents[1] / "robots",
        Path("/robots"),
    ]
    seen, out = set(), []
    for p in candidates:
        if p is not None and p.is_dir():
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def validate_manifest(doc: dict, source: str = "<manifest>") -> dict:
    for key in REQUIRED_FIELDS:
        if key not in doc or doc[key] in (None, ""):
            raise RobotManifestError(f"{source}: missing required field {key!r}")
    supports = doc.get("supports")
    if not isinstance(supports, dict):
        raise RobotManifestError(f"{source}: 'supports' must be a mapping")
    for cap in REQUIRED_SUPPORTS:
        if cap not in supports or not isinstance(supports[cap], bool):
            raise RobotManifestError(f"{source}: supports.{cap!r} must be a boolean")
    out = copy.deepcopy(doc)
    out.setdefault("worker_queue_id", out["id"])
    out.setdefault("export_formats", [])
    out.setdefault("recurrent", False)
    out.setdefault("default_job_types", ["TRAIN", "EVAL"])
    out.setdefault("deploy_shapes", {})
    out.setdefault("assets", {})
    out["curriculum"] = _curriculum_block(out.get("curriculum"), source)
    return out


def _curriculum_block(block, source):
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise RobotManifestError(f"{source}: 'curriculum' must be a mapping")
    out = {**copy.deepcopy(CURRICULUM_DEFAULTS), **copy.deepcopy(block)}
    unknown = [c for c in out["carry"] if c not in CARRY_ITEMS]
    if unknown:
        raise RobotManifestError(f"{source}: curriculum.carry has unknown items {unknown}; "
                                 f"known: {list(CARRY_ITEMS)}")
    if out["enabled"] and not out["budget_field"]:
        raise RobotManifestError(f"{source}: curriculum.enabled needs curriculum.budget_field")
    if out["enabled"] and out["carry"] and not out["checkpoint_field"]:
        raise RobotManifestError(f"{source}: curriculum.carry needs curriculum.checkpoint_field")
    return out


# Built-in defaults used only if a container lacks both ./robots and /robots mounts.
_BUILTIN_MANIFESTS = {
    "roboracer": {
        "id": "roboracer",
        "display_name": "Roboracer (car)",
        "morphology": "wheeled",
        "description": "1/18-scale Waveshare JetRacer rover (Unity 6.6 lockstep sim + TF-Agents SAC)",
        "badge_color": "sky",
        "engine": "tf_agents_unity",
        "worker_queue_id": "roboracer",
        "models_dir": "/saved_models/robotaxi",
        "saved_models_subdir": "robotaxi",
        "media_prefix": "/saved_models/",
        "checkpoint_format": "saved_model",
        "export_formats": [],
        "recurrent": False,
        "course_label": "Course",
        "iter_label": "Iterations",
        "default_job_types": ["TRAIN", "DEMO", "EVAL", "BC_TRAINING_ONLY"],
        "supports": {
            "eval": True, "export": False, "video": True, "weakness_map": True,
            "demos": True, "unity_clients": True, "reward_specs": True, "live_view": False,
        },
        "assets": {"urdf": "robots/jetracer/jetracer.urdf"},
    },
    "microduck": {
        "id": "microduck",
        "display_name": "MicroDuck",
        "morphology": "biped",
        "description": "14-DoF miniature biped (mjlab + MuJoCo Warp / Unity + rsl_rl PPO -> ONNX)",
        "badge_color": "emerald",
        "engine": "mjlab_rsl_rl",
        "worker_queue_id": "microduck",
        "models_dir": "/microduck_rl/logs",
        "saved_models_subdir": "microduck",
        "media_prefix": "/microduck_rl/logs/",
        "checkpoint_format": "pt",
        "export_formats": ["onnx"],
        "recurrent": False,
        "course_label": "Task",
        "iter_label": "Max iters",
        "default_job_types": ["TRAIN", "EVAL", "EXPORT", "VIDEO"],
        "deploy_shapes": {"Mjlab-Velocity-": [61, 14]},
        "supports": {
            "eval": True, "export": True, "video": True, "weakness_map": False,
            "demos": False, "unity_clients": False, "reward_specs": True, "live_view": True,
        },
        "assets": {"mjcf": "third_party/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml"},
    },
}


def load_manifests(base_dir: Path | str | None = None) -> dict[str, dict]:
    """Load and validate every ``<dir>/*/robot.yaml`` found."""
    dirs = [Path(base_dir)] if base_dir is not None else robots_dirs()
    manifests: dict[str, dict] = {}
    for root in dirs:
        if not root.is_dir():
            continue
        for yaml_path in sorted(root.glob("*/robot.yaml")):
            parsed = load_yaml_text(yaml_path.read_text(encoding="utf-8"))
            doc = validate_manifest(parsed, source=str(yaml_path))
            manifests.setdefault(doc["id"], doc)
    if not manifests and base_dir is None:
        return copy.deepcopy(_BUILTIN_MANIFESTS)
    return manifests


def list_robots(base_dir: Path | str | None = None) -> list[dict]:
    manifests = load_manifests(base_dir)
    order = {"roboracer": 0, "microduck": 1, "openbeni": 2}
    return sorted(manifests.values(), key=lambda r: (order.get(r["id"], 10), r["id"]))


def get_robot(robot_id: str | None, base_dir: Path | str | None = None) -> dict | None:
    rid = robot_id or DEFAULT_ROBOT_ID
    return load_manifests(base_dir).get(rid)


def robot_ids(base_dir: Path | str | None = None) -> list[str]:
    return [r["id"] for r in list_robots(base_dir)]


def seed_robots(db, now=None, base_dir: Path | str | None = None) -> int:
    """Upsert every manifest into ``db.robots`` (keyed by ``_id = robot['id']``)."""
    import datetime
    ts = now or datetime.datetime.now(datetime.timezone.utc)
    count = 0
    for doc in list_robots(base_dir):
        record = {k: v for k, v in doc.items() if k != "_id"}
        record["updated_at"] = ts
        db.robots.update_one(
            {"_id": doc["id"]},
            {"$set": record, "$setOnInsert": {"created_at": ts}},
            upsert=True,
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    base = args[0] if args else None
    print(json.dumps({"robots": list_robots(base)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
