"""Unit tests for declarative robot manifests (``robots/*/robot.yaml``)."""
import pathlib
import sys

import pytest

RL_AGENT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RL_AGENT.parent
if str(RL_AGENT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT))

import robots_registry  # noqa: E402


def test_repo_manifests_load_and_validate():
    robots_dir = REPO_ROOT / "robots"
    base = robots_dir if robots_dir.is_dir() else None
    robots = robots_registry.list_robots(base)
    ids = [r["id"] for r in robots]
    assert ids == ["roboracer", "microduck", "openbeni"]

    car = robots_registry.get_robot("roboracer", base)
    assert car["engine"] == "tf_agents_unity"
    assert car["supports"]["weakness_map"] is True
    assert car["supports"]["export"] is False
    if robots_dir.is_dir():
        assert (REPO_ROOT / car["assets"]["urdf"]).is_file()

    duck = robots_registry.get_robot("microduck", base)
    assert duck["engine"] == "mjlab_rsl_rl"
    assert duck["supports"]["export"] is True
    assert duck["supports"]["weakness_map"] is False
    assert duck["deploy_shapes"]["Mjlab-Velocity-"] == [61, 14]


def test_simple_yaml_parser_matches_pyyaml_if_installed(tmp_path):
    yaml = pytest.importorskip("yaml")
    sample = """
# Sample manifest
id: sample
display_name: "Sample Bot"
morphology: quadruped
engine: mjlab_rsl_rl
checkpoint_format: pt
export_formats: [onnx, mnn]
recurrent: true
deploy_shapes:
  "Task-": [40, 7]
supports:
  eval: true
  export: true
  video: true
  weakness_map: false
  demos: false
  unity_clients: false
"""
    assert robots_registry._parse_simple_yaml(sample) == yaml.safe_load(sample)
    if (REPO_ROOT / "robots").is_dir():
        for manifest in sorted((REPO_ROOT / "robots").glob("*/robot.yaml")):
            text = manifest.read_text(encoding="utf-8")
            assert robots_registry._parse_simple_yaml(text) == yaml.safe_load(text)
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "robot.yaml").write_text(sample, encoding="utf-8")
    loaded = robots_registry.list_robots(tmp_path)
    assert [r["id"] for r in loaded] == ["sample"]
    assert loaded[0]["recurrent"] is True
    assert loaded[0]["export_formats"] == ["onnx", "mnn"]


def test_validate_rejects_missing_capability():
    bad = {
        "id": "foo", "display_name": "Foo", "morphology": "quadruped",
        "engine": "mjlab_rsl_rl", "checkpoint_format": "pt",
        "supports": {"eval": True},
    }
    with pytest.raises(robots_registry.RobotManifestError):
        robots_registry.validate_manifest(bad)


def test_seed_robots_upserts_into_mongo_like_collection():
    class FakeRobotsCol:
        def __init__(self):
            self.docs = {}

        def update_one(self, filt, update, upsert=False):
            key = filt["_id"]
            doc = self.docs.setdefault(key, {"_id": key})
            if "$setOnInsert" in update and len(doc) == 1:
                doc.update(update["$setOnInsert"])
            doc.update(update["$set"])

    class FakeDb:
        def __init__(self):
            self.robots = FakeRobotsCol()

    db = FakeDb()
    base = (REPO_ROOT / "robots") if (REPO_ROOT / "robots").is_dir() else None
    count = robots_registry.seed_robots(db, base_dir=base)
    assert count == 3
    assert set(db.robots.docs) == {"roboracer", "microduck", "openbeni"}
    assert db.robots.docs["microduck"]["engine"] == "mjlab_rsl_rl"


def test_curriculum_blocks():
    base = (REPO_ROOT / "robots") if (REPO_ROOT / "robots").is_dir() else None
    beni = robots_registry.get_robot("openbeni", base)["curriculum"]
    assert beni["enabled"] is True
    assert beni["carry_atomic"] is True
    assert beni["checkpoint_field"] == "openbeni_checkpoint"
    assert "fall_rate" in beni["gate_metrics"]
    assert robots_registry.get_robot("roboracer", base)["curriculum"]["enabled"] is False
    assert robots_registry.get_robot("microduck", base)["curriculum"]["enabled"] is False


def test_curriculum_block_defaults_and_errors():
    doc = {
        "id": "foo", "display_name": "Foo", "morphology": "quadruped",
        "engine": "mjlab_rsl_rl", "checkpoint_format": "pt",
        "supports": {c: False for c in robots_registry.REQUIRED_SUPPORTS},
    }
    assert robots_registry.validate_manifest(doc)["curriculum"] == robots_registry.CURRICULUM_DEFAULTS
    with pytest.raises(robots_registry.RobotManifestError, match="unknown items"):
        robots_registry.validate_manifest({**doc, "curriculum": {"carry": ["brain"]}})
    with pytest.raises(robots_registry.RobotManifestError, match="budget_field"):
        robots_registry.validate_manifest({**doc, "curriculum": {"enabled": True}})
    with pytest.raises(robots_registry.RobotManifestError, match="checkpoint_field"):
        robots_registry.validate_manifest(
            {**doc, "curriculum": {"enabled": True, "budget_field": "max_iterations", "carry": ["weights"]}})
