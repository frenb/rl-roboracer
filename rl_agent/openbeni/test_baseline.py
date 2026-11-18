"""CPU-only tests for seeding open-beni's pretrained checkpoint as a model."""
import datetime
import pathlib
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openbeni import baseline  # noqa: E402

NOW = datetime.datetime(2026, 11, 18)


class FakeModels:
    def __init__(self):
        self.docs = {}

    def update_one(self, flt, update, upsert=False):
        key = (flt["robot"], flt["location"])
        doc = self.docs.get(key)
        if doc is None:
            assert upsert
            doc = dict(update.get("$setOnInsert", {}))
            self.docs[key] = doc
        doc.update(update["$set"])


def test_seed_registers_the_checkpoint_once(tmp_path):
    ckpt = tmp_path / "policy.pt"
    ckpt.write_bytes(b"x")
    db = NS(models=FakeModels())
    assert baseline.seed(db, NOW, checkpoint=ckpt) == [str(ckpt)]
    assert baseline.seed(db, NOW, checkpoint=ckpt) == [str(ckpt)]
    assert len(db.models.docs) == 1
    doc = db.models.docs[("openbeni", str(ckpt))]
    assert "job_id" not in doc
    assert doc["task_id"] == "Beni-Jump-Flat"
    assert doc["training_iterations"] == 15000
    assert doc["create_date"] == NOW


def test_missing_checkpoint_registers_nothing(tmp_path):
    db = NS(models=FakeModels())
    assert baseline.seed(db, NOW, checkpoint=tmp_path / "absent.pt") == []
    assert db.models.docs == {}
