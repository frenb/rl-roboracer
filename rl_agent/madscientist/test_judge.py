"""Unit tests for the Judge worker.

Three groups of tests:
  1. compute_verdict math (pure-function, no mocks needed)
  2. parse_judge_response edge cases (pure-function, no mocks needed)
  3. judge_one end-to-end with a mocked Anthropic client (writes to
     real Mongo, cleans up its test records).

Run via:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_judge

Exit 0 = all assertions passed.

Note: judge_loop() itself is NOT unit-tested - it's just a polling
shell. We test the inner judge_one() instead.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from typing import Any, Dict, List
from unittest.mock import MagicMock

from pymongo import MongoClient

from rl_agent.madscientist import constants, judge
from rl_agent.madscientist.schemas import (
    ExperimentArm,
    PaperReference,
    Proposal,
    SuccessCriteria,
)


_passed = 0
_failed = 0
_failures: List[str] = []


def _expect(label: str, predicate, detail: str = ""):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"{detail} | EXCEPTION: {type(e).__name__}: {e}"
    if ok:
        _passed += 1
        suffix = f" - {detail}" if detail else ""
        print(f"  [PASS] {label}{suffix}", flush=True)
    else:
        _failed += 1
        _failures.append(label)
        print(f"  [FAIL] {label} - {detail}", flush=True)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ---- Group 1: compute_verdict math --------------------------------------


def test_compute_verdict():
    print("\nGroup 1: compute_verdict math", flush=True)

    # Strong accept: all 5s
    all5 = {k: 5 for k in judge._AXIS_KEYS}
    v = judge.compute_verdict(all5, [])
    _expect(
        "all 5s -> strong_accept",
        lambda: v.overall == "strong_accept",
        f"got {v.overall}, normalized={v.normalized_score}")
    _expect(
        "all 5s -> normalized=1.0",
        lambda: v.normalized_score == 1.0)

    # Reject: all 1s
    all1 = {k: 1 for k in judge._AXIS_KEYS}
    v = judge.compute_verdict(all1, [])
    _expect(
        "all 1s -> reject (normalized=0.2 < 0.350)",
        lambda: v.overall == "reject",
        f"got {v.overall}, normalized={v.normalized_score}")

    # Weak accept: mid-range
    mid = {k: 3 for k in judge._AXIS_KEYS}
    v = judge.compute_verdict(mid, [])
    _expect(
        "all 3s -> weak_accept (normalized=0.6)",
        lambda: v.overall == "weak_accept",
        f"got {v.overall}, normalized={v.normalized_score}")

    # Any zero -> forced reject (override)
    one_zero = {k: 4 for k in judge._AXIS_KEYS}
    one_zero["novelty"] = 0
    v = judge.compute_verdict(one_zero, [])
    _expect(
        "any 0 -> forced reject",
        lambda: v.overall == "reject",
        f"got {v.overall}, normalized={v.normalized_score}")

    # axes_skipped: paper_faithfulness skipped, others all 4 -> accept
    no_paper = {k: 4 for k in judge._AXIS_KEYS if k != "paper_faithfulness"}
    v = judge.compute_verdict(no_paper, ["paper_faithfulness"])
    _expect(
        "paper_faithfulness skipped, others=4 -> accept",
        lambda: v.overall == "accept",
        f"got {v.overall}, normalized={v.normalized_score}, "
        f"max_possible={v.max_possible}")
    _expect(
        "skipped axis reduces max_possible to 35",
        lambda: v.max_possible == 35)

    # Robustness bump: weak_accept -> accept when goodhart + paper = 5.
    # We need scores that land at weak_accept (0.525-0.699 normalized).
    # 6 axes at 2 + goodhart 5 + paper 5 = 22/40 = 0.55 -> weak_accept.
    # After bump -> accept.
    bumpy = {k: 2 for k in judge._AXIS_KEYS}
    bumpy["goodhart_resistance"] = 5
    bumpy["paper_faithfulness"] = 5
    v = judge.compute_verdict(bumpy, [], apply_robustness_bump=True)
    _expect(
        "bump applies when goodhart+paper both 5",
        lambda: v.bumped is True,
        f"got bumped={v.bumped}, normalized={v.normalized_score}")
    _expect(
        "bumped verdict is accept (one notch above weak_accept)",
        lambda: v.overall == "accept",
        f"got {v.overall}, bumped={v.bumped}")

    # Robustness bump NOT applied when has_zero
    bumpy_zero = dict(bumpy)
    bumpy_zero["novelty"] = 0
    v = judge.compute_verdict(bumpy_zero, [], apply_robustness_bump=True)
    _expect(
        "bump NOT applied when has_zero",
        lambda: v.bumped is False and v.overall == "reject")

    # Invalid score raises
    bad = {k: 4 for k in judge._AXIS_KEYS}
    bad["novelty"] = 9
    try:
        judge.compute_verdict(bad, [])
        _expect("score>5 raises ValueError", lambda: False)
    except ValueError:
        _expect("score>5 raises ValueError", lambda: True)

    bad = {k: 4 for k in judge._AXIS_KEYS}
    bad["novelty"] = 2.5  # not an int
    try:
        judge.compute_verdict(bad, [])
        _expect("non-int score raises ValueError", lambda: False)
    except ValueError:
        _expect("non-int score raises ValueError", lambda: True)


# ---- Group 2: parse_judge_response --------------------------------------


def test_parse_judge_response():
    print("\nGroup 2: parse_judge_response", flush=True)

    # Pure JSON
    pure = '{"scores": {"novelty": 4}, "strengths": ["X"]}'
    parsed = judge.parse_judge_response(pure)
    _expect(
        "pure JSON parses",
        lambda: parsed["scores"]["novelty"] == 4)

    # Fenced JSON
    fenced = """
```json
{"scores": {"novelty": 3}, "concerns": ["Y"]}
```
"""
    parsed = judge.parse_judge_response(fenced)
    _expect(
        "fenced JSON parses",
        lambda: parsed["scores"]["novelty"] == 3)

    # Fenced JSON without language
    fenced_nolang = """
```
{"scores": {"novelty": 2}}
```
"""
    parsed = judge.parse_judge_response(fenced_nolang)
    _expect(
        "fenced JSON without language parses",
        lambda: parsed["scores"]["novelty"] == 2)

    # Prose + JSON
    prose = """
Here's my review:

{"scores": {"novelty": 5}, "strengths": ["Z"]}

End of review.
"""
    parsed = judge.parse_judge_response(prose)
    _expect(
        "prose-surrounded JSON parses",
        lambda: parsed["scores"]["novelty"] == 5)

    # No JSON
    try:
        judge.parse_judge_response("There is no JSON here, sorry.")
        _expect("no-JSON response raises ValueError", lambda: False)
    except ValueError:
        _expect("no-JSON response raises ValueError", lambda: True)

    # Broken JSON
    broken = '{"scores": {"novelty": 4,, broken'
    try:
        judge.parse_judge_response(broken)
        _expect("broken JSON raises ValueError", lambda: False)
    except ValueError:
        _expect("broken JSON raises ValueError", lambda: True)


# ---- Group 3: judge_one end-to-end --------------------------------------


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _build_test_proposal() -> Dict[str, Any]:
    """Same shape as smoke_test.build_synthetic_proposal but title
    has a distinctive prefix so the test cleanup is precise.
    """
    now = _now()
    p = Proposal(
        title="[judge-test] DAPG aux BC + Q-filter",
        status=constants.STATUS_PENDING_JUDGE,
        created_at=now,
        updated_at=now,
        source_papers=[
            PaperReference(
                arxiv_id="1709.10089",
                title="DAPG paper",
                section_refs=["Eq. 4", "Section 3.1"],
                supporting_evidence=(
                    "Eq. 4 defines the DAPG auxiliary BC loss term that "
                    "we add to the SAC actor objective; lambda is the "
                    "knob this proposal sweeps.")),
        ],
        hypothesis="DAPG aux BC loss raises avg_return >= 10% over baseline.",
        motivation="DAPG works in continuous control; untested in our env.",
        code_changes_summary="add aux_bc_loss_weight to SCHEMA",
        experiment_arms=[
            ExperimentArm(name="base", experiment_design_id="experiment-default"),
            ExperimentArm(
                name="exp1",
                experiment_design_fields={"gamma": 0.99}),
        ],
        n_seeds_per_arm=3,
        num_iterations_per_seed=5000,
        success_criteria=SuccessCriteria(
            primary="avg_return(exp1) - avg_return(base) >= 10%",
            secondary=["avg_goals_per_episode delta > 0"]),
    )
    return p.model_dump(mode="python")


def _mock_anthropic_returning(scores: Dict[str, int],
                              axes_skipped: List[str] = None,
                              wrap_in_prose: bool = False) -> MagicMock:
    """Build a mock anthropic client that returns a canned response
    encoding the given scores."""
    response_dict = {
        "scores": scores,
        "axes_skipped": axes_skipped or [],
        "strengths": ["Clear hypothesis"],
        "concerns": ["3 seeds is borderline"],
        "suggested_revisions": ["Increase n_seeds_per_arm to 5"],
    }
    text = json.dumps(response_dict)
    if wrap_in_prose:
        text = f"Here is my review:\n\n```json\n{text}\n```\n\nThank you."

    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=4000, output_tokens=300)
    resp.stop_reason = "end_turn"

    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _mock_anthropic_garbage() -> MagicMock:
    """Mock that returns un-parseable text both times."""
    block = MagicMock()
    block.text = "I am thinking about this proposal but cannot decide."
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=4000, output_tokens=300)
    resp.stop_reason = "end_turn"

    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def test_judge_one_end_to_end():
    print("\nGroup 3: judge_one end-to-end with mocked Anthropic", flush=True)

    db = _db()

    # Cleanup any prior test records.
    db.proposals.delete_many({"title": {"$regex": r"^\[judge-test\]"}})

    # ---- 3a: pre-rubric failure short-circuits LLM ----------------------
    bad_proposal = _build_test_proposal()
    bad_proposal["hypothesis"] = ""  # triggers check B failure
    bad_proposal["_id"] = db.proposals.insert_one(bad_proposal).inserted_id

    mock_client = MagicMock()  # this should NOT be called
    updated = judge.judge_one(db, bad_proposal, mock_client)

    _expect(
        "pre-rubric failure short-circuits LLM call",
        lambda: mock_client.messages.create.call_count == 0,
        f"call_count={mock_client.messages.create.call_count}")
    _expect(
        "pre-rubric failure sets status=rejected",
        lambda: updated["status"] == constants.STATUS_REJECTED,
        f"status={updated['status']}")
    _expect(
        "pre-rubric failure stamps judge_review.overall=reject",
        lambda: updated["judge_review"]["overall"] == "reject")
    _expect(
        "pre-rubric failure populates pre_rubric_check_failures",
        lambda: "B" in updated["judge_review"]["pre_rubric_check_failures"],
        f"failures={updated['judge_review']['pre_rubric_check_failures']}")

    # ---- 3b: passing pre-rubric + good LLM response -> pending_user -----
    good_proposal = _build_test_proposal()
    good_proposal["_id"] = db.proposals.insert_one(good_proposal).inserted_id

    # Mock the LLM to return solid scores (paper_faithfulness=4 etc).
    mock_client = _mock_anthropic_returning({
        "hypothesis_specificity": 5,
        "novelty": 4,
        "significance": 4,
        "statistical_power_and_baseline_rigor": 3,
        "goodhart_resistance": 4,
        "paper_faithfulness": 4,
        "implementation_feasibility": 4,
        "cost_and_reproducibility": 4,
    })

    updated = judge.judge_one(db, good_proposal, mock_client)

    _expect(
        "LLM called exactly once",
        lambda: mock_client.messages.create.call_count == 1)
    _expect(
        "good proposal status=pending_user",
        lambda: updated["status"] == constants.STATUS_PENDING_USER,
        f"status={updated['status']}")
    _expect(
        "judge_review.overall reflects mechanical computation",
        # Sum = 5+4+4+3+4+4+4+4 = 32, /40 = 0.8 -> accept
        lambda: updated["judge_review"]["overall"] == "accept",
        f"overall={updated['judge_review']['overall']}, "
        f"normalized={updated['judge_review']['normalized_score']}")
    _expect(
        "judge_review carries strengths/concerns/revisions from LLM",
        lambda: (updated["judge_review"]["strengths"]
                 and updated["judge_review"]["concerns"]
                 and updated["judge_review"]["suggested_revisions"]))
    _expect(
        "judge_review records token count + cost",
        lambda: (updated["judge_review"]["judge_token_count"] > 0
                 and updated["judge_review"]["judge_cost_usd"] > 0))
    _expect(
        "proposal.cost.judge_usd incremented",
        lambda: updated["cost"]["judge_usd"] > 0,
        f"judge_usd={updated['cost']['judge_usd']}")
    _expect(
        "audit_events has 'judged' entry",
        lambda: any(e.get("event") == "judged"
                    for e in updated.get("audit_events", [])))

    # ---- 3c: LLM returns reject-level scores -> status=rejected ---------
    reject_proposal = _build_test_proposal()
    reject_proposal["title"] = "[judge-test] reject scenario"
    reject_proposal["_id"] = db.proposals.insert_one(reject_proposal).inserted_id

    mock_client = _mock_anthropic_returning({
        "hypothesis_specificity": 1,
        "novelty": 1,
        "significance": 1,
        "statistical_power_and_baseline_rigor": 1,
        "goodhart_resistance": 1,
        "paper_faithfulness": 1,
        "implementation_feasibility": 1,
        "cost_and_reproducibility": 1,
    })
    updated = judge.judge_one(db, reject_proposal, mock_client)

    _expect(
        "reject-level scores -> status=rejected (skips pending_user)",
        lambda: updated["status"] == constants.STATUS_REJECTED)
    _expect(
        "reject verdict overall=reject",
        lambda: updated["judge_review"]["overall"] == "reject")

    # ---- 3d: LLM returns prose-wrapped JSON -> still parses --------------
    prose_proposal = _build_test_proposal()
    prose_proposal["title"] = "[judge-test] prose-wrapped response"
    prose_proposal["_id"] = db.proposals.insert_one(prose_proposal).inserted_id

    mock_client = _mock_anthropic_returning(
        {
            "hypothesis_specificity": 4,
            "novelty": 4,
            "significance": 4,
            "statistical_power_and_baseline_rigor": 4,
            "goodhart_resistance": 4,
            "paper_faithfulness": 4,
            "implementation_feasibility": 4,
            "cost_and_reproducibility": 4,
        },
        wrap_in_prose=True,
    )
    updated = judge.judge_one(db, prose_proposal, mock_client)

    _expect(
        "prose-wrapped JSON response still produces valid review",
        lambda: updated["judge_review"]["overall"] in {"accept", "strong_accept"})

    # ---- 3e: LLM returns garbage twice -> status=rejected with reason ---
    garbage_proposal = _build_test_proposal()
    garbage_proposal["title"] = "[judge-test] garbage response"
    garbage_proposal["_id"] = db.proposals.insert_one(garbage_proposal).inserted_id

    mock_client = _mock_anthropic_garbage()
    updated = judge.judge_one(db, garbage_proposal, mock_client)

    _expect(
        "garbage LLM response calls Anthropic exactly twice (initial + 1 retry)",
        lambda: mock_client.messages.create.call_count == 2,
        f"call_count={mock_client.messages.create.call_count}")
    _expect(
        "garbage LLM response -> status=rejected",
        lambda: updated["status"] == constants.STATUS_REJECTED)
    _expect(
        "garbage LLM response surfaces parse error in concerns",
        lambda: any("LLM call failed" in c
                    for c in updated["judge_review"]["concerns"]))

    # ---- 3f: cleanup ----------------------------------------------------
    res = db.proposals.delete_many({"title": {"$regex": r"^\[judge-test\]"}})
    print(f"  cleanup: removed {res.deleted_count} test proposals", flush=True)


# ---- Main ---------------------------------------------------------------


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/judge.py", flush=True)
    print("=" * 64, flush=True)

    test_compute_verdict()
    test_parse_judge_response()
    test_judge_one_end_to_end()

    print("\n" + "=" * 64, flush=True)
    print(f"PASSED: {_passed}", flush=True)
    print(f"FAILED: {_failed}", flush=True)
    if _failures:
        for f in _failures:
            print(f"  - {f}", flush=True)
    print("=" * 64, flush=True)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
