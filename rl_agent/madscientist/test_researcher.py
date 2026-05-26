"""End-to-end tests for the Researcher worker.

Mocks the Anthropic client + arxiv fetcher (so tests are deterministic
+ network-free), but uses real Mongo (to verify the inserted proposal
shape is what the Judge worker will accept).

Scenarios covered:

  1. Happy path - mocked LLM returns a valid proposal that passes all
     pre-rubric checks. Verify the proposal lands in db.proposals with
     status=pending_judge + cost stamped + audit_event.
  2. LLM returns parseable JSON that FAILS check B (empty hypothesis)
     -> first revision retry with stricter prompt -> LLM "fixes" it ->
     proposal accepted.
  3. LLM returns un-parseable garbage twice -> cycle aborts, no proposal
     inserted.
  4. LLM returns proposals that fail pre-rubric checks N+1 times ->
     cycle abandons after max_revisions, no proposal inserted.
  5. Budget cap exhausted -> cycle skipped, no LLM call.
  6. Daily proposal cap hit -> cycle skipped, no LLM call.
  7. Codebase context fetcher returns reasonable JSON given current
     Mongo state.

Run via:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_researcher
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

from rl_agent.madscientist import constants, researcher


_passed = 0
_failed = 0
_failures: List[str] = []
_inserted_proposals: List[Any] = []
_inserted_research_notes: List[Any] = []


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


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


# ---- Mock fixtures -------------------------------------------------------


def _good_proposal_dict() -> Dict[str, Any]:
    """A well-formed Proposal JSON that passes ALL pre-rubric checks."""
    return {
        "title": "[researcher-test] Lower gamma in SAC actor",
        "hypothesis": (
            "Reducing gamma from 0.99 to 0.95 improves avg_return by >=5% "
            "by shortening the credit-assignment horizon."),
        "motivation": (
            "Recent paper arxiv:2308.12345 shows gamma sensitivity in "
            "continuous control. Untested in our env."),
        "code_changes_summary": (
            "Override gamma in experiment_design_fields; no SCHEMA "
            "additions."),
        "source_papers": [
            {"arxiv_id": "2308.12345", "title": "fake paper",
             "section_refs": ["Sec 4.1"]},
        ],
        "experiment_arms": [
            {"name": "base", "experiment_design_id": "experiment-default"},
            {"name": "exp1", "experiment_design_fields": {"gamma": 0.95}},
        ],
        "n_seeds_per_arm": 3,
        "num_iterations_per_seed": 5000,
        "expected_wall_time_hours": 6.0,
        "success_criteria": {
            "primary": "avg_return(exp1) >= avg_return(base) + 5%",
            "primary_parsed": {
                "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
                "comparator": ">=", "threshold": 0.05,
                "threshold_kind": "relative",
            },
            "secondary": ["avg_goals_per_episode delta(exp1 - base) > 0"],
        },
    }


def _bad_proposal_dict_empty_hypothesis() -> Dict[str, Any]:
    """Identical to _good_proposal_dict but with hypothesis=''. Fails
    pre-rubric check B."""
    p = _good_proposal_dict()
    p["hypothesis"] = ""
    p["title"] = "[researcher-test] empty hypothesis variant"
    return p


def _mock_arxiv_fetcher(query=None, days_back=None, max_papers=None):
    """Returns a deterministic list of fake arxiv papers."""
    return [
        {
            "arxiv_id": "2308.12345",
            "title": "Discount Factor Sensitivity in Continuous Control",
            "summary": "We study gamma in SAC across MuJoCo tasks...",
            "authors": ["A. Researcher", "B. Coauthor"],
            "published": _now().isoformat(),
            "primary_category": "cs.LG",
        },
    ]


def _mock_anthropic_returning(*responses: Dict[str, Any]) -> MagicMock:
    """Returns a MagicMock anthropic client whose messages.create
    cycles through the given response payloads.

    Each response_dict is either:
      * dict -> wrapped as JSON text in a synthetic Anthropic response
      * str  -> used verbatim as text (e.g., for garbage / non-JSON tests)
    """
    response_iter = iter(responses)

    def _make_resp(payload):
        if isinstance(payload, dict):
            text = json.dumps(payload)
        else:
            text = str(payload)
        block = MagicMock()
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        resp.usage = MagicMock(input_tokens=8000, output_tokens=1500)
        resp.stop_reason = "end_turn"
        return resp

    def _create(*args, **kwargs):
        try:
            return _make_resp(next(response_iter))
        except StopIteration:
            # Reuse the last payload if tests call past the prepared
            # set (defensive; tests should match payload count).
            return _make_resp("Out of canned responses.")

    client = MagicMock()
    client.messages.create.side_effect = _create
    return client


def _cleanup(db):
    if _inserted_proposals:
        db.proposals.delete_many({"_id": {"$in": _inserted_proposals}})
    if _inserted_research_notes:
        db.research_notes.delete_many({"_id": {"$in": _inserted_research_notes}})
    # Also catch any research_notes we created with the "[researcher-test]"
    # prefix in case the test crashed mid-cycle and we don't have the
    # ids.
    extra = db.research_notes.delete_many({
        "source_ref": {"$regex": r"\[researcher-test\]"}})
    if extra.deleted_count:
        print(
            f"  cleanup: also removed {extra.deleted_count} stray "
            f"research_notes",
            flush=True)
    print(
        f"  cleanup: removed {len(_inserted_proposals)} proposals, "
        f"{len(_inserted_research_notes)} explicit research_notes",
        flush=True)


def _track_proposal(p):
    if p is not None:
        _inserted_proposals.append(p["_id"])


# ---- Tests ---------------------------------------------------------------


def test_happy_path():
    print("\nGroup 1: happy path - good LLM response -> proposal inserted",
          flush=True)
    db = _db()
    client = _mock_anthropic_returning(_good_proposal_dict())
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=250.0,
        max_proposals_per_day=10,  # generous so we don't hit the rate gate
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    _track_proposal(res)

    _expect(
        "research_one_cycle returns a proposal doc",
        lambda: res is not None and res.get("_id") is not None)
    _expect(
        "status=pending_judge",
        lambda: res["status"] == constants.STATUS_PENDING_JUDGE)
    _expect(
        "title carries the LLM's title",
        lambda: "Lower gamma" in res.get("title", ""))
    _expect(
        "experiment_arms parsed (2 arms)",
        lambda: len(res.get("experiment_arms") or []) == 2)
    _expect(
        "primary_parsed populated",
        lambda: (res.get("success_criteria") or {}).get("primary_parsed") is not None)
    _expect(
        "cost.madscientist_usd > 0",
        lambda: res["cost"]["madscientist_usd"] > 0,
        f"cost={res['cost']['madscientist_usd']}")
    _expect(
        "audit_event 'drafted' stamped",
        lambda: any(e.get("event") == "drafted"
                    for e in res.get("audit_events", [])))
    _expect(
        "LLM called exactly once",
        lambda: client.messages.create.call_count == 1)


def test_self_critique_revision():
    print("\nGroup 2: LLM returns bad-then-good -> revision retry path",
          flush=True)
    db = _db()
    # First response: empty hypothesis (fails check B).
    # Second response (after revision request): the good proposal.
    client = _mock_anthropic_returning(
        _bad_proposal_dict_empty_hypothesis(),
        _good_proposal_dict())
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=250.0,
        max_proposals_per_day=10,
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    _track_proposal(res)

    _expect(
        "second attempt accepted",
        lambda: res is not None and res["status"] == constants.STATUS_PENDING_JUDGE)
    _expect(
        "LLM called exactly twice",
        lambda: client.messages.create.call_count == 2,
        f"call_count={client.messages.create.call_count}")
    # The second LLM call should have included the failure list in
    # its messages list. Verify the call args.
    last_call_kwargs = client.messages.create.call_args_list[-1].kwargs
    last_messages = last_call_kwargs.get("messages") or []
    _expect(
        "revision call carries multi-turn history (>=3 messages)",
        # Initial user + assistant + revision user = 3 minimum
        lambda: len(last_messages) >= 3,
        f"len(messages)={len(last_messages)}")
    revision_text = next(
        (m["content"] for m in reversed(last_messages)
         if m.get("role") == "user"), "")
    _expect(
        "revision prompt includes 'check B' failure",
        lambda: "check B" in revision_text or "check_b" in revision_text.lower())


def test_garbage_then_garbage_aborts():
    print("\nGroup 3: LLM returns unparseable garbage repeatedly -> cycle aborts",
          flush=True)
    db = _db()
    client = _mock_anthropic_returning(
        "Not JSON, sorry.",
        "Still not JSON.",
        "Definitely not JSON either.")
    pre_count = db.proposals.count_documents({})
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=250.0,
        max_proposals_per_day=10,
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    post_count = db.proposals.count_documents({})
    _expect(
        "cycle returns None on parse failures",
        lambda: res is None)
    _expect(
        "no proposal inserted",
        lambda: post_count == pre_count,
        f"before={pre_count}, after={post_count}")
    # Should have tried 3 times (initial + 2 revisions).
    _expect(
        "LLM called 3 times (initial + max_revisions=2 retries)",
        lambda: client.messages.create.call_count == 3,
        f"call_count={client.messages.create.call_count}")


def test_persistent_check_failure_aborts():
    print("\nGroup 4: LLM keeps returning proposals that fail pre-rubric "
          "checks -> cycle abandons after max_revisions",
          flush=True)
    db = _db()
    # All responses fail check B (empty hypothesis).
    bad = _bad_proposal_dict_empty_hypothesis()
    client = _mock_anthropic_returning(bad, bad, bad)
    pre_count = db.proposals.count_documents({})
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=250.0,
        max_proposals_per_day=10,
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    post_count = db.proposals.count_documents({})
    _expect(
        "persistent-failure cycle returns None",
        lambda: res is None)
    _expect(
        "no proposal inserted",
        lambda: post_count == pre_count)
    _expect(
        "LLM called 3 times (initial + 2 revisions)",
        lambda: client.messages.create.call_count == 3)


def test_budget_gate():
    print("\nGroup 5: empty budget aborts cycle without LLM call",
          flush=True)
    db = _db()
    client = _mock_anthropic_returning(_good_proposal_dict())
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=0.10,  # absurd cap
        max_proposals_per_day=10,
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    _expect(
        "budget-exhausted cycle returns None",
        lambda: res is None)
    _expect(
        "LLM never called",
        lambda: client.messages.create.call_count == 0)


def test_rate_gate():
    print("\nGroup 6: hitting MAX_PROPOSALS_PER_DAY aborts cycle without LLM call",
          flush=True)
    db = _db()
    # Insert one already-existing proposal stamped "today" to trip the
    # daily rate gate. We tag it with the test prefix for cleanup.
    seed_prop = {
        "title": "[researcher-test] seed for rate test",
        "status": constants.STATUS_DONE,
        "created_at": _now(),
        "updated_at": _now(),
        "experiment_arms": [], "n_seeds_per_arm": 1,
        "num_iterations_per_seed": 1, "hypothesis": "x",
        "success_criteria": {"primary": "x"},
        "cost": {"madscientist_usd": 0.0, "judge_usd": 0.0, "cursor_usd": 0.0, "other_usd": 0.0},
        "audit_events": [], "training_job_ids": [],
    }
    seed_id = db.proposals.insert_one(seed_prop).inserted_id
    _inserted_proposals.append(seed_id)

    client = _mock_anthropic_returning(_good_proposal_dict())
    res = researcher.research_one_cycle(
        db, client,
        monthly_budget_usd=250.0,
        max_proposals_per_day=1,  # already hit by the seed above
        max_revisions=2,
        arxiv_fetcher=_mock_arxiv_fetcher)
    _expect(
        "rate-limited cycle returns None",
        lambda: res is None)
    _expect(
        "LLM never called (rate gate fired before LLM)",
        lambda: client.messages.create.call_count == 0)


def test_codebase_context_shape():
    print("\nGroup 7: fetch_codebase_context returns sensible structure",
          flush=True)
    db = _db()
    ctx = researcher.fetch_codebase_context(db)
    _expect(
        "context has experiment_designs_schema",
        lambda: "experiment_designs_schema" in ctx
                and isinstance(ctx["experiment_designs_schema"], dict))
    _expect(
        "context has existing_experiment_designs list",
        lambda: isinstance(ctx.get("existing_experiment_designs"), list))
    _expect(
        "context has top_models_by_avg_return list",
        lambda: isinstance(ctx.get("top_models_by_avg_return"), list))
    # Schema should include common knobs.
    schema = ctx["experiment_designs_schema"]
    has_known_field = (
        "gamma" in schema or "batch_size" in schema
        or "bc_pretrain_steps" in schema)
    _expect(
        "schema contains at least one known field name",
        lambda: has_known_field,
        f"keys={sorted(schema.keys())[:5]}")


# ---- Main ---------------------------------------------------------------


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/researcher.py", flush=True)
    print("=" * 64, flush=True)

    db = _db()
    try:
        test_happy_path()
        test_self_critique_revision()
        test_garbage_then_garbage_aborts()
        test_persistent_check_failure_aborts()
        test_budget_gate()
        test_rate_gate()
        test_codebase_context_shape()
    finally:
        _cleanup(db)

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
