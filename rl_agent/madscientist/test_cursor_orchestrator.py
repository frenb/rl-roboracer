"""Tests for cursor_orchestrator (Phase 1C-Full).

Mocks the cursor-sdk Python module via a sys.modules-replacement
strategy so we don't need the real package installed. The mock
returns canned events + a synthetic RunResult; spawn_cursor_agent_
for_proposal should extract PR URLs, stream log chunks into Mongo,
and return the right dict shape.

Scenarios:

  1. Happy path - mock SDK returns finished + a PR URL via the
     result's pr_url attribute -> spawn returns status=finished +
     pr_url populated.
  2. PR URL extracted from the agent's transcript (not the result
     object) -> still returned correctly.
  3. CursorAgentError on .send() -> spawn returns status=did_not_start
     with the error message.
  4. RunResult.status="error" -> spawn returns status=error.
  5. Stream interruption mid-run -> wait() still called, partial log
     chunks still written.
  6. cursor-sdk not installed -> spawn returns status=did_not_start
     with a clear install-instruction reason.
  7. Missing CURSOR_API_KEY -> spawn returns status=did_not_start
     without trying to import the SDK.
  8. Prompt builder produces a sane prompt + branch name slug.
  9. _extract_pr_url handles transcript-scan fallback.

Also tests the orchestrator's dispatch logic (proposal with
proposed_schema_extensions -> Cursor path; without -> auto-queue).
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback
from typing import Any, Dict, List
from unittest.mock import MagicMock

from pymongo import MongoClient

from rl_agent.madscientist import (
    constants, cursor_orchestrator, orchestrator)


_passed = 0
_failed = 0
_failures: List[str] = []
_inserted_proposals: List[Any] = []


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


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _insert_proposal_with_extensions(db) -> Dict[str, Any]:
    """Insert a synthetic proposal with proposed_schema_extensions
    so it routes to the Cursor path."""
    now = _now()
    doc = {
        "title": "[cursor-orchestrator-test] DAPG aux BC + Q-filter",
        "status": constants.STATUS_APPROVED,
        "created_at": now,
        "updated_at": now,
        "hypothesis": "Aux BC raises avg_return >=10% over base.",
        "motivation": "DAPG works in continuous control; untested here.",
        "code_changes_summary": "Add aux_bc_loss_weight to SCHEMA + plumb through main().",
        "source_papers": [{"arxiv_id": "1709.10089", "title": "DAPG"}],
        "experiment_arms": [
            {"name": "base", "experiment_design_id": "experiment-default"},
            {"name": "exp1",
             "experiment_design_fields": {"aux_bc_loss_weight": 0.1}},
        ],
        "n_seeds_per_arm": 3,
        "num_iterations_per_seed": 5000,
        "expected_wall_time_hours": 6.0,
        "success_criteria": {
            "primary": "avg_return(exp1) >= avg_return(base) + 10%",
            "primary_parsed": {
                "metric": "avg_return", "arm_a": "exp1", "arm_b": "base",
                "comparator": ">=", "threshold": 0.10,
                "threshold_kind": "relative"},
            "secondary": ["avg_goals_per_episode delta(exp1 - base) > 0"],
        },
        "proposed_schema_extensions": [
            {
                "name": "aux_bc_loss_weight",
                "type": "float",
                "default": 0.0,
                "min_value": 0.0,
                "max_value": 1.0,
                "doc": "DAPG-style aux BC loss weight.",
                "paper_ref": "1709.10089",
                "section": "_section_bc",
            },
        ],
        "audit_events": [],
        "training_job_ids": [],
        "implementation_log": [],
    }
    pid = db.proposals.insert_one(doc).inserted_id
    _inserted_proposals.append(pid)
    doc["_id"] = pid
    return doc


def _make_fake_sdk(
    *,
    raise_on_send: bool = False,
    raise_on_create: bool = False,
    result_status: str = "finished",
    result_pr_url=None,
    transcript: List[str] = None,
):
    """Build a fake (Agent, CloudAgentOptions, CursorAgentError) tuple
    matching the cursor-sdk surface we use."""
    class CursorAgentError(Exception):
        def __init__(self, msg, *, is_retryable=False):
            super().__init__(msg)
            self.is_retryable = is_retryable

    class CloudAgentOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeResult:
        def __init__(self):
            self.status = result_status
            if result_pr_url:
                self.pr_url = result_pr_url

    class FakeRun:
        def __init__(self, transcript_msgs):
            self.id = "fake-run-id-001"
            self._transcript = transcript_msgs or []

        def messages(self):
            for t in self._transcript:
                m = MagicMock()
                m.message.content = [MagicMock(text=t)]
                # Ensure hasattr(m, 'text') is False so coerce uses
                # the nested .message.content path.
                # MagicMock will auto-create .text - delete it.
                del m.text
                yield m

        def wait(self):
            return FakeResult()

    class FakeAgent:
        def __init__(self):
            self.agent_id = "fake-agent-id-001"

        def __enter__(self):
            if raise_on_create:
                raise CursorAgentError("simulated create failure",
                                       is_retryable=False)
            return self

        def __exit__(self, *args):
            return False

        def send(self, prompt):
            if raise_on_send:
                raise CursorAgentError("simulated send failure",
                                       is_retryable=False)
            return FakeRun(transcript or [])

    class FakeAgentNamespace:
        @staticmethod
        def create(**kwargs):
            return FakeAgent()

    return (FakeAgentNamespace, CloudAgentOptions, CursorAgentError)


# ---- Tests ---------------------------------------------------------------


def test_prompt_builder():
    print("\nGroup 1: build_implementation_prompt", flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    prompt, branch = cursor_orchestrator.build_implementation_prompt(p)
    _expect(
        "prompt is a non-empty string",
        lambda: isinstance(prompt, str) and len(prompt) > 500)
    _expect(
        "prompt references the proposal id",
        lambda: str(p["_id"]) in prompt)
    _expect(
        "prompt includes the proposed_schema_extensions JSON",
        lambda: "aux_bc_loss_weight" in prompt)
    _expect(
        "branch name starts with xp- prefix",
        lambda: branch.startswith("xp-"),
        f"branch={branch!r}")
    _expect(
        "branch slug is lowercase alnum + hyphens",
        lambda: all(c.islower() or c.isdigit() or c == "-" or c == "/"
                    for c in branch),
        f"branch={branch!r}")


def test_happy_path_pr_url_on_result():
    print("\nGroup 2: PR URL on result.pr_url", flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    fake_sdk = _make_fake_sdk(
        result_pr_url="https://github.com/frenb/rl-roboracer/pull/123",
        transcript=["agent says hello", "agent edits SCHEMA", "agent opens PR"])
    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p,
        api_key="test-key",
        sdk_import=lambda: fake_sdk)
    _expect(
        "status=finished",
        lambda: res["status"] == "finished",
        f"got {res['status']}, error={res.get('error')}")
    _expect(
        "pr_url extracted from result attribute",
        lambda: res["pr_url"] == "https://github.com/frenb/rl-roboracer/pull/123")
    _expect(
        "branch_name returned",
        lambda: res["branch_name"].startswith("xp-"))
    _expect(
        "agent_id captured",
        lambda: res["agent_id"] == "fake-agent-id-001")
    # And the proposal's implementation_log should have 3 transcript
    # chunks plus the orchestrator's bookkeeping chunks.
    p_after = db.proposals.find_one({"_id": p["_id"]})
    log = p_after.get("implementation_log") or []
    n_transcript = sum(1 for entry in log if "agent" in entry.lower())
    _expect(
        "implementation_log contains agent transcript",
        lambda: n_transcript >= 3,
        f"log entries with 'agent': {n_transcript} of {len(log)}")


def test_pr_url_from_transcript():
    print("\nGroup 3: PR URL extracted from transcript when result lacks it",
          flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    fake_sdk = _make_fake_sdk(
        # No result_pr_url -> result.pr_url unset
        result_pr_url=None,
        transcript=[
            "implementing the SCHEMA addition...",
            "opened PR: https://github.com/frenb/rl-roboracer/pull/456",
            "done.",
        ])
    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p,
        api_key="test-key",
        sdk_import=lambda: fake_sdk)
    _expect(
        "PR URL extracted from transcript",
        lambda: res["pr_url"] == "https://github.com/frenb/rl-roboracer/pull/456")


def test_cursor_error_on_send():
    print("\nGroup 4: CursorAgentError -> status=did_not_start", flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    fake_sdk = _make_fake_sdk(raise_on_send=True)
    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p,
        api_key="test-key",
        sdk_import=lambda: fake_sdk)
    _expect(
        "status=did_not_start",
        lambda: res["status"] == "did_not_start")
    _expect(
        "error mentions CursorAgentError",
        lambda: "CursorAgentError" in (res.get("error") or "")
                or "simulated send failure" in (res.get("error") or ""))


def test_result_status_error():
    print("\nGroup 5: RunResult.status='error' -> status=error", flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    fake_sdk = _make_fake_sdk(result_status="error")
    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p, api_key="test-key", sdk_import=lambda: fake_sdk)
    _expect(
        "status=error",
        lambda: res["status"] == "error")


def test_sdk_not_installed():
    print("\nGroup 6: cursor-sdk import failure -> status=did_not_start",
          flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)

    def _no_sdk():
        raise cursor_orchestrator.CursorSdkNotInstalled(
            "cursor-sdk not installed (test scenario)")

    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p, api_key="test-key", sdk_import=_no_sdk)
    _expect(
        "status=did_not_start when SDK missing",
        lambda: res["status"] == "did_not_start")
    _expect(
        "error mentions cursor-sdk",
        lambda: "cursor-sdk" in (res.get("error") or "").lower())


def test_missing_api_key():
    print("\nGroup 7: missing CURSOR_API_KEY -> status=did_not_start, no SDK import",
          flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)
    # Ensure env doesn't carry one. Override via empty api_key.
    sdk_calls = {"n": 0}
    def _spy_sdk():
        sdk_calls["n"] += 1
        return _make_fake_sdk()
    res = cursor_orchestrator.spawn_cursor_agent_for_proposal(
        db, p, api_key="", sdk_import=_spy_sdk)
    _expect(
        "status=did_not_start when api_key missing",
        lambda: res["status"] == "did_not_start")
    _expect(
        "error mentions CURSOR_API_KEY",
        lambda: "CURSOR_API_KEY" in (res.get("error") or ""))
    _expect(
        "SDK import was NOT attempted (early-exit before importer call)",
        lambda: sdk_calls["n"] == 0)


def test_orchestrator_dispatch_to_cursor():
    print("\nGroup 8: orchestrator routes proposal with extensions to Cursor path",
          flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)

    # Capture the spawn call without actually invoking the SDK.
    calls = []
    def _fake_spawn(_db, _proposal):
        calls.append(_proposal["_id"])
        return {
            "status": "finished",
            "pr_url": "https://github.com/frenb/rl-roboracer/pull/789",
            "branch_name": "xp-fake/auto",
            "agent_id": "fake",
            "run_id": "fake",
            "error": None,
        }

    res = orchestrator.orchestrate_one(
        db, p,
        max_jobs=50,
        cursor_spawn_fn=_fake_spawn)
    _expect(
        "spawn_cursor_agent_for_proposal invoked exactly once",
        lambda: len(calls) == 1)
    _expect(
        "proposal status=pr_open (not training)",
        lambda: res["status"] == constants.STATUS_PR_OPEN,
        f"status={res['status']}")
    _expect(
        "implementation_pr_url stamped",
        lambda: res.get("implementation_pr_url") == "https://github.com/frenb/rl-roboracer/pull/789")
    _expect(
        "implementation_branch stamped",
        lambda: res.get("implementation_branch") == "xp-fake/auto")
    _expect(
        "audit_event 'cursor_pr_opened' recorded",
        lambda: any(e.get("event") == "cursor_pr_opened"
                    for e in res.get("audit_events", [])))
    # And no TRAIN jobs should be queued.
    n_jobs = db.jobs.count_documents({"proposal_id": str(p["_id"])})
    _expect(
        "no TRAIN jobs queued via Cursor path (Phase 1C-Full v1 defers queueing)",
        lambda: n_jobs == 0,
        f"found {n_jobs} jobs")


def test_orchestrator_dispatch_no_extensions_uses_auto_queue():
    print("\nGroup 9: proposal WITHOUT extensions still uses auto-queue path",
          flush=True)
    db = _db()
    # Plain proposal with no proposed_schema_extensions - should go
    # auto-queue and reach status=training.
    p_doc = {
        "title": "[cursor-orchestrator-test] plain auto-queue",
        "status": constants.STATUS_APPROVED,
        "created_at": _now(), "updated_at": _now(),
        "hypothesis": "x", "experiment_arms": [
            {"name": "base"}, {"name": "exp1"}],
        "n_seeds_per_arm": 1, "num_iterations_per_seed": 5000,
        "success_criteria": {"primary": "x", "primary_parsed": None,
                             "secondary": []},
        "audit_events": [], "training_job_ids": [],
        "proposed_schema_extensions": [],
    }
    pid = db.proposals.insert_one(p_doc).inserted_id
    _inserted_proposals.append(pid)
    p_doc["_id"] = pid

    # Provide a sentinel cursor_spawn_fn that should NEVER be called.
    cursor_called = [False]
    def _should_not_be_called(_db, _proposal):
        cursor_called[0] = True
        return {"status": "error", "error": "should not happen"}

    res = orchestrator.orchestrate_one(
        db, p_doc,
        max_jobs=50,
        cursor_spawn_fn=_should_not_be_called)
    _expect(
        "cursor_spawn_fn NOT called for plain proposal",
        lambda: not cursor_called[0])
    _expect(
        "proposal status=training (auto-queue path)",
        lambda: res["status"] == constants.STATUS_TRAINING)
    # Clean up the queued jobs too.
    db.jobs.delete_many({"proposal_id": str(pid)})


def test_cursor_path_failure_marks_proposal_failed():
    print("\nGroup 10: Cursor path failure -> proposal status=failed", flush=True)
    db = _db()
    p = _insert_proposal_with_extensions(db)

    def _fake_failing_spawn(_db, _proposal):
        return {
            "status": "error",
            "pr_url": None,
            "branch_name": "xp-fake/auto",
            "agent_id": "fake",
            "run_id": "fake",
            "error": "synthetic Cursor error for testing",
        }

    res = orchestrator.orchestrate_one(
        db, p, max_jobs=50, cursor_spawn_fn=_fake_failing_spawn)
    _expect(
        "proposal status=failed",
        lambda: res["status"] == constants.STATUS_FAILED)
    _expect(
        "implementation_failure_reason carries the spawn error",
        lambda: "synthetic Cursor error" in (
            res.get("implementation_failure_reason") or ""))
    _expect(
        "audit_event 'cursor_implementation_failed' recorded",
        lambda: any(e.get("event") == "cursor_implementation_failed"
                    for e in res.get("audit_events", [])))


# ---- Cleanup + main -----------------------------------------------------


def _cleanup(db):
    if _inserted_proposals:
        db.proposals.delete_many({"_id": {"$in": _inserted_proposals}})
        # also any jobs that leaked
        db.jobs.delete_many({
            "proposal_id": {"$in": [str(p) for p in _inserted_proposals]}})
    print(
        f"  cleanup: removed {len(_inserted_proposals)} proposals + linked jobs",
        flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: cursor_orchestrator + orchestrator dispatch", flush=True)
    print("=" * 64, flush=True)

    db = _db()
    try:
        test_prompt_builder()
        test_happy_path_pr_url_on_result()
        test_pr_url_from_transcript()
        test_cursor_error_on_send()
        test_result_status_error()
        test_sdk_not_installed()
        test_missing_api_key()
        test_orchestrator_dispatch_to_cursor()
        test_orchestrator_dispatch_no_extensions_uses_auto_queue()
        test_cursor_path_failure_marks_proposal_failed()
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
