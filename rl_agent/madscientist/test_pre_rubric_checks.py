"""Unit tests for the 7 deterministic pre-rubric checks.

Each check gets at least one PASSING case and one FAILING case, plus
sanity edge cases. We don't pull in pytest - the file is a plain
script that asserts via a small _expect() helper and reports
pass/fail counts at the end. Run via:

    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.test_pre_rubric_checks

Exit code 0 = all assertions passed.

These tests are pure-Python: NO Mongo, NO network. The proposals are
hand-crafted Pydantic instances. Anything requiring Mongo lives in
smoke_test.py instead.
"""
from __future__ import annotations

import datetime
import sys
import traceback
from typing import Callable

from rl_agent.madscientist import pre_rubric_checks
from rl_agent.madscientist.schemas import (
    Proposal,
    PaperReference,
    ExperimentArm,
    SuccessCriteria,
)


# ---- Test scaffolding ----------------------------------------------------

_passed = 0
_failed = 0
_failures: list[str] = []


def _expect(label: str, predicate: Callable[[], bool], detail: str = ""):
    global _passed, _failed
    try:
        ok = predicate()
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"{detail} | EXCEPTION: {type(e).__name__}: {e}"
    suffix = f" - {detail}" if detail else ""
    if ok:
        _passed += 1
        print(f"  [PASS] {label}{suffix}", flush=True)
    else:
        _failed += 1
        _failures.append(label)
        print(f"  [FAIL] {label}{suffix}", flush=True)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# Minimal valid proposal builder. Each test starts from this and
# mutates one field to trigger the desired check failure (or to
# stress an edge case).
def _baseline_proposal(**overrides) -> Proposal:
    defaults = dict(
        title="test",
        status="pending_judge",
        created_at=_now(),
        updated_at=_now(),
        source_papers=[],
        hypothesis="X causes Y, measured by Z.",
        code_changes_summary="add a new schema field; nothing else",
        experiment_arms=[
            ExperimentArm(name="base"),
            ExperimentArm(name="exp1"),
        ],
        n_seeds_per_arm=3,
        num_iterations_per_seed=5000,
        success_criteria=SuccessCriteria(
            primary="avg_return(exp1) - avg_return(base) >= 5%"),
    )
    defaults.update(overrides)
    return Proposal(**defaults)


# ---- Check A: arxiv resolution -------------------------------------------

def test_check_a():
    print("\nCheck A: arxiv resolution (probe_urls=False - no network)", flush=True)

    # PASS: no papers
    _expect(
        "no papers - PASS",
        lambda: pre_rubric_checks.check_a_arxiv_resolution(
            _baseline_proposal()).passed)

    # PASS: well-formed papers
    p = _baseline_proposal(source_papers=[
        PaperReference(arxiv_id="1709.10089"),
        PaperReference(arxiv_id="2104.06129v2"),
        PaperReference(arxiv_id="cs.LG/0405123"),
    ])
    _expect(
        "well-formed arxiv ids - PASS",
        lambda: pre_rubric_checks.check_a_arxiv_resolution(p).passed)

    # FAIL: missing arxiv_id - have to construct dict-style since
    # PaperReference requires arxiv_id at construct time.
    p2 = _baseline_proposal()
    p2.source_papers = [{"title": "missing arxiv_id"}]
    _expect(
        "missing arxiv_id - FAIL",
        lambda: not pre_rubric_checks.check_a_arxiv_resolution(p2).passed)

    # FAIL: arxiv_id with characters that don't match the pattern.
    # Spaces are disallowed by the regex.
    p3 = _baseline_proposal(source_papers=[
        PaperReference(arxiv_id="not an arxiv id"),
    ])
    _expect(
        "malformed arxiv_id - FAIL",
        lambda: not pre_rubric_checks.check_a_arxiv_resolution(p3).passed)


# ---- Check B: hypothesis + success_criteria.primary ---------------------

def test_check_b():
    print("\nCheck B: hypothesis + success_criteria.primary", flush=True)

    _expect(
        "non-empty hypothesis + primary - PASS",
        lambda: pre_rubric_checks.check_b_hypothesis_and_criterion(
            _baseline_proposal()).passed)

    p = _baseline_proposal(hypothesis="")
    _expect(
        "empty hypothesis - FAIL",
        lambda: not pre_rubric_checks.check_b_hypothesis_and_criterion(p).passed)

    p = _baseline_proposal(hypothesis="   ")
    _expect(
        "whitespace-only hypothesis - FAIL",
        lambda: not pre_rubric_checks.check_b_hypothesis_and_criterion(p).passed)

    p = _baseline_proposal(
        success_criteria=SuccessCriteria(primary=""))
    _expect(
        "empty primary criterion - FAIL",
        lambda: not pre_rubric_checks.check_b_hypothesis_and_criterion(p).passed)


# ---- Check C: experiment_arms -------------------------------------------

def test_check_c():
    print("\nCheck C: >=2 arms + exactly one named 'base'", flush=True)

    _expect(
        "2 arms with one base - PASS",
        lambda: pre_rubric_checks.check_c_experiment_arms(
            _baseline_proposal()).passed)

    p = _baseline_proposal(experiment_arms=[ExperimentArm(name="base")])
    _expect(
        "1 arm only - FAIL",
        lambda: not pre_rubric_checks.check_c_experiment_arms(p).passed)

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="exp1"),
        ExperimentArm(name="exp2"),
    ])
    _expect(
        "no base arm - FAIL",
        lambda: not pre_rubric_checks.check_c_experiment_arms(p).passed)

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(name="base"),
    ])
    _expect(
        "two base arms - FAIL",
        lambda: not pre_rubric_checks.check_c_experiment_arms(p).passed)

    # Case-insensitive matching: "Base" should count.
    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="Base"),
        ExperimentArm(name="exp1"),
    ])
    _expect(
        "'Base' (mixed case) counts as base - PASS",
        lambda: pre_rubric_checks.check_c_experiment_arms(p).passed)


# ---- Check D: budget ----------------------------------------------------

def test_check_d():
    print("\nCheck D: cost estimate within remaining budget", flush=True)

    p = _baseline_proposal()  # 2 arms x 3 seeds x 5000 iters x 0.5s = 5h = ~$5
    _expect(
        "moderate-cost proposal in $250 budget - PASS",
        lambda: pre_rubric_checks.check_d_budget(
            p, monthly_budget_usd=250.0, spent_so_far_usd=0.0).passed)

    _expect(
        "moderate-cost proposal at 99% spent - FAIL",
        lambda: not pre_rubric_checks.check_d_budget(
            p, monthly_budget_usd=250.0, spent_so_far_usd=248.0).passed)

    # 10M iters x 2 arms x 3 seeds x 0.5s/iter = ~8333 hours = ~$8333.
    # Vastly over even an empty $250 budget.
    p_huge = _baseline_proposal(num_iterations_per_seed=10_000_000)
    _expect(
        "10M-iter proposal - FAIL",
        lambda: not pre_rubric_checks.check_d_budget(
            p_huge, monthly_budget_usd=250.0, spent_so_far_usd=0.0).passed)


# ---- Check E: schema keys -----------------------------------------------

def test_check_e():
    print("\nCheck E: experiment_design_fields + reward_design_fields "
          "keys map to real schema entries", flush=True)

    # Provide a known schema_keys override so the test isn't coupled
    # to the live experiment_designs SCHEMA (which evolves).
    valid_keys = ["gamma", "batch_size", "bc_pretrain_steps"]

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(
            name="exp1",
            experiment_design_fields={"gamma": 0.99, "batch_size": 256}),
    ])
    _expect(
        "all known keys - PASS",
        lambda: pre_rubric_checks.check_e_schema_keys(
            p, schema_keys=valid_keys).passed)

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(
            name="exp1",
            experiment_design_fields={"nonexistent_knob": 1.0}),
    ])
    _expect(
        "unknown experiment_design key - FAIL",
        lambda: not pre_rubric_checks.check_e_schema_keys(
            p, schema_keys=valid_keys).passed)

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(
            name="exp1",
            reward_design_fields={"reward_standard": "def reward_standard(...): return 0"}),
    ])
    _expect(
        "known reward_design key (reward_standard) - PASS",
        lambda: pre_rubric_checks.check_e_schema_keys(
            p, schema_keys=valid_keys).passed)

    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(
            name="exp1",
            reward_design_fields={"reward_foobar": "..."}),
    ])
    _expect(
        "unknown reward_design key - FAIL",
        lambda: not pre_rubric_checks.check_e_schema_keys(
            p, schema_keys=valid_keys).passed)

    # Phase 1C-Full: keys declared in proposed_schema_extensions are
    # accepted even if not in SCHEMA. The orchestrator's Cursor agent
    # will add them before training runs.
    p = _baseline_proposal(experiment_arms=[
        ExperimentArm(name="base"),
        ExperimentArm(
            name="exp1",
            experiment_design_fields={"aux_bc_loss_weight": 0.1}),
    ])
    # Inject the extensions field via setattr - simpler than reaching
    # for Pydantic's full constructor here.
    p.proposed_schema_extensions = [type("X", (), {
        "name": "aux_bc_loss_weight",
        "type": "float",
        "default": 0.0,
    })()]
    _expect(
        "key in proposed_schema_extensions is accepted (Phase 1C-Full)",
        lambda: pre_rubric_checks.check_e_schema_keys(
            p, schema_keys=valid_keys).passed)


# ---- Check F: safety-critical patterns ----------------------------------

def test_check_f():
    print("\nCheck F: no safety-critical path mentions in code_changes_summary",
          flush=True)

    _expect(
        "clean summary - PASS",
        lambda: pre_rubric_checks.check_f_safety_critical(
            _baseline_proposal()).passed)

    # Each safety pattern should trigger the check.
    safety_examples = [
        "_emergency_pause_handler",
        "_get_job_lifecycle_state",
        "move_all_jobs_data",
        "rl_agent/api.py",
        "dashboard/",
    ]
    for s in safety_examples:
        p = _baseline_proposal(
            code_changes_summary=f"need to edit {s} for this to work")
        _expect(
            f"summary mentions {s!r} - FAIL",
            lambda p=p: not pre_rubric_checks.check_f_safety_critical(p).passed)

    # Empty summary should pass (nothing to detect).
    p = _baseline_proposal(code_changes_summary="")
    _expect(
        "empty summary - PASS",
        lambda: pre_rubric_checks.check_f_safety_critical(p).passed)


# ---- Check G: reward-invariant secondary --------------------------------

def test_check_g():
    print("\nCheck G: reward-design proposals must list >=1 "
          "reward-invariant secondary metric", flush=True)

    # No reward design touched - check is N/A and passes vacuously.
    _expect(
        "no reward design - PASS (vacuously)",
        lambda: pre_rubric_checks.check_g_reward_invariant_secondary(
            _baseline_proposal()).passed)

    # Reward design + reward-invariant secondary present - PASS.
    p = _baseline_proposal(
        experiment_arms=[
            ExperimentArm(name="base"),
            ExperimentArm(
                name="exp1",
                reward_design_fields={"reward_standard": "def reward_standard(...): return 0"}),
        ],
        success_criteria=SuccessCriteria(
            primary="avg_return improves",
            secondary=[
                "avg_goals_per_episode delta(exp1 - base) > 0",
            ]))
    _expect(
        "reward design + invariant secondary - PASS",
        lambda: pre_rubric_checks.check_g_reward_invariant_secondary(p).passed)

    # Reward design + only unrelated secondaries - FAIL.
    p2 = _baseline_proposal(
        experiment_arms=[
            ExperimentArm(name="base"),
            ExperimentArm(
                name="exp1",
                reward_design_fields={"reward_standard": "..."}),
        ],
        success_criteria=SuccessCriteria(
            primary="avg_return improves",
            secondary=["something unrelated"]))
    _expect(
        "reward design + no invariant secondary - FAIL",
        lambda: not pre_rubric_checks.check_g_reward_invariant_secondary(p2).passed)

    # Reward design id (not fields) also triggers the check.
    p3 = _baseline_proposal(
        experiment_arms=[
            ExperimentArm(name="base"),
            ExperimentArm(name="exp1", reward_design_id="custom-id"),
        ],
        success_criteria=SuccessCriteria(
            primary="avg_return improves",
            secondary=["something unrelated"]))
    _expect(
        "reward_design_id (not fields) + no invariant - FAIL",
        lambda: not pre_rubric_checks.check_g_reward_invariant_secondary(p3).passed)


# ---- run_all aggregator -------------------------------------------------

def test_run_all_aggregator():
    print("\nrun_all: aggregator returns all 7 results + correct all_passed",
          flush=True)

    p = _baseline_proposal()
    res = pre_rubric_checks.run_all(p)
    _expect(
        "good proposal - all_passed=True",
        lambda: res.all_passed)
    _expect(
        "good proposal - 7 results returned",
        lambda: len(res.results) == 7,
        f"got {len(res.results)}")

    # Bad proposal: empty hypothesis (B) + only 1 arm (C).
    p2 = _baseline_proposal(
        hypothesis="",
        experiment_arms=[ExperimentArm(name="base")])
    res2 = pre_rubric_checks.run_all(p2)
    failed_ids = {r.check_id for r in res2.failed}
    _expect(
        "two-failure proposal - all_passed=False",
        lambda: not res2.all_passed)
    _expect(
        "two-failure proposal - B and C both reported",
        lambda: "B" in failed_ids and "C" in failed_ids,
        f"failed_ids={sorted(failed_ids)}")


# ---- Main ---------------------------------------------------------------


def main() -> int:
    print("=" * 64, flush=True)
    print("Unit tests: rl_agent/madscientist/pre_rubric_checks.py", flush=True)
    print("=" * 64, flush=True)

    test_check_a()
    test_check_b()
    test_check_c()
    test_check_d()
    test_check_e()
    test_check_f()
    test_check_g()
    test_run_all_aggregator()

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
