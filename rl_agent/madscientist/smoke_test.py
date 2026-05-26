"""End-to-end Pydantic + Mongo smoke test for the MadScientist schemas.

What this tests:
  1. Build a realistic synthetic Proposal via Pydantic - confirms the
     schemas accept the expected payload shape with no validation errors.
  2. Insert it into db.proposals - confirms Mongo accepts the BSON
     serialization (no unsupported types, indexes don't reject the doc).
  3. Read it back from Mongo - confirms the round-trip preserves field
     values.
  4. Re-validate the read-back doc through Pydantic - confirms BSON ->
     dict -> Pydantic recovers cleanly.
  5. Run the pre-rubric checks against it - confirms the proposal is
     judged as well-formed AND the checks themselves work end-to-end
     against a real Pydantic instance.
  6. Mutate the proposal into bad shapes and confirm pre-rubric checks
     correctly reject each one.

  7. Clean up - delete the test proposal from Mongo.

Run from the host:
    docker compose -f docker-compose.yml -f compose/scale.yml exec \
        madscientist python -m rl_agent.madscientist.smoke_test

Exit code 0 = all assertions passed. Non-zero = at least one failed.

NOTE: this test writes to the REAL db.proposals collection. It uses a
distinctive title prefix ("[smoke-test]") + always cleans up afterward,
but if you interrupt mid-run you may need to manually:

    db.proposals.deleteMany({title: /^\\[smoke-test\\]/})

to clear the test record.
"""
from __future__ import annotations

import datetime
import os
import sys
import traceback

from pymongo import MongoClient

# Inside the container the bind mount + PYTHONPATH=/repo make these
# imports work. From elsewhere (e.g., a CI runner without the bind
# mount) the test won't run; that's expected.
from rl_agent.madscientist import constants
from rl_agent.madscientist import pre_rubric_checks
from rl_agent.madscientist.schemas import (
    Proposal,
    PaperReference,
    ExperimentArm,
    SuccessCriteria,
)


# ---- Test fixture --------------------------------------------------------


def build_synthetic_proposal() -> Proposal:
    """A realistic-ish proposal: 'DAPG-style aux BC loss with Q-filter'.

    Hits most of the interesting parts of the schema:
      * Multiple papers in source_papers
      * 3 experiment arms (base + 2 variants)
      * Both experiment_design_fields and reward_design_fields used
      * A primary criterion + 2 secondary criteria (one of which is
        a reward-invariant metric, so check G is happy).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return Proposal(
        title="[smoke-test] DAPG-style aux BC loss with Q-filter",
        status=constants.STATUS_PENDING_JUDGE,
        created_at=now,
        updated_at=now,
        git_sha_at_proposal="0000000000000000000000000000000000000000",
        git_branch_at_proposal="smoke-test",
        source_papers=[
            PaperReference(
                arxiv_id="1709.10089",
                title="Learning Complex Dexterous Manipulation with Deep RL and Demonstrations",
                authors=["Aravind Rajeswaran", "Vikash Kumar", "Abhishek Gupta"],
                section_refs=["Eq. 4 - DAPG aux loss"],
                url="https://arxiv.org/abs/1709.10089",
            ),
            PaperReference(
                arxiv_id="1709.10087",
                title="Overcoming Exploration in RL with Demonstrations",
                section_refs=["Section 4.3 - Q-filter"],
            ),
        ],
        hypothesis=(
            "Adding a Q-filter-gated BC loss to SAC's actor update raises "
            "avg_return by >= 10% over the canonical baseline at 5000 iters."
        ),
        motivation=(
            "DAPG (Rajeswaran 2017) and Nair 2018's Q-filter both showed "
            "consistent gains in continuous control, but neither has been "
            "tested in this codebase's robotaxi env with our SAC + demo "
            "buffer config."
        ),
        code_changes_summary=(
            "Add aux_bc_loss_weight (float, default 0.0) and q_filter_enabled "
            "(bool, default false) to experiment_designs SCHEMA. Subclass "
            "SacAgent.train() to interleave a BC step weighted by "
            "aux_bc_loss_weight, optionally gated by Q-filter."
        ),
        experiment_arms=[
            ExperimentArm(
                name="base",
                description="Canonical Default experiment design + reward; no aux BC loss",
                experiment_design_id="experiment-default",
            ),
            ExperimentArm(
                name="exp1",
                description="Aux BC loss with weight 0.1, no Q-filter",
                experiment_design_fields={
                    "bc_pretrain_steps": 5000,
                    # Note: 'aux_bc_loss_weight' is the NEW field this
                    # proposal proposes to ADD. For the smoke test we
                    # use an existing SCHEMA key (gamma) so check E
                    # doesn't fail. A real DAPG proposal would carry
                    # the schema-extension request in
                    # code_changes_summary and check E would skip the
                    # new key.
                    "gamma": 0.99,
                },
            ),
            ExperimentArm(
                name="exp2",
                description="Aux BC loss with weight 0.1, Q-filter enabled",
                experiment_design_fields={
                    "bc_pretrain_steps": 5000,
                    "gamma": 0.99,
                },
            ),
        ],
        n_seeds_per_arm=3,
        num_iterations_per_seed=5000,
        expected_wall_time_hours=6.0,
        success_criteria=SuccessCriteria(
            primary=(
                "mean(avg_return | arm=exp2, train_step>=5000) - "
                "mean(avg_return | arm=base, train_step>=5000) >= 10% "
                "with bootstrap 95% CI excluding 0."
            ),
            secondary=[
                "avg_goals_per_episode delta(exp2 - base) > 0",
                "no regression in avg_speed",
            ],
        ),
    )


# ---- Test runner ---------------------------------------------------------


def _db():
    url = os.environ.get(
        "MONGO_URL", "mongodb://root:example@mongo:27017/")
    client = MongoClient(url, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client.robotaxi


def _print_check(label: str, passed: bool, detail: str = ""):
    marker = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"  [{marker}] {label}{suffix}", flush=True)


def main() -> int:
    print("=" * 64, flush=True)
    print("MadScientist smoke test - Pydantic + Mongo + pre-rubric checks", flush=True)
    print("=" * 64, flush=True)

    failures: list[str] = []
    inserted_id = None

    try:
        # ---- Step 1: build via Pydantic --------------------------------
        print("\nStep 1: build synthetic Proposal via Pydantic...", flush=True)
        try:
            proposal = build_synthetic_proposal()
            _print_check(
                "Proposal validates",
                True,
                f"title={proposal.title!r}, status={proposal.status!r}")
        except Exception as e:  # noqa: BLE001
            _print_check("Proposal validates", False, str(e))
            failures.append("step 1: Pydantic validation")
            return 1

        # ---- Step 2: insert to Mongo -----------------------------------
        print("\nStep 2: insert into db.proposals...", flush=True)
        db = _db()
        try:
            doc = proposal.model_dump(mode="python")
            ins = db.proposals.insert_one(doc)
            inserted_id = ins.inserted_id
            _print_check(
                "insert_one succeeded",
                True,
                f"inserted_id={inserted_id}")
        except Exception as e:  # noqa: BLE001
            _print_check("insert_one succeeded", False, str(e))
            failures.append("step 2: Mongo insert")
            return 1

        # ---- Step 3: read back -----------------------------------------
        print("\nStep 3: read back from db.proposals...", flush=True)
        try:
            read_back = db.proposals.find_one({"_id": inserted_id})
            assert read_back is not None
            _print_check(
                "find_one returned document",
                True,
                f"keys={sorted(read_back.keys())[:6]}...")
        except Exception as e:  # noqa: BLE001
            _print_check("find_one returned document", False, str(e))
            failures.append("step 3: Mongo read")
            return 1

        # ---- Step 4: re-validate via Pydantic --------------------------
        print("\nStep 4: re-validate Mongo doc through Pydantic...", flush=True)
        try:
            read_back.pop("_id", None)  # Pydantic schema doesn't have _id
            reconstructed = Proposal(**read_back)
            same_title = reconstructed.title == proposal.title
            same_arms = len(reconstructed.experiment_arms) == len(proposal.experiment_arms)
            same_status = reconstructed.status == proposal.status
            ok = same_title and same_arms and same_status
            _print_check(
                "round-trip preserves fields",
                ok,
                f"title_match={same_title} arms_match={same_arms} status_match={same_status}")
            if not ok:
                failures.append("step 4: round-trip field mismatch")
        except Exception as e:  # noqa: BLE001
            _print_check("round-trip preserves fields", False, str(e))
            failures.append("step 4: Pydantic re-validation")
            return 1

        # ---- Step 5: pre-rubric checks on the well-formed proposal -----
        print("\nStep 5: pre-rubric checks - well-formed proposal "
              "should pass all 7...", flush=True)
        result = pre_rubric_checks.run_all(proposal)
        if result.all_passed:
            _print_check("all 7 checks pass", True, result.summary())
        else:
            _print_check("all 7 checks pass", False, result.summary())
            for f in result.failed:
                print(f"      - {f.check_id}: {f.reason}", flush=True)
            failures.append("step 5: pre-rubric check on valid proposal")

        # ---- Step 6: mutate into bad shapes ----------------------------
        # We test each check by mutating ONE field and confirming the
        # corresponding check_id fails. Run-all returns all check results
        # so we can see exactly one failed check per mutation.
        print("\nStep 6: pre-rubric checks - bad proposals "
              "should fail the corresponding check...", flush=True)
        for mutation_name, mutator, expected_failed_check in [
            (
                "B: empty hypothesis",
                lambda p: setattr(p, "hypothesis", ""),
                "B",
            ),
            (
                "C: remove base arm",
                lambda p: setattr(p, "experiment_arms",
                                  [a for a in p.experiment_arms if a.name != "base"]),
                "C",
            ),
            (
                "D: cost overrun (10M iters, no budget)",
                lambda p: setattr(p, "num_iterations_per_seed", 10_000_000),
                "D",
            ),
            (
                "E: unknown schema key",
                lambda p: setattr(p.experiment_arms[1],
                                  "experiment_design_fields",
                                  {"nonexistent_field": 42}),
                "E",
            ),
            (
                "F: mentions safety-critical path",
                lambda p: setattr(p, "code_changes_summary",
                                  "edit _emergency_pause_handler"),
                "F",
            ),
        ]:
            mutated = build_synthetic_proposal()
            mutator(mutated)
            # For check D specifically, set spent_so_far high so any
            # overrun is detected even with a generous monthly cap.
            if expected_failed_check == "D":
                res = pre_rubric_checks.run_all(
                    mutated,
                    monthly_budget_usd=250.0,
                    spent_so_far_usd=240.0)
            else:
                res = pre_rubric_checks.run_all(mutated)
            failed_ids = {r.check_id for r in res.failed}
            ok = expected_failed_check in failed_ids
            _print_check(
                f"mutation '{mutation_name}' fails {expected_failed_check}",
                ok,
                f"failed={sorted(failed_ids)}")
            if not ok:
                failures.append(f"step 6: mutation '{mutation_name}'")

        # Special case: check G fires only when a reward design is
        # involved. Add a reward_design_field + remove the secondary
        # metric to trigger it.
        mutated = build_synthetic_proposal()
        mutated.experiment_arms[1].reward_design_fields = {
            "reward_standard": "def reward_standard(...): return 0.0",
        }
        mutated.success_criteria.secondary = ["unrelated metric"]
        res = pre_rubric_checks.run_all(mutated)
        failed_ids = {r.check_id for r in res.failed}
        ok = "G" in failed_ids
        _print_check(
            "mutation 'G: reward design with no invariant secondary' fails G",
            ok,
            f"failed={sorted(failed_ids)}")
        if not ok:
            failures.append("step 6: mutation 'G'")

        # ---- Step 7: cleanup -------------------------------------------
        print("\nStep 7: cleanup - delete test proposal from Mongo...", flush=True)
        try:
            res = db.proposals.delete_one({"_id": inserted_id})
            _print_check(
                "delete_one removed test record",
                res.deleted_count == 1,
                f"deleted_count={res.deleted_count}")
            inserted_id = None
        except Exception as e:  # noqa: BLE001
            _print_check("delete_one removed test record", False, str(e))
            failures.append("step 7: cleanup")

    finally:
        # Safety-net cleanup if something went sideways above.
        if inserted_id is not None:
            try:
                _db().proposals.delete_one({"_id": inserted_id})
                print(f"\n  safety-net cleanup removed {inserted_id}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"\n  WARNING: safety-net cleanup failed: {e}", flush=True)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 64, flush=True)
    if not failures:
        print("RESULT: ALL CHECKS PASSED ✓", flush=True)
        print("=" * 64, flush=True)
        return 0
    else:
        print(f"RESULT: {len(failures)} FAILURE(S)", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        print("=" * 64, flush=True)
        return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
