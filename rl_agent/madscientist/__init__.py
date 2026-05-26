"""MadScientist - autonomous experiment-proposal + implementation agent.

Three workers, all in this package, each backed by the `proposals` MongoDB
collection (see schemas.py):

  * researcher.py - long-running loop that reads new RL/BC/autonomous-
    driving papers + the current codebase + accumulated training results,
    and drafts experiment proposals. Writes proposals with status
    `pending_judge`. (Phase 1.)

  * judge.py - triggered on proposals with status `pending_judge`, applies
    the rubric in madscientist/JUDGE_RUBRIC.md and produces a structured
    review. Advances status to `pending_user`. (Phase 1.)

  * orchestrator.py - on user approval, spawns a Cursor SDK agent in a
    fresh worktree under /worktrees/proposal-<oid>/ which authors the
    code changes, opens a PR, and queues TRAIN jobs stamped with the
    proposal's _id. (Phase 1.)

Plus two passive utilities:

  * outcome_ingester.py - watches db.jobs for TRAIN jobs linked to
    proposals; when all are DONE, computes per-arm statistics + delta
    vs the hypothesis and writes `proposal.results`.

  * email_bridge.py - outbound proposal notifications (SMTP) + inbound
    reply parsing (Gmail API push subscription). Maps user "approve"
    /"reject"/etc to a proposal.decision write.

For Phase 0 (THIS commit), all of the above are stubs - the package
exists, the Mongo schema is locked in, the Docker container builds,
and the dashboard tab renders empty - but no worker actually runs
until MADSCIENTIST_ENABLED=true is set in the environment AND Phase 1
ships the worker bodies.
"""
