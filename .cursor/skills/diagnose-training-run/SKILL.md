---
name: diagnose-training-run
description: Diagnoses an rl-roboracer training job through the roboracer-console MCP server - whether it ran cleanly, how its models score, and where on the track the car crashes or slows down - and writes a short verdict with next steps. Use when the user asks how a run went, why a job did badly or failed, what the car is weak at, or to check on a training job.
---

# Diagnose a training run

Uses the `roboracer-console` MCP tools. The tools enforce their own rules (eligibility, confirmations); call them and report what they say rather than re-checking.

## Workflow

```
- [ ] 1. Pick the job
- [ ] 2. Did it run cleanly?
- [ ] 3. How good are its models?
- [ ] 4. Where is the car weak?
- [ ] 5. Write the verdict
```

**1. Pick the job.** Use the id the user gave (the last 8 characters are enough). Otherwise `list_jobs` with `job_type: "TRAIN"` and take the running job, else the newest. Say which job you chose.

**2. Did it run cleanly?** `get_job`, then read `status`, `percent_complete`, `training_steps` against `num_iterations`, `eval_error`, `paused_at_step`, and the designs and gym used.
- `FAILED`, or `DONE` well short of its iterations: `tail_trainer_log` with `grep: "error|exception|traceback"`, then without `grep` for context. The log holds the trainer's latest output only, so it explains the current or most recent job; say so if the job you're diagnosing is older.
- `IN_PROGRESS`: a progress report, not a verdict. Everything below is provisional.

**3. How good are its models?** `list_models` with `job_id`. For models with evals, `get_model` gives the pooled score and 95% confidence interval.
- No evals: suggest `eval_models` (10 trials standard) and don't queue them unasked.
- A baseline to compare against (the user's, or the top of `get_leaderboard` for the same `course_type`): `compare_models` with the baseline first. Treat overlapping confidence intervals or a non-significant Welch test as "no difference".

**4. Where is the car weak?**
- `get_difficult_sections` (job_id) ranks gates and sections by crash rate. Rank by the low end of `crash_rate_ci95`; don't name a gate with fewer than about 5 attempts as a problem on its own.
- `get_weakness_map` (job_id) gives spatial hotspots and the text map. Include the map in the answer.
- Look for agreement: a hotspot whose position sits near a hard gate's `position` or `crash_position` is the same problem seen twice and is the strongest finding.
- `get_weakness_at` to drill into a specific spot the user asks about.

Reading the weakness map:
- The start straight always reads slow because episodes begin from rest; ignore slowness there.
- `crash` layer = where crashes happen; `risk` layer = where approaches that end in crashes run. A high-risk cell upstream of a crash cluster means the mistake starts earlier (line choice or speed) than where the car hits.
- `steps_to_clear_vs_median` well above 1 with a low crash rate = slow but safe there (often over-braking); high crash rate with a low step ratio = carrying too much speed.

If `get_difficult_sections` reports that the dashboard has no `/weakness_by_gate` endpoint, carry on with `get_weakness_map` alone and mention that the dashboard needs restarting from a checkout that has it.

**5. Write the verdict.** Lead with one sentence: did it work and how well. Then:

```markdown
**Run health:** status, how far it got, any errors (with the log line).
**Performance:** score ± CI, versus baseline if there is one.
**Where it struggles:** the top 2-3 sections, each with its place on the track, crash rate with CI and attempt count, and whether the spatial map agrees.
**Next steps:** 1-3 concrete actions (see below).
```

## Next steps to suggest

Match the suggestion to the evidence; offer, don't execute:
- Crashed or failed: the fix the log points to, then `update_jobs` with `action: "restart"`.
- No evals yet: `eval_models`.
- Weakness concentrated in a few sections: weakness-targeted starts (`suggest_start_gates`). Note that the trainer can't apply start gates yet.
- Weakness spread evenly, score still rising at the end: more training (a larger `num_iterations`).
- Slow-but-safe sections: a reward design that pays more for progress there (`list_reward_designs` to start from the current one).
