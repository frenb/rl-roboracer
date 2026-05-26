# Judge Rubric — RL+BC Research Proposal Evaluation

| | |
|---|---|
| **Version** | 1 |
| **Effective from** | 2026-05-25 |
| **Authored by** | `deep_research_initial` (Claude Opus 4.7, grounded in the citations in §8) |
| **Applies to** | The MadScientist agent's experiment proposals stored in `db.proposals` |
| **Consumer** | `rl_agent/madscientist/judge.py` (Phase 1; not yet shipped) |

This is the source-of-truth rubric the Judge worker uses to score MadScientist proposals before they reach the operator's inbox. The rubric is intentionally **general-purpose for any RL + behavior-cloning + autonomous-driving research proposal** — the axes apply to a Procgen ablation, a CARLA driving paper, or this codebase's robotaxi runs alike. A short appendix in §6 maps each axis to specific fields in our `experiment_designs` schema, `reward_designs` Python signature, and Mongo collections, so the Judge can mechanically cross-check feasibility without re-reading the trainer.

The rubric is **versioned**: each material edit increments the integer in the front-matter, and `judge_rubric_history` records every snapshot (so a 2027-Q2 proposal's review is still interpretable against the rubric that produced it, even if we've rewritten the axes since). The `seed.py` script reads this file and upserts the current version into `db.judge_rubric_history` on every container start.

---

## 1. Philosophy

The Judge does three things, in order:

1. **Filter** — hard-fail any proposal that violates one of the pre-rubric automatic checks in §2 (these are cheap, deterministic, and short-circuit the LLM scoring step).
2. **Score** — assign a 0–5 integer on each of the eight rubric axes in §3, with the level anchors in mind.
3. **Aggregate** — sum the axis scores (max 40) into an overall verdict per §4 (`strong_accept`, `accept`, `weak_accept`, `weak_reject`, `reject`), then emit the structured JSON in §5.

**The Judge is independent, not adversarial.** Its job is to surface concerns to the operator, not to replace them. A `weak_accept` plus a substantive list of `concerns` is more useful than a `strong_accept` that hides the trade-offs. The Judge should err toward including strengths AND concerns even on its strongest verdicts.

**The Judge is conservative on novelty and feasibility.** It is much cheaper for the operator to skip a borderline proposal and have the MadScientist re-pitch later than to chase a half-baked one through implementation and training. When in doubt, lean toward `weak_reject` with concrete suggested revisions.

**The Judge does NOT make the final decision.** Every proposal — regardless of verdict — is forwarded to the operator's email for the actual approve/reject. The Judge's verdict is a signal, not a gate.

---

## 2. Pre-rubric automatic checks

These are deterministic. Any **single** failure triggers `status="rejected"` with `judge_review.overall="reject"` and a `concerns` entry pointing at the failed check. The LLM scoring step is skipped to save tokens.

| # | Check | Why |
|---|---|---|
| A | Every entry in `proposal.source_papers` must resolve to a live arxiv URL (`https://arxiv.org/abs/<id>`) returning 200. | Hallucinated citations are the most common MadScientist failure mode. Verify before scoring. |
| B | `proposal.hypothesis` and `proposal.success_criteria.primary` are both non-empty strings. | Can't score a proposal that doesn't specify what success means. |
| C | `proposal.experiment_arms` contains at least 2 arms, AND exactly one of them has `name="base"`. | No control = no comparison. |
| D | `proposal.n_seeds_per_arm * len(experiment_arms) * proposal.num_iterations_per_seed * 0.5s/iter` (wall-time estimate) fits inside the remaining monthly budget. | Cost containment; protects the $250/mo cap. |
| E | Every `experiment_arms[*].experiment_design_fields` key maps to a real key in `experiment_designs.SCHEMA` (in this repo). Same for `reward_design_fields` — must match the function names `reward_standard` / `reward_success` / `reward_failure`. | Field-name typos would silently no-op in the trainer; better to catch here. |
| F | No proposal modifies safety-critical code paths flagged in §6.4 (the emergency pause handler, the SIGTERM hook, etc.). | The MadScientist's Cursor agent must NEVER edit these. |
| G | If `proposal.experiment_arms` includes any arm with `reward_design_fields`, the proposal also identifies at least one **reward-invariant secondary metric** (avg_goals_per_episode, avg_speed, avg_episode_length, or avg_steering_angle_ratio). | Goodhart protection at the schema level: any reward-design experiment must be checkable against a metric the reward can't directly inflate. |

A proposal that fails A-G is not necessarily a *bad* idea, but it isn't review-ready. The Judge writes the failed check into `judge_review.concerns[0]` so the MadScientist can re-draft if it wants.

---

## 3. Scoring axes

Each axis is scored **0–5**. Use the anchors verbatim — they keep scoring stable across proposals and across Judge runs.

### 3.1 Hypothesis specificity & falsifiability

**Question:** Can this proposal **fail**? Is the success criterion a single concrete statement that the outcome ingester could mechanically check?

| Score | Anchor |
|---|---|
| **5** | Single numeric statement with a directional inequality + explicit metric + arm names. E.g., "`avg_return(exp2) - avg_return(base) ≥ 10%` at train_step ≥ 5000, measured over n_seeds ≥ 3 with bootstrap 95% CI excluding 0." |
| **4** | Single inequality but missing one of {effect size, CI requirement, metric specificity}. |
| **3** | Two or three inequalities ("avg_return AND avg_goals improve"), each individually specific. |
| **2** | One inequality without a quantified effect size ("avg_return is higher in exp2"). |
| **1** | Qualitative ("the agent generalizes better"). |
| **0** | Cannot be falsified ("we show the method is useful"). |

**Why this matters.** Henderson et al. (2018) found that deep RL papers without pre-registered success criteria over-claim 2–3× as often as those with them — because what counts as "better" gets adjusted post hoc after seeing the data. A 5 on this axis means a single fixed line in the sand.

### 3.2 Novelty

**Question:** Does this proposal differ meaningfully from what's already been tried in **this codebase's** `experiment_designs` / `reward_designs` collections **AND** from the most recent ~50 papers in the relevant arxiv categories?

| Score | Anchor |
|---|---|
| **5** | Tests something not yet in either the codebase's design DB nor in any cited paper from the last 2 years. Cite the **gap** explicitly. |
| **4** | Replicates an effect from a 2024+ paper that hasn't been attempted in this codebase. |
| **3** | Combines two known techniques whose combination is novel ("DAPG aux BC × Q-filter, gated by AWAC weighting"). |
| **2** | Replicates a well-known result (>1 year old) as a control sanity check. |
| **1** | Re-runs an existing experiment_design with cosmetic changes (different seeds, no field changes). |
| **0** | Identical to an existing design AND no rationale for re-running. |

A `2` is fine for the first proposals of a new project — establishing baselines IS useful work. The Judge should NOT punish baseline-establishing proposals on novelty if the proposal explicitly says "this is a control replication, not a novel claim."

### 3.3 Significance

**Question:** If the hypothesis is supported, **does it change** what subsequent proposals look like? Does it shift the project's working theory of the problem?

| Score | Anchor |
|---|---|
| **5** | Resolves an open question that bottlenecks several plausible future proposals. The MadScientist's next 2-3 cycles would be substantively different depending on the outcome. |
| **4** | Closes off a hypothesis class (e.g., "demo-protected replay doesn't help under our current reward shape") OR establishes a strong new direction. |
| **3** | Provides incremental evidence for/against a hypothesis already supported by adjacent results. |
| **2** | Confirms a result already established in the literature, with marginal value-add for this project. |
| **1** | Restates a known result. |
| **0** | The answer would not influence any subsequent action. |

### 3.4 Statistical power & baseline rigor

**Question:** Are the seeds, eval episodes, and baseline arm chosen well enough that a real effect would be detectable AND a noise effect would not be mistaken for a real one?

| Score | Anchor |
|---|---|
| **5** | n_seeds_per_arm ≥ 5 per Henderson et al; n_eval_episodes ≥ 10 per arm; same seed list across arms (paired comparison); same env, same hyperparams except the variable under test; explicit bootstrap-CI / IQM plan per Agarwal et al; effect-size estimate provided alongside p-value plan. |
| **4** | n_seeds_per_arm = 3–4 (acceptable for exploratory, per the cookbook), all other criteria met. |
| **3** | n_seeds_per_arm ≥ 3 + control arm + same hyperparams, but no explicit CI/effect-size plan. |
| **2** | n_seeds_per_arm < 3 OR control arm missing OR hyperparams drift between arms. |
| **1** | Single seed per arm. |
| **0** | No control arm at all (zeroes out by check C in §2, but listed here for completeness). |

**Note on seeds.** The Patterson-White cookbook (2023) gives 3-5 seeds as the minimum for "exploration / research", 5-10 for "publication-quality claims." Since MadScientist proposals are research-budget-bounded, the Judge accepts 3 as floor but rewards 5+.

### 3.5 Goodhart resistance

**Question:** Could a model trivially game the primary metric without actually improving driving behavior? If the proposal touches reward design, is the reward shape resistant to specification gaming?

| Score | Anchor |
|---|---|
| **5** | Primary metric is reward-invariant OR is the reward itself with explicit reward-invariant secondary metrics tracked. Reward function (if proposed) has been hand-traced for trivial exploits (camping at a single waypoint, oscillating around the goal, etc.). |
| **4** | Primary metric is reward-based but the proposal explicitly enumerates 1+ ways the agent could game it AND tracks reward-invariant secondaries. |
| **3** | Primary metric is reward-based, secondaries listed, but no explicit gaming analysis. |
| **2** | Primary metric is reward-based, no secondary metrics. |
| **1** | The proposed reward function rewards behaviors that are orthogonal to driving (e.g., minimizing steering for its own sake). |
| **0** | The metric is provably gameable in a few env steps (a known exploit). |

**Why this is its own axis.** Krakovna et al. (2020) catalog dozens of specification-gaming failure modes in RL agents; every reward-shaping experiment is at risk. Forcing the proposal to enumerate gaming routes upfront is cheap insurance.

### 3.6 Paper faithfulness

**Question:** If the proposal cites a paper as its motivating evidence, does the proposed implementation match what the paper actually describes?

| Score | Anchor |
|---|---|
| **5** | Cites specific equations / algorithm boxes from the paper and the proposed `experiment_design_fields` reflect them with correct ranges. Hyperparameters are within the paper's tested range, OR explicitly motivated when deviating. |
| **4** | General reference correct + parameter ranges reasonable; minor missing detail (e.g., a paper-suggested ablation not yet planned). |
| **3** | High-level idea matches the paper but implementation choices are not directly justified by it. |
| **2** | Loose connection ("inspired by"); risk of mis-attribution. |
| **1** | Paper is cited but the proposal's mechanism doesn't appear in the paper. |
| **0** | Citation is for a paper that doesn't exist OR appears to be hallucinated (check A in §2 would catch this, but the Judge should also score 0 here for completeness). |

For proposals that **don't** cite any paper, this axis is scored `N/A` and excluded from the verdict aggregation. Pure-codebase improvements ("tune our SAC's tau") don't need paper backing.

### 3.7 Implementation feasibility

**Question:** Can the changes ship as a single PR that touches `experiment_designs.SCHEMA`, possibly a new `reward_designs.py`-style code blob, possibly small additions to `robotaxi.py::main()`, and **nothing else**?

| Score | Anchor |
|---|---|
| **5** | Pure schema additions to `experiment_designs.SCHEMA` + corresponding kwarg in `main()` (which already exists or is a one-line plumb-through) + a new Reward design (Python function, ≤30 lines) when needed. No changes to SAC internals, no changes to env, no new dependencies. |
| **4** | Schema additions plus a small (≤100 lines) addition to `robotaxi.py::main()` (e.g., interleaved BC step in the training loop). |
| **3** | Requires subclassing or monkey-patching an existing class (e.g., a custom SacAgent subclass for an aux loss). Bounded but non-trivial. |
| **2** | Requires changes in two+ subsystems (e.g., env wrapper + agent + replay). |
| **1** | Requires a new external Python dependency OR changes the tf-agents version. |
| **0** | Requires changes to the trainer-pickup loop, the Mongo schema, the dashboard, or any safety-critical path from §6.4. |

A high score here is **strongly correlated with implementation success rate** of the Cursor SDK orchestrator. The Judge should be brutal on this axis — a brilliant idea that needs a refactor of the training loop is less valuable than a B+ idea that ships in 30 minutes.

### 3.8 Cost & reproducibility

**Question:** Will this experiment run within the monthly budget, AND will its results still be interpretable 6 months from now?

| Score | Anchor |
|---|---|
| **5** | Wall-time estimate ≤ 10% of the remaining monthly budget. All arms reference a versioned experiment_design + reward_design (not inline overrides). Explicit seed list specified. Will stamp git_sha + git_branch on every job and model (the trainer already does this — score just verifies the proposal isn't disabling provenance). |
| **4** | Within budget; one of {versioned design, explicit seeds, git stamping} weakened. |
| **3** | Within budget; uses inline `experiment_design_fields` rather than referencing a saved design (acceptable but reduces re-runnability). |
| **2** | Within budget but ≥ 30% of remaining; OR no explicit seed list. |
| **1** | Borderline over-budget AND no clear way to abort partway through. |
| **0** | Estimated cost exceeds remaining budget (caught by check D in §2, but listed for completeness). |

---

## 4. Overall verdict mapping

Sum the **non-N/A** axis scores. The maximum is `8 × 5 = 40` if all axes apply, `7 × 5 = 35` if §3.6 (Paper faithfulness) is N/A.

Normalize to a 0–1 scale: `normalized = sum_of_scores / max_possible`.

| Normalized score | Verdict |
|---|---|
| ≥ 0.875 | `strong_accept` |
| 0.700–0.874 | `accept` |
| 0.525–0.699 | `weak_accept` |
| 0.350–0.524 | `weak_reject` |
| < 0.350 | `reject` |

**Override conditions** (verdict is forced to `reject` regardless of normalized score):

* Any axis scored `0` (a single zero indicates an unfixable defect — wrong control, gameable metric, hallucinated paper, infeasible implementation, or over-budget).
* Pre-rubric check from §2 failed.

**Override condition** (verdict can be raised at most one notch):

* If §3.5 (Goodhart) and §3.6 (Paper faithfulness) are both 5, OR the proposal's `experiment_arms` use exclusively versioned designs (none inline), the Judge may bump the verdict one notch (e.g., `weak_accept` → `accept`) — these are robust-proposal markers that statistical noise on the other axes is unlikely to invalidate.

---

## 5. Output schema

The Judge must emit exactly this JSON object (no surrounding markdown, no comments). It is parsed by `judge.py` and persisted under `proposal.judge_review`.

```json
{
  "overall": "strong_accept | accept | weak_accept | weak_reject | reject",
  "scores": {
    "hypothesis_specificity": 0,
    "novelty": 0,
    "significance": 0,
    "statistical_power_and_baseline_rigor": 0,
    "goodhart_resistance": 0,
    "paper_faithfulness": 0,
    "implementation_feasibility": 0,
    "cost_and_reproducibility": 0
  },
  "normalized_score": 0.0,
  "axes_skipped": [],
  "pre_rubric_check_failures": [],
  "strengths": ["..."],
  "concerns": ["..."],
  "suggested_revisions": ["..."],
  "rubric_version": 1,
  "judge_model": "claude-opus-4-7",
  "judge_token_count": 0,
  "judge_cost_usd": 0.0,
  "judged_at": "ISO-8601 UTC"
}
```

* `axes_skipped` lists axis keys that received `N/A` (currently only `paper_faithfulness`).
* `pre_rubric_check_failures` is empty for proposals that passed §2 and contains the check letter(s) (e.g., `["A", "G"]`) for proposals that didn't (in which case `overall="reject"` and `scores` keys are all `0`).
* The Judge is encouraged to write **2–4 strengths** and **2–5 concerns**. Empty lists indicate the Judge isn't doing its job; the Judge must list at least one of each unless the verdict is `strong_accept` (in which case `concerns` may be empty).
* `suggested_revisions` is the most actionable output — concrete edits the MadScientist could make to nudge the proposal toward acceptance. Empty for `strong_accept` / `accept`; required for `weak_*` / `reject` verdicts.

---

## 6. Codebase mapping appendix

This is where the general-purpose rubric grounds in this specific project. The Judge consults this section when scoring §3.4, §3.5, §3.7, and §3.8.

### 6.1 Experiment design schema mapping

The `proposal.experiment_arms[*].experiment_design_fields` keys must be subset of the keys defined in `rl_agent/experiment_designs.py::SCHEMA` (excluding `_section_*` headers). Currently the schema has five sections of fields:

| Section | Fields (subset; see SCHEMA for the authoritative list) |
|---|---|
| `_section_rl_loop` | `num_iterations`, `initial_collect_steps`, `collect_steps_per_iteration`, `eval_interval`, `num_eval_episodes`, `log_interval`, `policy_save_interval` |
| `_section_bc` | `bc_pretrain_steps` |
| `_section_replay` | `replay_buffer_capacity`, `demo_prefill_count`, `demo_min_keep`, `demo_sample_ratio` |
| `_section_optimizer` | `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate`, `gamma`, `target_update_tau`, `target_update_period`, `reward_scale_factor`, `batch_size` |
| `_section_network` | `actor_fc_layer_params_x`, `actor_fc_layer_params_y`, `critic_joint_fc_layer_params_x`, `critic_joint_fc_layer_params_y` |

A proposal that wants to add a NEW field (e.g., `aux_bc_loss_weight` for a DAPG-style experiment) must include both:
* an entry in the proposal's `code_changes_summary` for adding to SCHEMA,
* a corresponding `kwarg` plumbed through `main()`'s signature.

The Judge scores §3.7 lower if the proposal asks for a new field but doesn't describe both pieces.

### 6.2 Reward design structure

`rl_agent/reward_designs.py` defines three pluggable Python functions a custom reward design can override:

```python
def reward_standard(data, data_arr, step_costs, *, defaults, course):
    """Returns the per-step reward when the agent hasn't yet succeeded/failed."""

def reward_success(data, data_arr, step_costs, position_history, *,
                   defaults, course):
    """Returns the reward at episode success (e.g., goal reached)."""

def reward_failure(data, data_arr, step_costs, position_history, *,
                   defaults, course):
    """Returns the reward at episode failure (e.g., crash)."""
```

A proposal's `reward_design_fields` may override any subset. The Judge scores §3.5 lower if the proposal:
* Overrides `reward_standard` with constants that don't depend on `data` (signals a misunderstanding of the API).
* Doesn't track any of the reward-invariant metrics (`avg_goals_per_episode`, `avg_speed`, `avg_episode_length`, `avg_steering_angle_ratio`) as a secondary metric.

### 6.3 Mongo collections the Judge reads

The Judge has read-only access to:

* `db.proposals` — the proposal under review + history (for §3.2 novelty checks).
* `db.experiment_designs` — existing named designs (for §3.2 novelty checks, §3.8 versioning checks).
* `db.reward_designs` — existing named reward functions.
* `db.models` — past training results (for §3.2 / §3.3 — "has this hypothesis already been answered?").
* `db.leaderboard_scores` — per-trial arrays for past evals (for §3.4 — "is the historical variance enough that 3 seeds would be underpowered?").

### 6.4 Safety-critical code paths (do NOT touch)

Pre-rubric check F fails if the proposal's `code_changes_summary` mentions edits to any of these:

* `rl_agent/robotaxi.py::_emergency_pause_handler` and `_emergency_state`
* `rl_agent/robotaxi.py::_get_job_lifecycle_state` and the SIGTERM/SIGINT registration
* `rl_agent/robotaxi.py::_restore_paused_active_dir` / `_detect_resume_for_train_job` / `_has_learner_checkpoint_for_job`
* `rl_agent/robotaxi.py::move_all_jobs_data` (the resume-aware archival path)
* `rl_agent/api.py::RpcClient.__init__` (gRPC keepalive options — moving these can re-trigger the `too_many_pings` GOAWAY bug)
* `docker/ros_server/ROS/src/virtual_endpoint/src/virtual_endpoint/virtual.py` (server-side gRPC config)
* Anything under `dashboard/` (the dashboard is human-maintained for now)

A proposal that needs any of these as scaffolding to test a hypothesis is **valid** but is too large for the MadScientist's autonomous flow. The Judge rejects with a `concerns` entry suggesting the operator implement the scaffolding manually first.

---

## 7. Sources

The axes and anchors above were grounded in (in rough order of influence):

1. **Henderson, Islam, Bachman, Pineau, Precup, Meger. "Deep Reinforcement Learning That Matters."** AAAI 2018. arxiv:1709.06560. The foundational paper on RL reproducibility. Established that 3-5 seeds is the floor for any defensible RL claim, that single-seed results should never be reported as representative, and that significance metrics + standardized baselines matter more than absolute SOTA gains.
2. **Agarwal, Schwarzer, Castro, Courville, Bellemare. "Deep Reinforcement Learning at the Edge of the Statistical Precipice."** NeurIPS 2021 Outstanding Paper. arxiv:2108.13264. Brought interval estimates, bootstrap CIs, interquartile mean, and performance profiles into the RL evaluation mainstream. The `rliable` library that accompanies the paper is what shapes axis §3.4's CI-plan anchor.
3. **Patterson, Cahdoke, Bowling, White. "Empirical Design in Reinforcement Learning."** 2023. arxiv:2304.01315 ("The Cookbook"). The most thorough modern walk-through of seed selection, hyperparameter sweeps, confidence intervals (bootstrap vs Student-t), and the difference between online-view and policy-optimization evaluation. The 3-5 / 5-10 / 10-20 seed tiers in axis §3.4 come from here.
4. **NeurIPS Reviewer Guidelines (2024 & 2026).** The three-pillar structure (soundness / contribution / presentation) and the use of separate sub-ratings plus an overall verdict score informed §4's mapping. The reproducibility checklist behind NeurIPS submissions is the implicit template for axis §3.8.
5. **Krakovna, Uesato, Mikulik, Rahtz, Everitt, Kumar, Kenton, Leike, Legg. "Specification gaming examples in AI."** DeepMind blog post + database, 2020. Catalogs ~60 RL specification-gaming failures. Justifies giving Goodhart-resistance its own axis (§3.5) rather than folding it into "soundness."
6. **NeurIPS Paper Checklist.** Required as of 2022. Items on "experimental statistical significance," "ablation studies," "reproducibility details," and "negative results" map directly to axes §3.4 and §3.8.
7. **Sutton, Barto. "Reinforcement Learning: An Introduction" (2nd ed.).** §13.6 and §16.* on benchmark design and the distinction between learning curves vs final-performance evaluation. Background for axis §3.1.

When the rubric is next revised, the new sources go here too — chronologically appended.

---

## 8. Versioning policy

* Each material edit to this file bumps the integer in the front-matter `Version` row.
* The MadScientist agent's `seed.py` reads this file on every container start; if the file's content hash differs from `db.judge_rubric_history.find_one(sort=[("version", -1)]).rubric_markdown`, a new `JudgeRubricVersion` document is inserted with `version = max_existing + 1`, `effective_from = now`, and `authored_by = <whoever made the edit, defaulted to "operator_manual">`.
* The Judge worker stamps `judge_review.rubric_version` on every review with the version it applied. Old reviews remain interpretable.
* **Material** edits include: adding/removing an axis, changing an anchor's score threshold, changing the verdict-mapping formula, changing pre-rubric checks A-G.
* **Non-material** edits (typos, prose clarifications, link updates) do not bump version.

---

*End of Judge Rubric v1.*
