"""Experiment designs - the structured-config sibling of reward_designs.

A reward design captures the *what does the agent want* part of an
experiment (the reward function). An experiment design captures the
*how does the agent learn* part: BC pretraining steps, replay buffer
capacity, optimizer learning rates, network sizes, etc. Selecting one
of each on the New-job form fully characterises a training run.

This module is the source of truth for which fields are honoured by
the trainer (``rl_agent/robotaxi.py``'s ``main()``) and at what
defaults. Adding a new field requires:

  1. A new entry in SCHEMA below (name, type, default, min/max, doc,
     paper_ref, kwarg).
  2. ``main()`` already accepts the corresponding kwarg (Tier 1 is
     all-pre-existing), or you plumb it through.
  3. Optional: update DEFAULT_DESIGN_FIELDS if you want the canonical
     "Default" experiment design to expose the new field with the
     trainer's current hardcoded value.

The schema is exposed to the future research-planning agent via the
dashboard's ``/get_experiment_design_schema`` endpoint, so structure
+ self-documentation matter. Each field carries:

  type:        "int" | "float" | "bool" | "enum" | "list[int]" - drives
               UI input type + server-side validation.
  default:     The trainer's current built-in value (what main() does
               if you pass nothing).
  min/max:     Soft validation bounds. None = no bound on that side.
  doc:         One-line human description (also used as the dashboard
               form-field tooltip).
  paper_ref:   Optional arxiv id when the knob ties to a specific
               published technique. Tier 1 (SAC hyperparams) leaves
               this empty; later tiers (aux BC, AWAC) populate.
  kwarg:       The exact name of main()'s kwarg. Lets the trainer
               overlay the experiment_design doc onto main()'s
               signature without a hand-maintained mapping table.

Module is bare-name-imported as a sibling of robotaxi.py - see the
matching note in rl_agent/reward_designs.py.
"""

import datetime
import time


# Field schema. Order is the order rendered in the dashboard form (so
# logically-grouped fields stay together). Sections are demarcated
# with leading underscored keys for the form UI to render headers
# (the trainer overlay ignores anything whose key starts with "_").
SCHEMA = {
    "_section_rl_loop": "Reinforcement learning loop",
    "num_iterations": {
        "type": "int",
        "default": 50000,
        "min": 1, "max": 10_000_000,
        "doc": "Total SAC training iterations after BC pretrain. Each iter = one collect step + one gradient update.",
        "paper_ref": None,
        "kwarg": "num_iterations_val",
    },
    "env_discount": {
        "type": "float",
        "default": 0.90,
        "min": 0.0, "max": 1.0,
        "doc": "Per-step discount the env returns on non-terminal transitions; compounds with gamma (effective discount = gamma * env_discount). Legacy default 0.90 yields a short ~9-step horizon at gamma=0.99; set 1.0 to let gamma alone govern the horizon (better for speed / lap objectives).",
        "paper_ref": None,
        "kwarg": "env_discount_val",
    },
    "initial_collect_steps": {
        "type": "int",
        "default": 500,
        "min": 0, "max": 100_000,
        "doc": "Pre-RL random-policy collection steps to seed the replay buffer with diverse experience.",
        "paper_ref": None,
        "kwarg": "initial_collect_steps_val",
    },
    "collect_steps_per_iteration": {
        "type": "int",
        "default": 1,
        "min": 1, "max": 100,
        "doc": "Env steps collected per training iteration (between gradient updates).",
        "paper_ref": None,
        "kwarg": "collect_steps_per_iteration_val",
    },
    "eval_interval": {
        "type": "int",
        "default": 5000,
        "min": 1, "max": 100_000,
        "doc": "How often (in train_steps) to pause training and run an in-loop eval cycle.",
        "paper_ref": None,
        "kwarg": "eval_interval_val",
    },
    "num_eval_episodes": {
        "type": "int",
        "default": 10,
        "min": 1, "max": 500,
        "doc": "Eval episodes per in-loop eval cycle.",
        "paper_ref": None,
        "kwarg": "num_eval_episodes_val",
    },
    "log_interval": {
        "type": "int",
        "default": 5000,
        "min": 100, "max": 100_000,
        "doc": "TensorBoard scalar write cadence (in train_steps).",
        "paper_ref": None,
        "kwarg": "log_interval_val",
    },
    "policy_save_interval": {
        "type": "int",
        "default": 50,
        "min": 1, "max": 10_000,
        "doc": "How often (in train_steps) the PolicySavedModelTrigger writes a checkpoint.",
        "paper_ref": None,
        "kwarg": "policy_save_interval_val",
    },

    "_section_bc": "Behavior cloning pretrain",
    "bc_pretrain_steps": {
        "type": "int",
        "default": 5000,
        "min": 0, "max": 1_000_000,
        "doc": "BC gradient steps run on the actor before SAC starts. Set 0 to skip and run pure SAC.",
        "paper_ref": None,
        "kwarg": "bc_pretrain_steps_val",
    },

    "_section_replay": "Replay buffer",
    "replay_buffer_capacity": {
        "type": "int",
        "default": 75000,
        "min": 1000, "max": 10_000_000,
        "doc": "Max samples held in the online Reverb table (RL collection). Over capacity, FIFO eviction.",
        "paper_ref": None,
        "kwarg": "replay_buffer_capacity_val",
    },
    "batch_size": {
        "type": "int",
        "default": 256,
        "min": 1, "max": 16_384,
        "doc": "SAC gradient-update batch size, also used by the BC pretrain phase.",
        "paper_ref": None,
        "kwarg": "batch_size_val",
    },
    "demo_prefill_count": {
        "type": "int",
        "default": 50000,
        "min": 0, "max": 10_000_000,
        "doc": "Expert-demonstration steps to prefill the buffer with at job start. 0 = no demo prefill (pure SAC from random init).",
        # DDPGfD / DQfD prefill demos into the same replay buffer the
        # RL collector writes to.
        "paper_ref": "1707.08817",
        "kwarg": "demo_prefill_count_val",
    },
    "demo_min_keep": {
        "type": "int",
        "default": 0,
        "min": 0, "max": 10_000_000,
        "doc": "Demo samples PROTECTED from FIFO eviction. 0 = single-table mode (demos pre-fill the online buffer and get FIFO-displaced by RL data over time, current default). >0 = two-table mode where this many demo samples live in a separate Reverb table that never gets new writes, so they stay forever.",
        # DQfD: "We use a separate demonstration buffer that we never
        # over-write". Section 3.
        "paper_ref": "1704.03732",
        "kwarg": "demo_min_keep_val",
    },
    "demo_sample_ratio": {
        "type": "float",
        "default": 0.0,
        "min": 0.0, "max": 1.0,
        "doc": "Two-table mode only (demo_min_keep > 0): fraction of each training batch drawn from the demo table vs the online table. 0.0 = pure online sampling (demos kept but never sampled - acts like demo_min_keep=0). 1.0 = pure demo sampling (online experience kept but never sampled, mostly useful for ablations). Typical 0.1-0.3 for DDPGfD-style demo over-sampling.",
        "paper_ref": "1707.08817",
        "kwarg": "demo_sample_ratio_val",
    },

    "_section_optimizer": "SAC optimizer",
    "actor_learning_rate": {
        "type": "float",
        "default": 3e-5,
        "min": 1e-7, "max": 1.0,
        "doc": "Adam learning rate for the actor network.",
        "paper_ref": None,
        "kwarg": "actor_learning_rate_val",
    },
    "critic_learning_rate": {
        "type": "float",
        "default": 3e-5,
        "min": 1e-7, "max": 1.0,
        "doc": "Adam learning rate for the critic (twin Q-network).",
        "paper_ref": None,
        "kwarg": "critic_learning_rate_val",
    },
    "alpha_learning_rate": {
        "type": "float",
        "default": 3e-5,
        "min": 1e-7, "max": 1.0,
        "doc": "Adam learning rate for the temperature parameter (entropy coefficient).",
        "paper_ref": None,
        "kwarg": "alpha_learning_rate_val",
    },
    "target_update_tau": {
        "type": "float",
        "default": 0.005,
        "min": 0.0, "max": 1.0,
        "doc": "Polyak averaging factor for the target critic. Typical SAC value 0.005.",
        "paper_ref": None,
        "kwarg": "target_update_tau_val",
    },
    "target_update_period": {
        "type": "int",
        "default": 1,
        "min": 1, "max": 10_000,
        "doc": "Update the target critic every N train_steps (Polyak averaging cadence).",
        "paper_ref": None,
        "kwarg": "target_update_period_val",
    },
    "gamma": {
        "type": "float",
        "default": 0.99,
        "min": 0.0, "max": 1.0,
        "doc": "Discount factor for future rewards in the Bellman target.",
        "paper_ref": None,
        "kwarg": "gamma_val",
    },
    "reward_scale_factor": {
        "type": "float",
        "default": 1.0,
        "min": 0.0, "max": 1000.0,
        "doc": "Multiplier applied to environment rewards before they enter the Q-target. SAC is sensitive to this.",
        "paper_ref": None,
        "kwarg": "reward_scale_factor_val",
    },

    "_section_arch": "Network architecture",
    "actor_fc_layers_x": {
        "type": "int",
        "default": 512,
        "min": 1, "max": 8192,
        "doc": "First-layer width of the actor MLP.",
        "paper_ref": None,
        "kwarg": "actor_fc_layer_params_x",
    },
    "actor_fc_layers_y": {
        "type": "int",
        "default": 512,
        "min": 1, "max": 8192,
        "doc": "Second-layer width of the actor MLP.",
        "paper_ref": None,
        "kwarg": "actor_fc_layer_params_y",
    },
    "critic_fc_layers_x": {
        "type": "int",
        "default": 512,
        "min": 1, "max": 8192,
        "doc": "First-layer width of the critic joint MLP (after obs+action concatenation).",
        "paper_ref": None,
        "kwarg": "critic_joint_fc_layer_params_x",
    },
    "critic_fc_layers_y": {
        "type": "int",
        "default": 512,
        "min": 1, "max": 8192,
        "doc": "Second-layer width of the critic joint MLP.",
        "paper_ref": None,
        "kwarg": "critic_joint_fc_layer_params_y",
    },

    "_section_env": "Track / environment (curriculum)",
    "corner_radius": {
        "type": "float",
        "default": 10.0,
        "min": 2.0, "max": 12.0,
        "doc": (
            "[Curriculum / track geometry] Centreline turn radius (m) of the "
            "procedurally-generated track corners. Smaller = tighter, harder "
            "turns; larger = gentler. This is the primary curriculum lever: "
            "schedule a sequence of arms/jobs with DECREASING corner_radius to "
            "progressively harden the track. Applied live in the simulator: "
            "the trainer forwards it on every episode reset and Unity's "
            "TrackGenerator rebuilds the track at this radius."
        ),
        "paper_ref": None,
        "kwarg": "corner_radius_val",
    },
    "curvature_difficulty": {
        "type": "float",
        "default": 0.0,
        "min": 0.0, "max": 1.0,
        "doc": (
            "[Curriculum / track geometry] Chicane density on the track's long "
            "edges, 0..1. 0 = plain rectangle (four gentle corners); 1 = many "
            "tight chicanes. A second difficulty axis that pairs with "
            "corner_radius. Applied live in the simulator: forwarded on every "
            "episode reset and used by Unity's TrackGenerator to rebuild the "
            "track."
        ),
        "paper_ref": None,
        "kwarg": "curvature_difficulty_val",
    },
}


# Subset of SCHEMA keys that are real fields (drops "_section_..." UI
# headers). Iteration helpers below use this so adding sections to
# SCHEMA never accidentally treats them as data fields.
def field_keys():
    return [k for k in SCHEMA.keys() if not k.startswith("_")]


def get_schema_for_endpoint():
    """JSON-serialisable view of SCHEMA for /get_experiment_design_schema.

    Preserves order via a list of {key, ...} entries (so the dashboard
    + the future research-planning agent see the same field ordering
    the form does).
    """
    out = []
    for key, val in SCHEMA.items():
        if key.startswith("_section_"):
            out.append({
                "kind": "section",
                "label": val,
            })
            continue
        out.append({
            "kind": "field",
            "name": key,
            "type": val["type"],
            "default": val["default"],
            "min": val.get("min"),
            "max": val.get("max"),
            "doc": val.get("doc", ""),
            "paper_ref": val.get("paper_ref"),
            "kwarg": val["kwarg"],
        })
    return out


def apply_to_main_kwargs(design_doc, base_kwargs):
    """Overlay an experiment_design doc onto kwargs destined for main().

    ``base_kwargs`` is whatever do_job has already built up from
    legacy job-doc fields (e.g. num_iterations from job["num_iterations"]).
    Fields explicitly present on the experiment_design doc override
    the base; fields absent or None on the design leave the base
    (or main()'s own default) untouched.

    Returns a *new* dict; doesn't mutate ``base_kwargs``.

    Soft-clamps values to the SCHEMA's min/max so a typo'd design
    document can't push a learning rate to 1e30 or a layer width to
    1 million. Out-of-range writes a warning to stdout but proceeds.
    """
    merged = dict(base_kwargs or {})
    if not design_doc:
        return merged
    for key in field_keys():
        if key not in design_doc:
            continue
        raw = design_doc[key]
        if raw is None:
            continue
        # Type coercion + soft validation.
        spec = SCHEMA[key]
        try:
            if spec["type"] == "int":
                val = int(raw)
            elif spec["type"] == "float":
                val = float(raw)
            elif spec["type"] == "bool":
                val = bool(raw)
            else:
                val = raw
        except (TypeError, ValueError) as e:
            print(
                f"experiment_designs: field {key!r} coercion failed "
                f"({raw!r}, expected {spec['type']}): {e}. Skipping.",
                flush=True)
            continue
        # Soft clamp.
        if isinstance(val, (int, float)):
            lo = spec.get("min")
            hi = spec.get("max")
            clamped = val
            if lo is not None and clamped < lo:
                clamped = lo
            if hi is not None and clamped > hi:
                clamped = hi
            if clamped != val:
                print(
                    f"experiment_designs: field {key!r} value {val} "
                    f"out of [{lo}, {hi}]; clamping to {clamped}.",
                    flush=True)
                val = clamped
        merged[spec["kwarg"]] = val
    return merged


# Canonical seed design. Inserted into the experiment_designs MongoDB
# collection at sim-controller startup so the New-job form always has
# at least one option and the future research-planning agent has a
# stable "control" reference to compare new designs against. Unique
# string _id makes the upsert idempotent across container restarts.
DEFAULT_DESIGN_ID = "experiment-default"
DEFAULT_DESIGN_NAME = "Default"
DEFAULT_DESIGN_DESCRIPTION = (
    "Canonical defaults baked into rl_agent/robotaxi.py::main(). "
    "Every other experiment design should be compared against this one "
    "so changes are interpretable. Don't edit - clone for variants."
)


def default_design_fields():
    """Snapshot the trainer's current built-in defaults as a dict
    matching the SCHEMA's field keys.

    Used by _seed_canonical_experiment_design() in robotaxi.py to
    write the Default design into Mongo, and by callers who want
    the same baseline for comparison.
    """
    return {k: SCHEMA[k]["default"] for k in field_keys()}
