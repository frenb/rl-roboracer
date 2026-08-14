from abc import ABC, abstractmethod
import os
import numpy as np
import math
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

# Stuck-detection tuning (see BaseCourse.check_if_moving). Env-overridable so
# these can be dialed without a code change / trainer rebuild:
#   ROBOTAXI_STUCK_WINDOW    - number of trailing per-step positions to inspect
#   ROBOTAXI_STUCK_RADIUS_M  - min net travel (metres) over that window to count
#                              as "moving"; below it the car is treated as wedged
STUCK_WINDOW = int(os.environ.get("ROBOTAXI_STUCK_WINDOW", "200"))
STUCK_RADIUS_M = float(os.environ.get("ROBOTAXI_STUCK_RADIUS_M", "0.5"))

class BaseCourse(ABC):
    def __init__(self, api, env):
        self.env = env
        self._api = api
    
    @abstractmethod
    def get_empty_state(self):
        """Returns an empty state array for initialization"""
        pass
        
    @abstractmethod
    def has_failed(self, data, data_arr, step_costs, position_history):
        """Determines if the current episode has failed"""
        pass
    
    @abstractmethod
    def has_succeeded(self, data, data_arr):
        """Determines if the current episode has succeeded"""
        pass
    
    @abstractmethod
    def scene_data_array(self, scene_data):
        """Converts scene data to observation array"""
        pass
    
    # ------------------------------------------------------------------
    # Reward methods come in two layers:
    #
    #   * Public ``reward_standard`` / ``reward_success`` /
    #     ``reward_failure`` (abstract here, implemented by concrete
    #     courses). They own state mutation, logging, and wrapping the
    #     scalar reward into the right tf-agents ``TimeStep`` (transition
    #     vs termination).
    #
    #   * Scalar ``_compute_standard_reward`` /
    #     ``_compute_success_reward`` / ``_compute_failure_reward`` -
    #     pure-math float-returning methods (default to 0 here). The
    #     public methods call these to get the reward value. This split
    #     gives the reward-design system in ``rl_agent/reward_designs``
    #     a clean monkey-patch surface: replacing just the scalar method
    #     swaps the reward formula while leaving every side effect
    #     (course-state updates, logging, episode termination) intact.
    #
    # Subclasses MUST override the public methods (they're abstract),
    # SHOULD override the scalar methods if they want a non-zero reward,
    # and MUST keep the scalar methods side-effect-free so they're safe
    # to call from user code via ``RewardDesign.defaults.*``.
    # ------------------------------------------------------------------

    @abstractmethod
    def reward_standard(self, data, data_arr, step_costs, job_id):
        """Calculates standard step reward"""
        pass

    @abstractmethod
    def reward_success(self, curr_step_cost, job_id, data, data_arr, step_costs, position_history):
        """Calculates reward for successful completion"""
        pass

    @abstractmethod
    def reward_failure(self, job_id, step_costs, data, data_arr, position_history):
        """Calculates reward for failure"""
        pass

    # Pure-math scalar reward methods. Default to 0; subclasses override.
    # Reward designs replace these via runtime monkey-patch (see
    # rl_agent/reward_designs.install_on_course). Keep these
    # side-effect-free.
    def _compute_standard_reward(self, data, data_arr, step_costs):
        """Scalar reward for a standard (non-terminal) step."""
        return 0.0

    def _compute_success_reward(self, data, data_arr, step_costs, position_history):
        """Scalar reward when the episode terminates in success."""
        return 0.0

    def _compute_failure_reward(self, data, data_arr, step_costs, position_history):
        """Scalar reward when the episode terminates in failure."""
        return 0.0

    # Names of the public-facing course metrics surfaced to TensorBoard via
    # robotaxi.main(). Concrete courses (DonutCourse) own these as instance
    # attributes; courses that don't define them all (SimpleCourse) report 0.
    # Names starting with "max_" are aggregated across actors with max();
    # all others are aggregated with mean(). See aggregate_course_metrics().
    METRIC_KEYS = (
        'avg_goals_per_episode',
        'avg_goals_per_episode_last_30',
        'max_goals_per_episode',
        'max_goals_per_episode_last_30',
        'avg_steering_angle_ratio',
        'avg_steering_angle_ratio_last_30',
        'max_speed',
        'max_speed_last_30',
        'avg_speed',
        'avg_speed_last_30',
        'max_accel',
        'max_accel_last_30',
        'avg_accel',
        'avg_accel_last_30',
        # Crash-rate and episode-length metrics.
        # crashes_per_1k_steps: episodes (crashes) per 1000 env steps.
        #   Higher = more frequent crashes. Useful for spotting policy
        #   degradation (rate rises) and gym wedges (drops to ~0 when
        #   the gym stops responding and no episodes complete).
        # avg_steps_per_episode: mean env steps between crashes.
        #   Longer = car survives longer. Complement to goals/ep.
        'crashes_per_1k_steps',
        'avg_steps_per_episode',
        # Per-location traversal/crash counters (added 2026-07-19). Raw
        # cumulative counts, not rates - across multiple parallel actors
        # these get MEAN-aggregated (see read_course_metrics in
        # robotaxi.py), same as every other non-'max_' key here, which
        # dilutes the absolute total but preserves the RATIO between the
        # 5 counters (all actors run the same course logic, so the mean
        # divides each by the same actor count) - that ratio, not the
        # absolute total, is the signal this was added to track. See
        # track_zones.py for the position -> zone classifier.
        'easy_corner_traversals',
        'hard_corner_traversals',
        'crashes_easy_corner',
        'crashes_hard_corner',
        'crashes_straight',
        # Crash-only total (2026-07-20) - see donut_course.py's
        # crashes_per_1k_steps comment for why this had to be split out
        # from num_episodes_total once GOALS_PER_EPISODE_CAP gave episodes
        # a second, non-crash way to end.
        'crashes_total',
    )

    def get_metrics(self):
        """Snapshot the public-facing course metrics as a plain dict.

        Designed to be RPC-friendly so it can be called via
        ``ParallelPyEnvironment.call('get_course_metrics')`` and have the
        result aggregated across actors in the main process.
        """
        return {k: float(getattr(self, k, 0)) for k in self.METRIC_KEYS}

    # Names of the raw counters that ``get_raw_counters()`` returns.
    # These are the unaveraged sums + episode counts the course
    # accumulates over its lifetime. The Analysis tab's reward-design-
    # invariant comparison machinery (see run_policy in robotaxi.py)
    # snapshots these before/after each EVAL trial and computes per-
    # trial averages via deltas - more accurate than re-deriving
    # per-trial means from running cumulative averages.
    #
    # Concrete courses MAY define a subset; missing counters default to
    # 0 via getattr below. SimpleCourse currently doesn't track any of
    # these, so it reports zeros and the Analysis tab degrades to just
    # the avg_return metric for that course.
    RAW_COUNTER_KEYS = (
        'steps_total',                # denominator for avg_speed / avg_steering_angle_ratio
        'num_episodes_total',         # denominator for avg_goals_per_episode
        'speeds_total',               # numerator for avg_speed
        'accel_total',                # numerator for avg_accel (commanded force)
        'goals_per_episode_total',    # numerator for avg_goals_per_episode
        'steering_angle_ratio_total', # numerator for avg_steering_angle_ratio
    )

    def get_raw_counters(self):
        """Snapshot the cumulative raw counters as a plain dict.

        Sibling of ``get_metrics()``: ``get_metrics()`` returns the
        running averages, this returns the underlying numerators +
        denominators. Designed for snapshot-and-delta arithmetic so a
        caller can compute the per-window average over an arbitrary
        sub-interval (e.g., one EVAL trial) by reading this before
        and after.

        Missing counters default to 0 - the courses that don't track
        a particular counter report zero deltas, which the consumer
        treats as "data unavailable for this metric on this course"
        rather than "metric was zero".
        """
        return {k: float(getattr(self, k, 0)) for k in self.RAW_COUNTER_KEYS}

    # Lifetime-cumulative course fields that should survive a pause/resume so
    # the course's TensorBoard curves (avg/max goals-per-episode, avg/max
    # speed + accel, avg steering ratio, crashes_per_1k_steps,
    # avg_steps_per_episode, per-location traversal/crash counts) continue from
    # their pre-pause values instead of restarting at 0. These are exactly the
    # fields update_stats() derives the cumulative (non-last_30) metrics from;
    # the rolling *_last_30 windows and their backing arrays are intentionally
    # NOT restored (they re-stabilize within ~30 episodes). The trainer splits
    # the SUM-type totals evenly across actors before restore and passes the
    # aggregated value for the mean/max-type fields - see
    # robotaxi._seed_resume_counters.
    RESTORABLE_CUMULATIVE_KEYS = (
        'steps_total',
        'num_episodes_total',
        'speeds_total',
        'accel_total',
        'goals_per_episode_total',
        'steering_angle_ratio_total',
        'crashes_total',
        'easy_corner_traversals',
        'hard_corner_traversals',
        'crashes_easy_corner',
        'crashes_hard_corner',
        'crashes_straight',
        'max_speed',
        'max_accel',
        'max_goals_per_episode',
    )

    def restore_cumulative_counters(self, stats):
        """Re-seed lifetime cumulative counters after a pause/resume.

        Only whitelisted keys (RESTORABLE_CUMULATIVE_KEYS) present BOTH in
        ``stats`` and as an existing attribute are applied, so a course that
        doesn't track a given field (e.g. SimpleCourse) silently skips it.
        Best-effort: never raises into the env RPC path - a failed restore
        just means that actor's cumulative metrics start from 0 as before.
        """
        try:
            for k in self.RESTORABLE_CUMULATIVE_KEYS:
                if isinstance(stats, dict) and k in stats and hasattr(self, k):
                    setattr(self, k, stats[k])
        except Exception as e:  # noqa: BLE001
            self.debug_print(f"restore_cumulative_counters failed: {e}")

    def recent_goals_per_episode(self, n):
        """Return the goal counts of the most recent ``n`` completed episodes.

        Reads the tail of ``goals_per_episode_arr`` (one entry appended per
        episode in ``reset_after_episode``). Used by the trainer's curriculum
        gate to isolate a single eval cycle's per-episode goal counts (so it
        can, e.g., average the top-K episodes) rather than only the running
        aggregates ``get_metrics()`` exposes. Courses that don't track the
        array (SimpleCourse) return an empty list.
        """
        arr = getattr(self, 'goals_per_episode_arr', None)
        if not arr:
            return []
        n = int(n)
        return [float(x) for x in (arr[-n:] if n > 0 else arr)]

    def check_if_moving(self, arr):
        """Helper method to check if robot is moving.

        Window widened 6 -> 200 (2026-07-19) to give a legitimately slow car
        (e.g. easing through a chicane per collect_expert_demos' braking) a
        multi-second runway to prove it's still making progress before this
        gives up on it as wedged.

        NET-DISPLACEMENT test (2026-08-09): the previous logic returned
        "moving" the moment the CURRENT position differed from ANY of the last
        WINDOW-1 samples by >= 0.0001 m (0.1 mm). That effectively required a
        car to be *perfectly frozen* to be flagged stuck - but a car
        high-centered / wedged against geometry with the policy still applying
        throttle (spinning wheels) and sweeping the steering micro-oscillates
        FAR more than 0.1 mm, so it always cleared that bar and was never
        reset. It just burned steps until the 10 000-step cap
        (has_too_many_steps), dumping thousands of near-zero-reward wedged
        transitions into the buffer (observed on the w-course hairpin neck).

        Instead we measure how far the car has travelled from its CURRENT spot
        across the whole window: a wedge stays inside a small ball (jitter /
        steer sweep don't move the chassis anywhere), while a slow-but-
        progressing car escapes STUCK_RADIUS_M. This is immune to
        steering/wheel-spin jitter. Both the window and the radius are
        env-tunable (ROBOTAXI_STUCK_WINDOW / ROBOTAXI_STUCK_RADIUS_M).
        """
        window = STUCK_WINDOW
        if len(arr) < window:
            return True
        recent = arr[-window:]
        current = recent[-1]
        # Farthest the car has been from where it is now, across the window.
        max_dist = max(math.dist(current, p) for p in recent)
        return max_dist >= STUCK_RADIUS_M
    
    def debug_print(self, text):
        debug_print_enabled = True
        if debug_print_enabled:
            print(text, flush=True)