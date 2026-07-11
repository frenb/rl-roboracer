from abc import ABC, abstractmethod
import numpy as np
import math
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

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

    def check_if_moving(self, arr):
        """Helper method to check if robot is moving"""
        last_position = len(arr)-1
        if len(arr) < 6:
            return True
        for i in reversed(range(last_position-5, last_position)):
            dist = math.dist(arr[last_position], arr[i])
            if dist >= 0.0001:
                return True
        return False 
    
    def debug_print(self, text):
        debug_print_enabled = True
        if debug_print_enabled:
            print(text, flush=True)