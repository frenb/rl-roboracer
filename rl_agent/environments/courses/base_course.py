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
    )

    def get_metrics(self):
        """Snapshot the public-facing course metrics as a plain dict.

        Designed to be RPC-friendly so it can be called via
        ``ParallelPyEnvironment.call('get_course_metrics')`` and have the
        result aggregated across actors in the main process.
        """
        return {k: float(getattr(self, k, 0)) for k in self.METRIC_KEYS}

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