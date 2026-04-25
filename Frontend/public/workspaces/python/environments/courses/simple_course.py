import numpy as np
from scipy.interpolate import interp1d
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .base_course import BaseCourse
from .utils.logging import log_blob, log_reward

class SimpleCourse (BaseCourse):
    def __init__(self, api, env):
        self.env = env
        self._api = api
        self.action_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.float32, 
                minimum=[0,-1], 
                maximum=[10,1], name='action')
        self.observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), dtype=np.float32,
            minimum=[0, -25, -10000, -10000, -10000,-100,0,0,0],
            maximum=[1, 100, 10000, 10000, 10000,100,1000,1000,360],
            name='observation')
    
    def get_empty_state(self):
        emptyState = [0,0,0,0,0,0,0,0,0]
        return emptyState
        
    def has_failed(self, data_arr, step_costs, position_history):
        has_fallen = data_arr[3] < -0.5
        int_rotation_z = round(data_arr[8])
        has_too_many_steps = len(step_costs) > 150
        is_stuck = not super().check_if_moving(arr=position_history)
        has_flipped = int_rotation_z == 90 or int_rotation_z == 180 or int_rotation_z == 270
        return has_fallen or has_flipped or is_stuck or has_too_many_steps
    
    def has_succeeded(self, data_arr):
        has_succeeded = data_arr[0] == 1
        return has_succeeded
    
    def scene_data_array(self, scene_data):
        #print("in scene_data_array")
        #print(scene_data)
        # float64 speed
        # float64 location_x
        # float64 location_y
        # float64 location_z
        # float64 cost
        # float64 dist_from_traj
        # float64 dist_from_goal
        # bool has_reached_goal
        # bool has_crashed
        arr = [
            1 if scene_data["car"]['has_reached_goal'] else 0,
            scene_data["car"]['speed'],
            scene_data["car"]['location_x'],
            scene_data["car"]['location_y'],
            scene_data["car"]['location_z'],
            scene_data["car"]["cost"],
            scene_data["car"]["dist_from_traj"],
            scene_data["car"]["dist_from_goal"],
            scene_data["car"]["rotation_z"],
        ]
        return np.array(arr, dtype=np.float32)
    
    def reward_standard(self, data, data_arr, step_costs, job_id):
        self.env._episode_ended = False
        last_step_cost = 0 if len(step_costs) < 2 else step_costs[len(step_costs)-2]
        curr_step_cost = step_costs[len(step_costs)-1]
        diff = abs(curr_step_cost) - abs(last_step_cost)
        if curr_step_cost == 0:
            reward = 1
        elif diff < 0:
            reward = 1
        else:
            reward = -1
        log_reward(self.env.job_id, "did not fail", float(reward), diff=float(diff), extra_data=data)
        return ts.transition(np.array(data_arr, dtype=np.float32), reward=reward, discount=0.90)
    
    def reward_success(self, curr_step_cost, job_id, data, data_arr, step_costs, position_history):
        self.env._episode_ended = True
        m = interp1d([0,1,5,100],[100,10,1,0.5],fill_value="extrapolate")
        reward = float(m(curr_step_cost)) 
        log_reward(self.env.job_id, "has succeeded", float(reward),extra_data=data, step_costs=step_costs, position_history=position_history)
        term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), reward=reward)
        return term_time_step
    
    def reward_failure(self, job_id, step_costs, data, data_arr, position_history):
        self.env._episode_ended = True
        reward = -1 * np.mean(step_costs) if len(step_costs) > 0 and np.mean(step_costs) > 0  else -1
        log_reward(self.env.job_id, "has failed", float(reward),extra_data=data, step_costs=step_costs, position_history=position_history)
        term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), reward=reward)
        return term_time_step
    
    def get_num_obstacles(self):
        return 0
    
    def do_reset_blocking(self):
        num_obstacles = self.get_num_obstacles()
        self._api.DoResetBlocking(num_obstacles)