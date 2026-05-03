from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from environments.courses import donut_course,simple_course

class RobotaxiEnv(py_environment.PyEnvironment):
    def __init__(self, api, course_type='donut'):
        if course_type == 'donut':
            self.course = donut_course.DonutCourse(api, self)
        else:
            self.course = simple_course.SimpleCourse(api, self)
        self._api = api
        self._action_spec = self.course.action_spec
        self._observation_spec = self.course.observation_spec
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._state = self.course.get_empty_state()
        self._episode_ended = False
        self.job_id=""
        self.has_reset = False
        self.pass_through_actions = False
        self.data = {}

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_course_metrics(self):
        """Snapshot the inner course's public metrics as a plain dict.

        Exposed on the env so it can be invoked through
        ``ParallelPyEnvironment.call('get_course_metrics')``, which only
        dispatches methods declared on the env class itself.
        """
        return self.course.get_metrics()

    def configure(self, job_id="", pass_through_actions=False):
        """Apply per-job configuration to this env.

        Exposed as a method (rather than direct attribute writes) so the
        same call site can dispatch through ``ParallelPyEnvironment.call``
        in multi-env training.
        """
        self.job_id = job_id
        self.pass_through_actions = pass_through_actions

    def _reset(self):
        self._episode_ended = False
        #self._api.DoResetBlocking()
        self.course.do_reset_blocking()
        data = self._api.GetCarSceneDataBlocking()
        self.data = data
        data_arr=self.course.scene_data_array(data)
        self._step_costs=[]
        self._position_history=[]
        return ts.restart(np.array(data_arr, dtype=np.float32))
   
    def __has_failed(self, data, data_arr):
        return self.course.has_failed(
            data, data_arr, self._step_costs, self._position_history)
    
    def __has_succeeded(self, data, data_arr):
        return self.course.has_succeeded(data, data_arr)
    
    def _apply_force(self):
        return self._api.DoApplyForceBlocking()

    def _step(self, action):
        m = interp1d([0,1,5,100],[100,10,1,0.5],fill_value="extrapolate")
        n = interp1d([-3,-0.001,1,3],[5,0,-2,-5],fill_value="extrapolate")
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode
            return self.reset()
        data = self._do_action(action)
        self.data = data
        data_arr = self._scene_data_array(data)
        curr_step_cost = abs(data_arr[7]) #abs(data_arr[6])
        self._step_costs.append(curr_step_cost)
        self._position_history.append(
            [self.data["car"]['location_x'], #x position
            self.data["car"]['location_y'], #y position
            self.data["car"]['location_z']
            ]) #z position
        if self.__has_failed(data, data_arr):
            return self.course.reward_failure(
                self.job_id, self._step_costs, data,
                data_arr, self._position_history)
            # reward = -1 * np.mean(self._step_costs) if len(self._step_costs) > 0 and np.mean(self._step_costs) > 0  else -1
            # log_reward(self.job_id, "has failed", float(reward),extra_data=data,step_costs=self._step_costs, position_history=self._position_history)
            # term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), reward=reward)
            # return term_time_step
        if self.__has_succeeded(data, data_arr):
            return self.course.reward_success(
                curr_step_cost, self.job_id, data, data_arr,
                self._step_costs, self._position_history)
        else:
            return self.course.reward_standard(
                data, data_arr, self._step_costs, self.job_id)
    
    def _do_action(self, action):
        if type(action).__name__ == "ndarray":
            action_arr = action.tolist()
        else:
            action_arr = action.numpy().tolist()
        
        if (self.pass_through_actions == True):
            steering_angle=self.data["car"]["dist_from_traj"]
            acceleration = 2 if self.data["car"]["speed"] < 10 else -10
        else:
            acceleration=action_arr[0]
            steering_angle=action_arr[1]
    
        data = self._api.DoApplyForceBlocking(
            acceleration,
            steering_angle)
        self.course.do_action_after(action, data)
        return data

    def _get_empty_state(self):
        return self.course.get_empty_state()
    
    def _scene_data_array(self, scene_data):
        return self.course.scene_data_array(scene_data)