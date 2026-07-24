import os

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from environments.courses import donut_course,simple_course

# Module-level switch for the per-step `ACTION accel=... steer=...` trace
# emitted by `_do_action` below. Default ON (preserves the diagnostic
# visibility added in commit 374d8aa). Set the env var
# ROBOTAXI_LOG_ACTIONS to 0 / false / no / off in the sim-controller
# container to silence:
#
#   docker compose ... exec -e ROBOTAXI_LOG_ACTIONS=0 sim-controller \
#       bash -c 'cd /python_ws/src && python -u robotaxi.py ...'
#
# or persistently in docker-compose.yml's sim-controller `environment:`
# block. Under multi-env training each actor emits one ACTION line per
# env step, so this is a major contributor to robotaxi.out volume
# (~tens of lines per second per actor) once you've already verified
# per-actor routing is working and don't need the per-step trace
# anymore.
_LOG_ACTIONS = os.environ.get("ROBOTAXI_LOG_ACTIONS", "1").lower() not in (
    "0", "false", "no", "off"
)

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
        self._reset_pending = False  # True if Unity was already reset (avoid double reset)
        self._immediate_reset_on_failure = True  # Set False during eval to avoid blocking
        self.job_id=""
        self.has_reset = False
        self.pass_through_actions = False
        # Procedural-track curriculum knobs, forwarded to Unity's
        # TrackGenerator on each reset (see course.do_reset_blocking ->
        # RobotApi.DoReset). Defaults match the trainer/SCHEMA defaults so an
        # unconfigured env reproduces the baseline track.
        self.corner_radius = 10.0
        self.curvature_difficulty = 0.0
        # Per-edge absolute chicane counts (2026-07-18), forwarded alongside
        # corner_radius/curvature_difficulty. Replaces curvature_difficulty
        # as the actual chicane-count driver (see TrackGenerator.cs);
        # curvature_difficulty is retained above for logging/back-compat.
        self.chicanes_north = 0
        self.chicanes_east = 0
        self.chicanes_south = 0
        self.chicanes_west = 0
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

    def get_course_raw_counters(self):
        """Snapshot the inner course's raw counters as a plain dict.

        Same dispatch pattern as ``get_course_metrics`` above, but
        returns the unaveraged numerators + denominators
        (steps_total, num_episodes_total, speeds_total, ...). The
        trainer's run_policy uses snapshot-and-delta arithmetic on
        these to compute per-trial unbiased metrics for the
        leaderboard_scores collection without needing to mutate
        course state between trials.
        """
        return self.course.get_raw_counters()

    def get_timeout_counts(self):
        """Snapshot the RobotApi's asyncio.TimeoutError counters.

        Exposed on the env (rather than only on the RobotApi) so that
        ``ParallelPyEnvironment.call('get_timeout_counts')`` can reach
        it - the multi-env RPC only dispatches methods declared on the
        env class itself. Returns the per-actor counts; the trainer in
        robotaxi.py aggregates across actors before writing to
        TensorBoard.
        """
        return self._api.get_timeout_counts()

    def configure(self, job_id="", pass_through_actions=False,
                  corner_radius=10.0, curvature_difficulty=0.0,
                  chicanes_north=0, chicanes_east=0, chicanes_south=0,
                  chicanes_west=0, env_discount=None):
        """Apply per-job configuration to this env.

        Exposed as a method (rather than direct attribute writes) so the
        same call site can dispatch through ``ParallelPyEnvironment.call``
        in multi-env training.

        ``corner_radius`` / ``curvature_difficulty`` are the procedural-track
        curriculum knobs; they're stored here and read by the course's
        do_reset_blocking() so each Unity reset regenerates the track at the
        requested difficulty. ``chicanes_north/east/south/west`` (2026-07-18)
        are the per-edge ABSOLUTE chicane counts that actually drive chicane
        placement (curvature_difficulty is kept for logging/back-compat
        only - see TrackGenerator.cs).
        """
        self.job_id = job_id
        self.pass_through_actions = pass_through_actions
        if corner_radius is not None:
            self.corner_radius = float(corner_radius)
        if curvature_difficulty is not None:
            self.curvature_difficulty = float(curvature_difficulty)
        self.chicanes_north = int(chicanes_north or 0)
        self.chicanes_east = int(chicanes_east or 0)
        self.chicanes_south = int(chicanes_south or 0)
        self.chicanes_west = int(chicanes_west or 0)
        # Per-step env discount (see DonutCourse.env_discount). None -> keep
        # the course's default; a value overrides it for this job's course.
        if env_discount is not None:
            self.course.env_discount = float(env_discount)

    def set_immediate_reset_on_failure(self, enabled):
        """Enable/disable immediate Unity reset on episode failure.

        Called via ParallelPyEnvironment.call() to toggle behavior between
        training (immediate=True) and eval (immediate=False).
        """
        self._immediate_reset_on_failure = bool(enabled)

    def publish_rollouts(self, payload_json):
        """Forward a policy-rollout-viz payload to Unity via this env's api.

        Exposed as an env method so the trainer can dispatch it through
        ``ParallelPyEnvironment.call('publish_rollouts', payload_json)`` on
        a single actor's subprocess (the one whose Unity client we want to
        render on). The main process samples the policy and builds the JSON;
        this hop just hands it to the subprocess's existing RobotApi channel
        (which is already connected to that actor's ros-server). Best-effort.
        """
        try:
            self._api.PublishRolloutsBlocking(payload_json)
        except Exception as e:  # noqa: BLE001
            print(f"publish_rollouts skipped: {e}", flush=True)

    def install_reward_design(self, name, code):
        """Compile a reward-design module and patch it onto our course.

        Same dispatch shape as ``configure`` so the caller in robotaxi.py
        can fire this through ``ParallelPyEnvironment.call`` to install
        the design into every subprocess env in multi-env training. The
        compile happens INSIDE the subprocess so we never have to pickle
        function objects across the process boundary; only the small
        ``code`` string crosses.

        Raises:
          rl_agent.reward_designs.RewardDesignError on compile / load
          failure. Caller (robotaxi.do_job) catches this and surfaces it
          as job.eval_error so a bad design doesn't crash the trainer.
        """
        # Import lazily inside the method so subprocess envs that never
        # need it don't pay the import cost at construction time.
        #
        # Bare-name sibling import: inside the sim-controller container
        # the trainer runs with CWD=/python_ws/src/ (the bind-mount of
        # host ./rl_agent), so `reward_designs` is a sibling, not a
        # `rl_agent.*` package member. See the matching note in
        # robotaxi.py::do_job for the long version.
        from reward_designs import load_reward_design, install_on_course
        funcs = load_reward_design(name, code)
        return install_on_course(self.course, funcs)

    def _reset(self):
        self._episode_ended = False
        # Skip the Unity reset if it was already triggered immediately on
        # episode end (see donut_course.reward_failure). This avoids a
        # redundant 4-second reset wait when the trainer eventually calls
        # step() after a termination.
        if not self._reset_pending:
            self.course.do_reset_blocking()
        self._reset_pending = False
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

        # One-line per-step action trace. In multi-actor training each
        # worker's stdout is wrapped with [actor-N] (see
        # rl_agent/envs.py::_install_actor_prefix), so the rendered
        # output looks like
        #     [actor-0] ACTION accel=+1.234 steer=-0.567
        #     [actor-1] ACTION accel=-0.890 steer=+0.123
        #     [actor-2] ACTION accel=+0.456 steer=+0.789
        #     [actor-3] ACTION accel=+1.999 steer=-1.000
        # making it trivial to confirm visually that the learner is
        # producing N independently-conditioned actions per step
        # (different cars, different observations, stochastic SAC
        # sampling) rather than broadcasting one action to all actors.
        # `flush=True` belt-and-suspenders against pipe block-buffering
        # if anyone ever runs without `python -u`. Gated on
        # ROBOTAXI_LOG_ACTIONS (see module top) so this can be silenced
        # once the per-actor routing is validated and the trace is just
        # log-volume noise.
        if _LOG_ACTIONS:
            print("ACTION accel={:+.4f} steer={:+.4f}".format(
                acceleration, steering_angle), flush=True)

        data = self._api.DoApplyForceBlocking(
            acceleration,
            steering_angle)
        self.course.do_action_after(action, data)
        return data

    def _get_empty_state(self):
        return self.course.get_empty_state()
    
    def _scene_data_array(self, scene_data):
        return self.course.scene_data_array(scene_data)