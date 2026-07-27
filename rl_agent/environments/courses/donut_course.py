import numpy as np
from scipy.interpolate import interp1d
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .base_course import BaseCourse
from .utils.logging import log_blob, log_reward
from . import track_zones

class DonutCourse (BaseCourse):
    # Second (non-crash) episode-termination criterion, added 2026-07-20:
    # cap how many goals a single episode can accumulate before it ends
    # "successfully" (see reward_success below), instead of only ever
    # ending via has_failed (crash/stuck/flip/too-many-steps). Without this,
    # a well-driving policy's episodes run indefinitely (up to the
    # has_too_many_steps step cap) and pack in 100+ goals each, which makes
    # every episode-shaped stat (avg_steps_per_episode, goals_per_episode,
    # DEMO curriculum's goal_budget accounting, etc.) far lumpier / less
    # frequent than intended - bounding episode length by goal count keeps
    # episode boundaries (and the env.reset()/course-config refresh that
    # comes with them) happening at a steady, predictable cadence.
    GOALS_PER_EPISODE_CAP = 30

    def __init__(self, api, env):
        self.env = env
        self._api = api
        # Per-step discount returned on every non-terminal transition. This
        # compounds with the SAC agent's gamma, so the EFFECTIVE discount is
        # gamma * env_discount. Default 0.90 (legacy; with gamma=0.99 that's
        # an effective ~0.89 / ~9-step horizon). Override per-experiment via
        # the experiment_designs `env_discount` knob (set 1.0 to let gamma
        # alone govern the horizon - important for speed/lap objectives).
        self.env_discount = 0.90
        self.last_goal_reached=""
        self.speeds_arr=[]
        self.accel_arr=[]
        self.steering_angle_ratio_arr=[]
        self.goals_per_episode_arr=[]
        self.steps_per_goal_arr=[]
        self.num_obstacles_arr=[]
        self.avg_return_arr=[]
        self.max_speed=0
        self.avg_speed=0
        self.avg_speed_last_30=0
        self.max_speed_last_30=0
        # Commanded acceleration/force (action[0], range [-1, 1] as of
        # 2026-07-19 - was [0.1, 2.0], drive-only) stats. Surfaced to
        # TensorBoard so we can track whether the policy is actually pushing
        # the throttle toward the ceiling or hugging the floor (the
        # slow-driving symptom).
        self.max_accel=0
        self.avg_accel=0
        self.avg_accel_last_30=0
        self.max_accel_last_30=0
        self.max_steps_per_goal=0
        self.avg_steps_per_goal=0
        self.avg_steps_per_goal_last_30=0
        self.max_steps_per_goal_last_30=0
        self.max_goals_per_episode=0
        self.avg_goals_per_episode=0
        self.max_goals_per_episode_last_30=0
        self.avg_goals_per_episode_last_30=0
        self.avg_steering_angle_ratio=0
        self.avg_steering_angle_ratio_last_30=0
        self.steps_since_last_goal=0
        self.goals_reached=0
        self.max_avg_return=0
        self.steps_total=0
        self.num_episodes_total=0
        self.speeds_total=0
        self.accel_total=0
        self.goals_per_episode_total=0
        self.steering_angle_ratio_total=0
        # Crash-rate and episode-length metrics (populated in update_stats)
        self.crashes_per_1k_steps=0.0
        self.avg_steps_per_episode=0.0
        # Per-location traversal/crash counters (added 2026-07-19 to answer
        # "how well do we do on easy corners (the 4 base loop turns) vs hard
        # corners (chicanes) vs straightaways" - see track_zones.py for the
        # position -> zone classifier these feed off of. Cumulative for the
        # life of this course instance (never reset per-episode), same
        # convention as goals_per_episode_total etc.
        self.easy_corner_traversals=0
        self.hard_corner_traversals=0
        self.crashes_easy_corner=0
        self.crashes_hard_corner=0
        self.crashes_straight=0
        # Authoritative crash-only counter (2026-07-20), distinct from
        # num_episodes_total: once episodes can ALSO end via the
        # GOALS_PER_EPISODE_CAP success path below, num_episodes_total
        # counts BOTH crash- and goal-cap-terminated episodes, so it can no
        # longer stand in for "number of crashes" the way
        # crashes_per_1k_steps (see update_stats) relies on. Incremented
        # only in reward_failure - unlike crashes_easy/hard/straight above,
        # this is never skipped for an 'unknown' zone classification, so
        # it's always the true total.
        self.crashes_total=0
        # shape=(2,) == [acceleration, steering_angle]
        # acceleration range widened 0.1..2.0 -> -1..1 (2026-07-19): the old
        # range was drive-only (minimum 0.1 => the motor could never apply
        # negative/reverse torque), so the ONLY way to shed speed was to
        # coast down toward the 0.1 floor and rely on rolling
        # friction/drag - far too weak to bleed off overshoot speed (e.g.
        # 5.6 m/s) in the ~10m of straight remaining before a corner, which
        # is exactly what was causing "the car drives straight into the
        # wall" crashes right after long straights (confirmed via
        # video+[steer-diag]/[crash-pos] cross-referencing, see
        # scripts/analyze_video.py). Negative acceleration now maps to
        # negative motorTorque in CarController.cs, i.e. genuine active
        # braking (and, past zero speed, reverse) instead of coast-only.
        #
        # acceleration minimum -1 -> -0.01 (2026-07-22): full -1 active
        # braking proved too strong, so the acceleration floor is pulled
        # back to a near-coast -0.01 (barely-negative torque) while the
        # +1 drive ceiling and the full [-1, 1] steering range are kept.
        #
        # acceleration minimum -0.01 -> 0.05, maximum kept at 1.0
        # (2026-07-24): demo-distribution analysis showed the [-0.01, 1]
        # range let the policy idle at ~0 throttle (a ~0.5 m/s "crawl"), so
        # the floor is raised to a POSITIVE 0.05 - the policy can no longer
        # coast/brake below a gentle forward torque and is forced to keep
        # driving. This restores the always-forward property the original
        # 2026-04-26 spec had (that spec was accel [0.1, 2.0]); the new range
        # [0.05, 1.0] uses a slightly lower floor (0.05 vs 0.1) and half the
        # top-end torque (1.0 vs 2.0). Steering range [-1, 1] unchanged. Keep
        # minimum[0] in sync with _DEMO_MIN_ACCEL in robotaxi.py.
        self.action_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.float32, 
                minimum=[0.05,-1], 
                maximum=[1, 1], name='action')
        self.observation_spec = array_spec.BoundedArraySpec(
            shape=(32,), dtype=np.float32,
            minimum=[
                -1, #scene_data["car"]["dist_from_traj"] angle to next goal
                -10, #scene_data["car"]["speed"] magnitude of car velocity
                -1, #scene_data["car"]["goal_2"] angle from velocity to car
                0, #scene_data["car"]["left"]
                0, #scene_data["car"]["forward_left"]
                0, #scene_data["car"]["forward_left_left"]
                0, #scene_data["car"]["n_27_50"]
                0, #scene_data["car"]["n_25_00"]
                0, #scene_data["car"]["n_22_50"]
                0, #scene_data["car"]["n_20_00"]
                0, #scene_data["car"]["n_17_50"]
                0, #scene_data["car"]["n_15_00"]
                0, #scene_data["car"]["n_12_50"]
                0, #scene_data["car"]["n_10_00
                0, #scene_data["car"]["n_07_50"],
                0, #scene_data["car"]["n_05_00"],
                0, #scene_data["car"]["n_02_50"]
                0, #scene_data["car"]["forward"], # float64 forward
                0, #scene_data["car"]["p_02_50"],# float64 p_02_50
                0, #scene_data["car"]["p_05_00"],# float64 p_05_00
                0, #scene_data["car"]["p_07_50"],# float64 p_07_50
                0, #scene_data["car"]["p_10_00"],# float64 p_10_00
                0, #scene_data["car"]["p_12_50"],# float64 p_12_50
                0, #scene_data["car"]["p_15_00"],# float64 p_15_00
                0, #scene_data["car"]["p_17_50"],# float64 p_17_50
                0, #scene_data["car"]["p_20_00"],# float64 p_20_00
                0, #scene_data["car"]["p_22_50"],# float64 p_22_50
                0, #scene_data["car"]["p_25_00"],# float64 p_25_00
                0, #scene_data["car"]["p_27_50"],# float64 p_27_50
                0, #scene_data["car"]["forward_right_right"],# float64 forward_right_right
                0, #scene_data["car"]["forward_right"],
                0, #scene_data["car"]["right"]
                ],
            maximum=[
                1, #scene_data["car"]["dist_from_traj"]
                10, #scene_data["car"]["speed"] magnitude of car velocity
                1, #scene_data["car"]["goal_2"] angle from velocity to car
                1000, #scene_data["car"]["left"]
                1000, #scene_data["car"]["forward_left"]
                1000, #scene_data["car"]["forward_left_left"]
                1000, #scene_data["car"]["n_27_50"]
                1000, #scene_data["car"]["n_25_00"]
                1000, #scene_data["car"]["n_22_50"]
                1000, #scene_data["car"]["n_20_00"]
                1000, #scene_data["car"]["n_17_50"]
                1000, #scene_data["car"]["n_15_00"]
                1000, #scene_data["car"]["n_12_50"]
                1000, #scene_data["car"]["n_10_00
                1000, #scene_data["car"]["n_07_50"],
                1000, #scene_data["car"]["n_05_00"],
                1000, #scene_data["car"]["n_02_50"]
                1000, #scene_data["car"]["forward"], # float64 forward
                1000, #scene_data["car"]["p_02_50"],# float64 p_02_50
                1000, #scene_data["car"]["p_05_00"],# float64 p_05_00
                1000, #scene_data["car"]["p_07_50"],# float64 p_07_50
                1000, #scene_data["car"]["p_10_00"],# float64 p_10_00
                1000, #scene_data["car"]["p_12_50"],# float64 p_12_50
                1000, #scene_data["car"]["p_15_00"],# float64 p_15_00
                1000, #scene_data["car"]["p_17_50"],# float64 p_17_50
                1000, #scene_data["car"]["p_20_00"],# float64 p_20_00
                1000, #scene_data["car"]["p_22_50"],# float64 p_22_50
                1000, #scene_data["car"]["p_25_00"],# float64 p_25_00
                1000, #scene_data["car"]["p_27_50"],# float64 p_27_50
                1000, #scene_data["car"]["forward_right_right"],# float64 forward_right_right
                1000, #scene_data["car"]["forward_right"],
                1000, #scene_data["car"]["right"]
                ],
            name='observation')
    
    def get_empty_state(self):
        emptyState = [
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0
        ]
        return emptyState
        
    def has_failed(self, data, data_arr, step_costs, position_history):
        has_fallen = data["car"]["location_y"] < -0.5
        int_rotation_z = round(data["car"]["rotation_z"])
        has_crashed = data["car"]["has_crashed"]
        has_too_many_steps = len(step_costs) > 10000
        is_too_slow_arr = np.array(self.speeds_arr)
        is_too_slow_avg = np.average(is_too_slow_arr[-100:]) if len(is_too_slow_arr) > 100 else 1
        is_too_slow = False #True if is_too_slow_avg < 1 else False
        log_blob({"type": "is_too_slow", "is_too_slow":is_too_slow, "is_too_slow_avg":is_too_slow_avg})
        is_stuck = not super().check_if_moving(arr=position_history)
        has_flipped = int_rotation_z == 90 or int_rotation_z == 180 or int_rotation_z == 270
        log_blob ({"type":"has_failed","has_fallen":str(has_fallen),"has_flipped": str(has_flipped),
            "has_too_many_steps": str(has_too_many_steps), "has_crashed": str(has_crashed), 
            "is_too_slow": str(is_too_slow)})
        return has_fallen or has_flipped or is_stuck or has_crashed or has_too_many_steps # or is_too_slow
    
    def _classify_car_zone(self, data):
        """Best-effort track_zones.classify_zone() lookup for the car's
        CURRENT position, using this course's live chicane configuration
        (set per curriculum stage via env.configure - see do_reset_blocking
        below for the same getattr pattern). Never raises - returns
        'unknown' on any lookup failure so a stats hiccup can't break
        reward/termination handling.
        """
        try:
            car = data.get("car", {})
            return track_zones.classify_zone(
                car.get("location_x"), car.get("location_z"),
                chicanes_north=getattr(self.env, "chicanes_north", 0),
                chicanes_east=getattr(self.env, "chicanes_east", 0),
                chicanes_south=getattr(self.env, "chicanes_south", 0),
                chicanes_west=getattr(self.env, "chicanes_west", 0))
        except Exception:  # noqa: BLE001 - stats must never break the env
            return "unknown"

    def has_succeeded(self, data, data_arr):
        
        #print(f"has reached goal {self._api.has_reached_goal} current goal {data['car']['current_goal']}")
        has_succeeded = \
            self._api.has_reached_goal \
            and data["car"]["current_goal"] != "" \
            and data["car"]["last_goal_reached"] != self.last_goal_reached
        if has_succeeded:
            self.last_goal_reached = data["car"]["last_goal_reached"]
            self._api.has_reached_goal = False
            log_blob({"type":"has_succeeded","has_succeeded":str(has_succeeded),
                "data[car][last_goal_reached]": data["car"]["last_goal_reached"],
                "data[car][current_goal]": data["car"]["current_goal"],
                "self.last_goal_reached": self.last_goal_reached,
                "goals reached": str(self.goals_reached)})
            print(f"has_succeeded {self._api.has_reached_goal} current goal {data['car']['current_goal']} last goal reached {data['car']['last_goal_reached']}")
        return has_succeeded
    
    
    def scene_data_array(self, scene_data):
        # float64 speed
        # float64 location_x
        # float64 location_y
        # float64 location_z
        # float64 cost
        # float64 dist_from_traj
        # float64 dist_from_goal
        # float64 rotation_z
        # float64 angular_velocity
        # float64 acceleration
        # float64 left
        # float64 forward_left
        # float64 forward_left_left
        # float64 n_27_50
        # float64 n_25_00
        # float64 n_22_50
        # float64 n_20_00
        # float64 n_17_50
        # float64 n_15_00
        # float64 n_12_50
        # float64 n_10_00
        # float64 n_07_50
        # float64 n_05_00
        # float64 n_02_50
        # float64 forward
        # float64 p_02_50
        # float64 p_05_00
        # float64 p_07_50
        # float64 p_10_00
        # float64 p_12_50
        # float64 p_15_00
        # float64 p_17_50
        # float64 p_20_00
        # float64 p_22_50
        # float64 p_25_00
        # float64 p_27_50
        # float64 forward_right_right
        # float64 forward_right
        # float64 right
        # float64 goal_1
        # float64 goal_2
        # float64 goal_3
        # float64 goal_4
        # bool has_reached_goal
        # bool has_crashed
        arr = [
            scene_data["car"]["dist_from_traj"], # angle from car to next goal
            scene_data["car"]["speed"], 
            scene_data["car"]["goal_2"], # angle from velocity to car direction 
            scene_data["car"]["left"],
            scene_data["car"]["forward_left"],
            scene_data["car"]["forward_left_left"], # float64 forward_left_left
            scene_data["car"]["n_27_50"],# float64 n_27_50
            scene_data["car"]["n_25_00"],# float64 n_25_00
            scene_data["car"]["n_22_50"],# float64 n_22_50
            scene_data["car"]["n_20_00"],# float64 n_20_00
            scene_data["car"]["n_17_50"],# float64 n_17_50
            scene_data["car"]["n_15_00"],# float64 n_15_00
            scene_data["car"]["n_12_50"],# float64 n_12_50
            scene_data["car"]["n_10_00"],# float64 n_10_00
            scene_data["car"]["n_07_50"],# float64 n_07_50
            scene_data["car"]["n_05_00"],# float64 n_05_00
            scene_data["car"]["n_02_50"], # float64 n_02_50
            scene_data["car"]["forward"], # float64 forward
            scene_data["car"]["p_02_50"],# float64 p_02_50
            scene_data["car"]["p_05_00"],# float64 p_05_00
            scene_data["car"]["p_07_50"],# float64 p_07_50
            scene_data["car"]["p_10_00"],# float64 p_10_00
            scene_data["car"]["p_12_50"],# float64 p_12_50
            scene_data["car"]["p_15_00"],# float64 p_15_00
            scene_data["car"]["p_17_50"],# float64 p_17_50
            scene_data["car"]["p_20_00"],# float64 p_20_00
            scene_data["car"]["p_22_50"],# float64 p_22_50
            scene_data["car"]["p_25_00"],# float64 p_25_00
            scene_data["car"]["p_27_50"],# float64 p_27_50
            scene_data["car"]["forward_right_right"],# float64 forward_right_right
            scene_data["car"]["forward_right"],
            scene_data["car"]["right"],
        ]
        return np.array(arr, dtype=np.float32)
    
    # Public reward methods. Each one:
    #   1. Mutates course state (episode flags, counters)
    #   2. Calls the matching scalar `_compute_*_reward` for the
    #      reward value (overridable by a reward design, see
    #      rl_agent/reward_designs.py)
    #   3. Logs the reward + context
    #   4. Wraps the scalar into the appropriate tf-agents TimeStep
    # The split exists so reward designs only need to provide the
    # scalar math; every side effect above is owned by the course.

    def reward_standard(self, data, data_arr, step_costs, job_id):
        self.env._episode_ended = False
        last_step_cost = 0 if len(step_costs) < 2 else step_costs[len(step_costs)-2]
        curr_step_cost = step_costs[len(step_costs)-1]
        diff = abs(curr_step_cost) - abs(last_step_cost)
        self.steps_since_last_goal += 1
        reward = float(self._compute_standard_reward(data, data_arr, step_costs))
        log_reward(self.env.job_id, "did not fail", reward, diff=float(diff), extra_data=data, stat_array=data_arr)
        return ts.transition(np.array(data_arr, dtype=np.float32), reward=reward, discount=self.env_discount)

    def reward_success(self, curr_step_cost, job_id, data, data_arr, step_costs, position_history):
        self.steps_since_last_goal += 1
        self.steps_per_goal_arr.append(self.steps_since_last_goal)
        # Per-location traversal counters (see _classify_car_zone) - only
        # corners count as a "traversal" here (straightaway goal-reaches
        # aren't the interesting signal the easy-vs-hard-corner tracking is
        # for), so 'straight'/'unknown' zones are intentionally not tallied.
        _zone = self._classify_car_zone(data)
        if _zone == "easy_corner":
            self.easy_corner_traversals += 1
        elif _zone == "hard_corner":
            self.hard_corner_traversals += 1
        # Compute reward AFTER incrementing steps_since_last_goal so a
        # reward design that reads it via the `course` proxy sees the
        # same value the legacy formula did (which uses the post-
        # increment count).
        reward = float(self._compute_success_reward(data, data_arr, step_costs, position_history))
        print("goal reached: " + str(reward))
        log_reward(self.env.job_id, "has succeeded", reward, extra_data=data, step_costs=step_costs, position_history=position_history, stat_array=data_arr)
        self.reset_after_goal_reached()
        # Second episode-termination criterion (2026-07-20, see
        # GOALS_PER_EPISODE_CAP docstring): once this episode has banked
        # enough goals, end it "successfully" here rather than only ever
        # via has_failed. reset_after_goal_reached() above already bumped
        # self.goals_reached, so this check sees the post-increment count.
        if self.goals_reached >= self.GOALS_PER_EPISODE_CAP:
            self.env._episode_ended = True
            self.reset_after_episode()
            # Same reasoning as reward_failure's immediate-reset block: get
            # Unity resetting right away during TRAINING instead of sitting
            # idle for a learner update; skip during EVAL (steps quickly,
            # a blocking reset here just slows it down).
            if getattr(self.env, '_immediate_reset_on_failure', True):
                try:
                    api = self._api
                    timeouts_before = getattr(api, 'reset_timeouts', 0)
                    self.do_reset_blocking()
                    timeouts_after = getattr(api, 'reset_timeouts', 0)
                    if timeouts_after == timeouts_before:
                        self.env._reset_pending = True
                except Exception as e:
                    print(f"[donut_course] immediate reset after goal cap: {e}", flush=True)
            return ts.termination(np.array(data_arr, dtype=np.float32), reward=reward)
        self.env._episode_ended = False
        return ts.transition(np.array(data_arr, dtype=np.float32), reward=reward, discount=self.env_discount)

    def reward_failure(self, job_id, step_costs, data, data_arr, position_history):
        self.env._episode_ended = True
        reward = float(self._compute_failure_reward(data, data_arr, step_costs, position_history))
        log_reward(self.env.job_id, "has failed - reward", reward, extra_data=data, step_costs=step_costs, position_history=position_history, stat_array=data_arr)
        # Per-location crash counters (see _classify_car_zone / track_zones.py).
        # 'unknown' (lookup failure, e.g. missing location data) is
        # deliberately not tallied into any bucket rather than guessed as
        # 'straight', so these three always sum to <= total crash count.
        _zone = self._classify_car_zone(data)
        if _zone == "easy_corner":
            self.crashes_easy_corner += 1
        elif _zone == "hard_corner":
            self.crashes_hard_corner += 1
        elif _zone == "straight":
            self.crashes_straight += 1
        self.crashes_total += 1
        self.reset_after_episode()
        # Trigger Unity reset IMMEDIATELY during TRAINING so the car doesn't
        # sit dead for 10+ seconds waiting for the learner. Skip during EVAL
        # because eval steps quickly and the blocking reset just slows it down.
        if getattr(self.env, '_immediate_reset_on_failure', True):
            try:
                api = self._api
                timeouts_before = getattr(api, 'reset_timeouts', 0)
                self.do_reset_blocking()
                timeouts_after = getattr(api, 'reset_timeouts', 0)
                if timeouts_after == timeouts_before:
                    # Reset succeeded without timeout - skip redundant reset later
                    self.env._reset_pending = True
            except Exception as e:
                print(f"[donut_course] immediate reset after failure: {e}", flush=True)
        term_time_step = ts.termination(np.array(data_arr, dtype=np.float32), reward=reward)
        return term_time_step

    # Scalar reward formulas. These are the methods a reward design
    # monkey-patches when it wants to change the reward function.
    # Side-effect-free by contract - state mutation lives in the public
    # methods above. Both arguments and the return type are kept
    # deliberately simple (scalars + plain dicts/arrays) so user code
    # never has to know about tf-agents TimeStep types.

    def _compute_standard_reward(self, data, data_arr, step_costs):
        """DonutCourse default: zero per-step reward."""
        return 0

    def _compute_success_reward(self, data, data_arr, step_costs, position_history):
        """DonutCourse default: scaled bonus for reaching a goal quickly."""
        return (max(1, 100 - self.steps_since_last_goal) / 100)

    def _compute_failure_reward(self, data, data_arr, step_costs, position_history):
        """DonutCourse default: zero penalty on failure."""
        return 0

    def get_num_obstacles(self):
        return 0
    
    def reset_after_episode(self):
        self.goals_per_episode_arr.append(self.goals_reached) #append number of goals reached for finished episode to array
        self.goals_per_episode_total+=self.goals_reached
        self.num_episodes_total+=1
        self.steps_since_last_goal=0
        self.goals_reached=0
        self.last_goal_reached=""
    
    def reset_after_goal_reached(self):
        self.steps_since_last_goal=0
        self.goals_reached += 1
    
    def do_action_after(self, action, data):
        num_obstacles = self.get_num_obstacles()
        self.update_stats()
        # Guard against dist_from_traj == 0 (car perfectly aligned to goal)
        # which would produce inf/nan that then cascade through numpy stats.
        dist = data["car"]["dist_from_traj"]
        if dist != 0:
            steering_angle_ratio = action[1] / dist
            self.steering_angle_ratio_arr.append(steering_angle_ratio)
            self.steering_angle_ratio_total += steering_angle_ratio
        self.speeds_arr.append(data["car"]["speed"])
        self.speeds_total += data["car"]["speed"]
        # Track the commanded acceleration/force (action[0]). pass_through
        # mode drives a hand-coded accel, so only record the policy's own
        # command when the agent is actually in control.
        if not getattr(self.env, "pass_through_actions", False):
            accel = float(action[0])
            self.accel_arr.append(accel)
            self.accel_total += accel

    def update_stats(self):
        #truncate arrays to last 100 elements
        self.steps_per_goal_arr=self.steps_per_goal_arr[-100:]
        self.avg_return_arr=self.avg_return_arr[-100:]
        self.speeds_arr=self.speeds_arr[-100:]
        self.accel_arr=self.accel_arr[-100:]
        self.steering_angle_ratio_arr=self.steering_angle_ratio_arr[-100:]
        self.goals_per_episode_arr=self.goals_per_episode_arr[-100:]
        self.num_obstacles_arr=self.num_obstacles_arr[-100:]
        # increment total steps
        self.steps_total+=1
        # get number of obstacles in scene, normally is 0
        self.num_obstacles = self.get_num_obstacles()
        # speed stats
        self.max_speed=0 if len(self.speeds_arr) == 0 else max(np.max(self.speeds_arr), self.max_speed)
        self.avg_speed=self.speeds_total / self.steps_total
        self.max_speed_last_30=0 if len(self.speeds_arr) <30 else np.max(self.speeds_arr[-30:])
        self.avg_speed_last_30=np.average(self.speeds_arr[-30:]) if len(self.speeds_arr) > 0 else 0.0
        # commanded acceleration/force stats (action[0])
        self.max_accel=0 if len(self.accel_arr) == 0 else max(np.max(self.accel_arr), self.max_accel)
        self.avg_accel=self.accel_total / self.steps_total
        self.max_accel_last_30=0 if len(self.accel_arr) <30 else np.max(self.accel_arr[-30:])
        self.avg_accel_last_30=np.average(self.accel_arr[-30:]) if len(self.accel_arr) > 0 else 0.0
        # steps per goal stats
        self.max_steps_per_goal=0
        self.avg_steps_per_goal=0
        self.avg_steps_per_goal_last_30=0
        self.max_steps_per_goal_last_30=0
        # goals per episode stats
        self.max_goals_per_episode=0 if len(self.goals_per_episode_arr) == 0 else max(np.max(self.goals_per_episode_arr), self.max_goals_per_episode)
        self.avg_goals_per_episode=self.goals_per_episode_total / max(1,self.num_episodes_total)
        self.max_goals_per_episode_last_30=0 if len(self.goals_per_episode_arr) <30 else np.max(self.goals_per_episode_arr[-30:])
        self.avg_goals_per_episode_last_30=np.average(self.goals_per_episode_arr[-30:]) if len(self.goals_per_episode_arr) > 0 else 0.0
        # steering angle ratio stats, used to determine if car is driving off track
        steering_angle_ratio_arr = np.array(self.steering_angle_ratio_arr)
        self.avg_steering_angle_ratio=self.steering_angle_ratio_total / self.steps_total
        if len(steering_angle_ratio_arr) > 0:
            steering_angle_ratio_arr_no_nan = steering_angle_ratio_arr[~(np.isinf(steering_angle_ratio_arr)) & ~(np.isnan(steering_angle_ratio_arr))]
            self.avg_steering_angle_ratio_last_30=np.average(steering_angle_ratio_arr_no_nan[-30:]) if len(steering_angle_ratio_arr_no_nan) > 0 else 0.0
        else:
            self.avg_steering_angle_ratio_last_30=0.0
        # Crash-rate and episode-length metrics.
        # crashes_per_1k_steps: how often the car crashes per 1000 env steps.
        #   Low values during a gym wedge (gym stops responding → no episodes
        #   complete) or a very good policy. High values = frequent crashes.
        #   Uses crashes_total (crash-only), NOT num_episodes_total, since
        #   2026-07-20 added a second, non-crash way for an episode to end
        #   (GOALS_PER_EPISODE_CAP) - num_episodes_total now counts BOTH, so
        #   it would understate the actual crash rate if used here.
        # avg_steps_per_episode: mean steps per episode (crash- or
        #   goal-cap-terminated alike). Longer = car survives longer per
        #   episode before EITHER kind of ending.
        if self.steps_total > 0 and self.num_episodes_total > 0:
            self.avg_steps_per_episode = (
                self.steps_total / self.num_episodes_total)
        else:
            self.avg_steps_per_episode = 0.0
        if self.steps_total > 0:
            self.crashes_per_1k_steps = self.crashes_total / self.steps_total * 1000
        else:
            self.crashes_per_1k_steps = 0.0
    
    def do_reset_blocking(self):
        num_obstacles = self.get_num_obstacles()
        #self.reset_stats()
        # Forward the per-job track-curriculum knobs (set via env.configure)
        # so Unity regenerates the procedural track at the requested
        # difficulty on this reset. getattr defaults keep older envs working.
        corner_radius = getattr(self.env, "corner_radius", 10.0)
        curvature_difficulty = getattr(self.env, "curvature_difficulty", 0.0)
        chicanes_north = getattr(self.env, "chicanes_north", 0)
        chicanes_east = getattr(self.env, "chicanes_east", 0)
        chicanes_south = getattr(self.env, "chicanes_south", 0)
        chicanes_west = getattr(self.env, "chicanes_west", 0)
        self._api.DoResetBlocking(
            num_obstacles, corner_radius, curvature_difficulty,
            chicanes_north, chicanes_east, chicanes_south, chicanes_west)
