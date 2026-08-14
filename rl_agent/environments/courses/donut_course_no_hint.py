import numpy as np
from tf_agents.specs import array_spec

from .donut_course import DonutCourse


class DonutCourseNoHint(DonutCourse):
    """DonutCourse variant that drops the goal-derived steering hint from
    the observation.

    The base DonutCourse observation vector's leading element is
    ``scene_data["car"]["dist_from_traj"]`` == Unity's ``GetAngleToGoal()``
    (the signed angle from the car's heading to the next Goal object). That
    quantity only exists because the simulator knows where the goals are; it
    has no analogue on a real "JetRacer ROS AI Robot", which has no goal
    objects to measure an angle to. Training against it teaches the policy to
    lean on a signal that vanishes at transfer time.

    This subclass removes ONLY that first element, shrinking the observation
    from 32 -> 31 dims. Everything else (reward logic, curriculum, stats,
    termination, action_spec) is inherited unchanged.

    Note: index 2 (stored in the ``goal_2`` field) is intentionally KEPT. It
    is ``GetVelocityCarAngleDiff()`` - the angle between the car's heading and
    its velocity vector (sideslip). Despite its field name it is not
    goal-derived and is reproducible on the real robot from IMU/odometry, so
    it stays in the observation.

    ``dist_from_traj`` remains available in the raw ``data`` dict, so the
    per-step stats (``steering_angle_ratio``) and the ``pass_through_actions``
    expert/PID controller that generates demos are unaffected - the expert can
    still steer using the hint while the RECORDED observation excludes it,
    which is exactly the imitation-learning setup we want for demo bootstrap.
    """

    def __init__(self, api, env):
        super().__init__(api, env)
        # Rebuild the observation spec by slicing off the leading
        # dist_from_traj element. Deriving from the parent's spec (rather than
        # re-listing all bounds) keeps this in sync automatically if the base
        # DonutCourse observation ever gains/reorders trailing features.
        base_spec = self.observation_spec
        base_min = np.asarray(base_spec.minimum, dtype=base_spec.dtype)
        base_max = np.asarray(base_spec.maximum, dtype=base_spec.dtype)
        self.observation_spec = array_spec.BoundedArraySpec(
            shape=(base_spec.shape[0] - 1,),
            dtype=base_spec.dtype,
            minimum=base_min[1:],
            maximum=base_max[1:],
            name='observation')

    def get_empty_state(self):
        # One shorter than the base course (31 instead of 32).
        return super().get_empty_state()[1:]

    def scene_data_array(self, scene_data):
        # Same feature vector as the base course, minus the leading
        # dist_from_traj (angle-to-next-goal) element.
        return super().scene_data_array(scene_data)[1:]
