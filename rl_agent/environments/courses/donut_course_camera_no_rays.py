import numpy as np
from tf_agents.specs import array_spec

from .donut_course_camera import DonutCourseCamera, IMAGE_HW

# Number of leading elements of the 31-D no-hint vector the policy keeps:
#   [0] scene_data["car"]["speed"]  - magnitude of car velocity
#   [1] scene_data["car"]["goal_2"] - sideslip, angle from velocity to heading
# Everything after index 1 is the 29 raycast distances this ablation removes.
POLICY_VECTOR_DIMS = 2


class DonutCourseCameraNoRays(DonutCourseCamera):
    """Ablation arm: ``donut_camera`` with the raycasts cut from the policy input.

    Answers whether the CSI image alone can replace the 29 raycast distances.
    The policy sees ``{vector: (2,), image: (84, 84, 3)}`` - speed and sideslip,
    both reproducible on the real JetRacer from IMU/odometry, plus the camera.

    ``scene_data_array()`` is deliberately NOT overridden. It stays 31-D so the
    reward designs, ``has_failed()``, curriculum, and ``_step``'s ``data_arr[7]``
    step cost keep indexing exactly as they do on every other donut course. Only
    ``policy_vector()`` narrows, and the env packs that into the observation.

    The 29 ray slots are dropped, not zeroed. A 31-D vector of mostly zeros
    would produce a spec identical to ``donut_camera`` and let checkpoints
    cross-load between the two arms, silently contaminating the comparison.
    That also means ``donut_camera`` checkpoints will not load here, which is
    intended - the spec mismatch is the guardrail.
    """

    def __init__(self, api, env):
        super().__init__(api, env)
        # Derive the narrowed bounds by slicing the parent's vector spec rather
        # than re-listing them, so this stays in sync if the no-hint vector's
        # leading features ever change.
        vec = self.observation_spec["vector"]
        self.observation_spec = {
            "vector": array_spec.BoundedArraySpec(
                shape=(POLICY_VECTOR_DIMS,),
                dtype=vec.dtype,
                minimum=np.asarray(vec.minimum, dtype=vec.dtype)[:POLICY_VECTOR_DIMS],
                maximum=np.asarray(vec.maximum, dtype=vec.dtype)[:POLICY_VECTOR_DIMS],
                name="vector"),
            "image": self.observation_spec["image"],
        }
        self.vector_spec = self.observation_spec["vector"]

    def get_empty_state(self):
        return {
            "vector": np.zeros(POLICY_VECTOR_DIMS, dtype=np.float32),
            "image": np.zeros(IMAGE_HW, dtype=np.float32),
        }

    def policy_vector(self, data_arr):
        # Speed and sideslip only; drop the 29 raycasts. Callers that need the
        # full vector (rewards, stats) keep using data_arr directly.
        return np.asarray(data_arr[:POLICY_VECTOR_DIMS], dtype=np.float32)
