import numpy as np
from tf_agents.specs import array_spec

from .donut_course_no_hint import DonutCourseNoHint

IMAGE_HW = (84, 84, 3)


class DonutCourseCamera(DonutCourseNoHint):
    """31-D no-hint vector plus a CSI image. Dict observation, not a flatten.

    ``scene_data_array()`` stays the 31-D vector so rewards, stuck
    detection, and demo metrics keep indexing ``data_arr`` as before.
    The env packs ``{vector, image}`` onto the TimeStep. Do not load a
    ``donut`` / ``donut_no_hint`` checkpoint into this spec.
    """

    def __init__(self, api, env):
        super().__init__(api, env)
        vec = self.observation_spec
        self.vector_spec = vec
        self.observation_spec = {
            "vector": array_spec.BoundedArraySpec(
                shape=vec.shape,
                dtype=vec.dtype,
                minimum=np.asarray(vec.minimum, dtype=vec.dtype),
                maximum=np.asarray(vec.maximum, dtype=vec.dtype),
                name="vector"),
            "image": array_spec.BoundedArraySpec(
                shape=IMAGE_HW,
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
                name="image"),
        }

    def get_empty_state(self):
        return {
            "vector": np.asarray(super().get_empty_state(), dtype=np.float32),
            "image": np.zeros(IMAGE_HW, dtype=np.float32),
        }

    def policy_vector(self, data_arr):
        """Slice of ``scene_data_array()`` the policy is allowed to see.

        The env packs this into the ``vector`` slot of the dict observation
        while rewards, stuck detection, and demo metrics keep using the full
        ``data_arr``. Here they are the same 31-D array; the no-rays ablation
        subclass narrows it. Whatever this returns must match
        ``observation_spec["vector"]``.
        """
        return data_arr
