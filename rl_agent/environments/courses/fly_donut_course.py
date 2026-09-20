"""The policy observes the fly connectome instead of the raycasts.

Step 10 of docs/flybrain-driver-plan.md, the point where this stops being
behaviour cloning. Steps 5 and 6 fit a ridge readout from the descending-neuron
trace onto expert actions; here the trace *is* the observation and SAC learns
that readout from reward instead.

That matters because the ridge could not fit throttle. It reached R^2 0.620 on
steering but only 0.176 on acceleration, and its acceleration prediction sits
below the course's own 0.05 action floor on ~54% of frames, so the clip to the
action spec pins the car near minimum throttle and it drives at 1.9-2.4 m/s
against SAC's 5.4. Learning the readout from reward is the direct answer.

``scene_data_array()`` is untouched - still the 31-D no-hint vector - so
rewards, stuck detection, the curriculum and the per-step stats keep reading
exactly what they read on donut_no_hint. Only ``policy_vector`` changes, which
is the seam donut_camera_no_rays already established.

There is no demo corpus at this width and there cannot be one: the trace
depends on the brain's own history, so a recorded observation is not
reconstructible from a stored scene. TRAIN from scratch only; do_job refuses
DEMO and BC_TRAINING_ONLY here.
"""
import numpy as np
from tf_agents.specs import array_spec

from .donut_course_no_hint import DonutCourseNoHint


class FlyDonutCourse(DonutCourseNoHint):
    """31-D scene vector in, descending-neuron trace out.

    Stateful in a way the other courses are not: the brain's voltages carry
    across steps, so the observation at step t depends on the whole episode so
    far. ``on_episode_start`` clears it, and the seed advances per episode so
    runs stay reproducible while the spiking noise still varies between them.
    """

    def __init__(self, api, env):
        super().__init__(api, env)
        # Imported here rather than at module scope so that loading the
        # courses package does not pull in grpc or dial the fly-brain service
        # for the four courses that have nothing to do with it.
        from fly_brain.client import FlyBrainClient, SUBSTEPS
        from fly_brain.encoder import RayEncoder, resolve_cells
        from fly_brain.viz import FlyBrainViz

        self._client = FlyBrainClient()
        self._enc = RayEncoder()
        self._cells = resolve_cells(self._client)
        self._substeps = SUBSTEPS
        # The overlay rides along on the step we already make, so watching the
        # brain during training costs one extra field rather than a second
        # round trip. Part 4 exists for this.
        self._viz = FlyBrainViz(self._client)

        info = self._client.info
        self._trace_len = int(info.trace_len)
        # A spike adds 1.0 and the trace decays by exp(-dt/tau) each substep,
        # so a neuron firing every substep converges on 1/(1-decay) ~= 5.5.
        # Derived rather than hardcoded because both dt and tau are the
        # service's to choose.
        decay = float(np.exp(-info.dt / info.trace_tau))
        ceiling = float(1.0 / (1.0 - decay))
        self.observation_spec = array_spec.BoundedArraySpec(
            shape=(self._trace_len,),
            dtype=np.float32,
            minimum=0.0,
            maximum=ceiling,
            name='observation')

        self._episode = 0
        self._steps = 0
        # Whatever the last job left in the service, this episode starts clean.
        self._client.reset(seed=self._episode)
        self._enc.reset()
        print("fly_donut: trace_len=%d obs_max=%.2f substeps=%d device=%s"
              % (self._trace_len, ceiling, self._substeps, info.device),
              flush=True)

    def get_empty_state(self):
        return np.zeros(self._trace_len, dtype=np.float32)

    def policy_vector(self, data_arr):
        """Advance the brain one control step and hand back its trace.

        Called exactly once per env step, from ``_pack_observation``. Stepping
        here rather than in a policy is the whole point: the trace reaches the
        replay buffer as the observation, so SAC's critic sees it.
        """
        _, inject = self._enc.encode(
            np.asarray(data_arr, dtype=np.float32), self._cells)
        trace, snap, spikes = self._client.step(
            inject, substeps=self._substeps, want_snapshot=self._viz.enabled)
        self._steps += 1
        self._viz.submit(snap, step=self._steps, spikes=spikes)
        return trace

    def on_episode_start(self):
        super().on_episode_start()
        self._episode += 1
        self._client.reset(seed=self._episode)
        self._enc.reset()

    def close(self):
        self._viz.stop()
        self._client.close()
