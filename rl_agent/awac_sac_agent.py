"""SAC + advantage-weighted behavior-cloning regularization (AWAC-style).

A drop-in subclass of tf-agents ``SacAgent`` that adds, to the actor loss, an
advantage-weighted imitation term computed on a *separate* batch sampled from
the protected demo (expert) replay table. This lets the demonstrations shape
the POLICY directly (not just the critic) every gradient step, so the actor
inherits the expert's survival/goal-chaining behavior without the one-shot
BC-pretrain -> SAC degradation we observed (best policy was the BC baseline,
SAC drifted off it). The advantage weighting (AWAC) means it only imitates
expert actions the critic rates better than the current policy, so it inherits
the expert's survival WITHOUT being chained to the expert's slowness.

Toggling
--------
Fully inert unless BOTH:
  * ``awac_lambda > 0``, and
  * a demo iterator has been attached via ``set_demo_iter(...)``.
Otherwise ``actor_loss`` is bit-identical to the base ``SacAgent``, so the
default training path is unchanged.

Wiring (see robotaxi.main):
  1. Construct ``AwacSacAgent`` with the awac_* knobs and ``demo_iter=None``
     (the demo buffer doesn't exist yet - it needs the agent's
     collect_data_spec).
  2. After ``make_local_replay`` builds the demo table, build a demo-only
     ``tf.data`` iterator and call ``agent.set_demo_iter(it)`` BEFORE the
     Learner first runs (so actor_loss is traced with the iterator live).

Validation notes (tf-agents 0.11 internals this relies on):
  * ``self._actor_network`` returns (action_distribution, network_state).
  * twin critics live at ``self._critic_network_1`` / ``self._critic_network_2``.
  * ``SacAgent.actor_loss(time_steps, weights=None)`` is the public hook
    ``_train`` calls. If a future tf-agents renames these, adjust here only.
"""
import tensorflow as tf
from tf_agents.agents.sac import sac_agent


# Demo iterators live here (keyed by id(agent)), NOT as agent attributes.
# A tf.data iterator is a Trackable; if it were an attribute the Learner's
# checkpointer would try to serialize it and raise UnimplementedError
# (Op:SerializeIterator). Keeping it module-level keeps the agent's Trackable
# graph clean so checkpointing works, while the iterator stays usable in-graph.
_DEMO_ITERS = {}


class AwacSacAgent(sac_agent.SacAgent):
    def __init__(self, *args,
                 awac_lambda=0.0,
                 awac_beta=1.0,
                 awac_weight_clip=20.0,
                 awac_lambda_decay_steps=0,
                 awac_lambda_min=0.0,
                 demo_iter=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._awac_lambda = float(awac_lambda)
        self._awac_beta = float(awac_beta)
        self._awac_weight_clip = float(awac_weight_clip)
        self._awac_lambda_decay_steps = int(awac_lambda_decay_steps)
        self._awac_lambda_min = float(awac_lambda_min)
        # Demo iterator goes in the module-level registry (not a tracked
        # attribute) so the Learner's checkpointer never serializes it. Usually
        # None at construction (the demo table doesn't exist yet); attached
        # later via set_demo_iter().
        if demo_iter is not None:
            _DEMO_ITERS[id(self)] = demo_iter

    def set_demo_iter(self, demo_iter):
        """Attach the demo-only tf.data iterator. Call before Learner.run().

        Stored in the module-level registry (not a tracked attribute) so it is
        never checkpointed - see the _DEMO_ITERS rationale above.
        """
        _DEMO_ITERS[id(self)] = demo_iter

    def _current_lambda(self):
        """λ schedule: constant, or linear decay lambda->lambda_min over
        awac_lambda_decay_steps (strong early survival, then let RL refine)."""
        if self._awac_lambda_decay_steps <= 0:
            return tf.constant(self._awac_lambda, tf.float32)
        s = tf.cast(self.train_step_counter, tf.float32)
        frac = tf.minimum(1.0, s / float(self._awac_lambda_decay_steps))
        return self._awac_lambda + frac * (self._awac_lambda_min - self._awac_lambda)

    def _critic_min(self, obs, act, step_type):
        """Pessimistic twin-critic value, matching SAC's min-of-two.

        tf-agents networks require (inputs, step_type, network_state)
        positionally; pass the demo batch's step_type and an empty state.
        Flatten to [B] so it broadcasts cleanly against log_prob ([B]).
        """
        q1, _ = self._critic_network_1((obs, act), step_type, (), training=False)
        q2, _ = self._critic_network_2((obs, act), step_type, (), training=False)
        return tf.reshape(tf.minimum(q1, q2), [-1])

    def actor_loss(self, time_steps, *args, **kwargs):
        sac_loss = super().actor_loss(time_steps, *args, **kwargs)
        demo_iter = _DEMO_ITERS.get(id(self))
        if demo_iter is None or self._awac_lambda <= 0.0:
            return sac_loss

        # ---- advantage-weighted BC on a fresh demo batch ----------------
        # demo dataset is as_dataset(sample_batch_size, num_steps=2) ->
        # (Trajectory, SampleInfo). Use the first transition's (s, a).
        demo, _info = next(demo_iter)
        s = demo.observation[:, 0]
        a_demo = demo.action[:, 0]
        st = demo.step_type[:, 0]

        # current policy distribution at the demo states. Networks require
        # (observations, step_type, network_state) positionally.
        dist, _ = self._actor_network(s, st, (), training=False)
        a_pi = dist.sample()

        # advantage A(s, a_demo) = Q(s, a_demo) - V(s),  V(s) ~ Q(s, a~pi)
        q_demo = self._critic_min(s, a_demo, st)
        v_s = self._critic_min(s, a_pi, st)
        adv = tf.clip_by_value(q_demo - v_s, -10.0, 10.0)

        # exp(A/beta) weight; stop-grad (it's a coefficient), clip for stability
        w = tf.exp(adv / self._awac_beta)
        w = tf.stop_gradient(tf.minimum(w, self._awac_weight_clip))

        # Advantage-weighted regression toward the expert action via MSE
        # (TD3+BC-style), NOT tanh-squashed log_prob. log_prob's value AND
        # gradient blow up to inf/nan when the expert action sits at the
        # action-space bounds (steering +/-1, accel near the 0.1 floor) - the
        # source of the iter-1 NaN. MSE is bounded with well-behaved grads.
        # a_pi is the reparameterized policy sample, so the gradient pulls the
        # policy toward the (advantage-weighted) expert action.
        low = tf.constant(self.action_spec.minimum, dtype=a_demo.dtype)
        high = tf.constant(self.action_spec.maximum, dtype=a_demo.dtype)
        a_demo_c = tf.clip_by_value(a_demo, low, high)
        bc_se = tf.reduce_sum(tf.square(a_pi - a_demo_c), axis=-1)
        awac_bc = tf.reduce_mean(w * bc_se)

        return sac_loss + self._current_lambda() * awac_bc
