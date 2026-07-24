# Double (twin) critic implementation

This document explains how the RoboRacer policy ends up with **two critic
networks** even though the trainer only ever constructs and passes **one**, and
how those twin critics are used during training. It exists because the
single-argument `SacAgent(critic_network=...)` constructor makes it look like
there is only one critic — the second is created *inside* tf-agents.

TL;DR: **You provide one `CriticNetwork`; SAC copies it into a second
independent network (`CriticNetwork2`), and also builds target copies of both.**
This is the standard SAC *clipped double-Q* design (Fujimoto et al. TD3 /
Haarnoja et al. SAC) for reducing Q-value overestimation. Our `AwacSacAgent`
inherits this unchanged and additionally uses the twin-critic minimum in its
advantage-weighted BC term.

---

## 1. Where the second critic comes from

The trainer builds exactly one critic and one actor
(`rl_agent/robotaxi.py`, network-build phase):

```python
critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params,   # (512, 512) by default
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform')
```

and passes only `critic_network=critic_net` (no `critic_network_2`) when it
constructs the agent.

tf-agents `SacAgent.__init__` then creates the second critic itself. From the
installed source
(`/usr/local/lib/python3.8/dist-packages/tf_agents/agents/sac/sac_agent.py`):

```python
self._critic_network_1 = critic_network
if critic_network_2 is not None:
    self._critic_network_2 = critic_network_2
else:
    self._critic_network_2 = critic_network.copy(name='CriticNetwork2')
    # Do not use target_critic_network_2 if critic_network_2 is None.
    target_critic_network_2 = None
...
self._critic_network_1.create_variables(critic_spec)
self._critic_network_2.create_variables(critic_spec)
```

Key point: `critic_network.copy(name='CriticNetwork2')` clones the **topology**
but `create_variables` then initializes a **fresh, independent set of weights**
for critic 2. The two online critics are *not* weight-tied — they diverge
during training, which is exactly what makes the min-of-two estimate useful.

### Networks that actually exist at runtime

After construction there are **four** critic networks:

| Network | Attribute | Role |
|---|---|---|
| Critic 1 (online) | `self._critic_network_1` | trained by critic loss |
| Critic 2 (online) | `self._critic_network_2` | trained by critic loss (independent weights) |
| Target critic 1 | `self._target_critic_network_1` | slow-moving copy of critic 1 for TD targets |
| Target critic 2 | `self._target_critic_network_2` | slow-moving copy of critic 2 for TD targets |

The two target networks are also created by copying inside `SacAgent.__init__`
and are updated every `target_update_period` steps toward their online
counterparts (Polyak/`target_update_tau`).

---

## 2. Critic network architecture (each critic)

Each of the four critics is the same MLP topology:

- **Input:** the joint `(observation, action)` tuple — 32-dim observation +
  2-dim action. No separate observation/action pre-encoders
  (`observation_fc_layer_params=None`, `action_fc_layer_params=None`).
- **Hidden:** joint MLP `joint_fc_layer_params = (512, 512)` — two dense layers,
  ReLU activation (tf-agents default), `glorot_uniform` init.
- **Output:** a single scalar Q-value (`value` layer), `glorot_uniform` init.

Sizes are job-configurable via `nn_size_x` / `nn_size_y` (both default `512`),
which feed `critic_joint_fc_layer_params_x/y`.

In TensorBoard the two online critics log under the tag prefixes
`CriticNetwork/joint_mlp/...` + `CriticNetwork/value/...` and
`CriticNetwork2/joint_mlp/...` + `CriticNetwork2/value/...`, with independent
`summarize_vars/` (weights) and `summarize_grads/` (gradients) series — direct
runtime proof that both exist and are being trained.

---

## 3. How the two critics are used

### 3.1 SAC (base class) — clipped double-Q

Standard SAC (inherited unchanged from `sac_agent.SacAgent`):

- **Critic loss:** the Bellman target uses the **minimum** of the two *target*
  critics at the next state,
  `y = r + γ · (min(Q'₁(s',a'), Q'₂(s',a')) − α·log π(a'|s'))`,
  and both online critics are regressed toward that same `y`. Taking the min is
  the "clipped double-Q" trick that curbs Q overestimation.
- **Actor loss:** the policy is improved against the **minimum** of the two
  *online* critics, `min(Q₁(s,a~π), Q₂(s,a~π))`, again to avoid exploiting an
  over-optimistic single critic.
- **Alpha (entropy temperature) loss:** unaffected by the critic count.

### 3.2 AWAC subclass — advantage-weighted BC term

`rl_agent/awac_sac_agent.py` (`AwacSacAgent(sac_agent.SacAgent)`) inherits the
twin critics (`super().__init__(...)`) and adds an advantage-weighted imitation
term to the actor loss, computed on a separate batch from the protected demo
replay table. It uses the **same pessimistic min-of-two** as SAC:

```python
def _critic_min(self, obs, act, step_type):
    """Pessimistic twin-critic value, matching SAC's min-of-two."""
    q1, _ = self._critic_network_1((obs, act), step_type, (), training=False)
    q2, _ = self._critic_network_2((obs, act), step_type, (), training=False)
    return tf.reshape(tf.minimum(q1, q2), [-1])
```

The advantage of the expert action is then
`A(s, a_demo) = min-critic(s, a_demo) − min-critic(s, a~π)`, turned into an
`exp(A/β)` weight (clipped) that scales an MSE regression of the policy toward
the expert action:

```python
q_demo = self._critic_min(s, a_demo, st)   # Q(s, a_demo)
v_s    = self._critic_min(s, a_pi,  st)    # V(s) ~ Q(s, a~pi)
adv = tf.clip_by_value(q_demo - v_s, -10.0, 10.0)
w   = tf.stop_gradient(tf.minimum(tf.exp(adv / self._awac_beta),
                                  self._awac_weight_clip))
awac_bc = tf.reduce_mean(w * bc_se)        # bc_se = ||a_pi - a_demo||^2
return sac_loss + self._current_lambda() * awac_bc
```

So the demonstrations shape the **policy** directly every step, but only toward
expert actions the (pessimistic) critic rates better than the current policy.
The AWAC term is fully inert (bit-identical to base SAC) unless `awac_lambda > 0`
**and** a demo iterator has been attached via `set_demo_iter(...)`.

---

## 4. Configuration knobs

| Knob | Where | Default | Effect on critics |
|---|---|---|---|
| `nn_size_x`, `nn_size_y` | job doc / experiment design | 512, 512 | joint MLP layer sizes for *each* critic |
| `critic_network_2` | `SacAgent.__init__` arg | not passed | when omitted, SAC copies critic 1 → critic 2 |
| `target_update_tau`, `target_update_period` | SAC | tf-agents defaults | how fast target critics track online critics |
| `awac_lambda`, `awac_beta`, `awac_weight_clip` | experiment design | see design | weight of the twin-critic-based AWAC BC term |

We never pass `critic_network_2`, so the auto-copy path is always taken.

---

## 5. How to verify it yourself

1. **tf-agents source** (authoritative): grep the installed `sac_agent.py`
   for `critic_network.copy(` / `_critic_network_2` — see §1.
2. **Runtime TensorBoard tags:** a live TRAIN run logs both
   `summarize_vars/CriticNetwork/...` and `summarize_vars/CriticNetwork2/...`
   (plus matching `summarize_grads/...`). Two independent series ⇒ two networks.
3. **In-process probe:**
   ```python
   # inside the sim-controller container, after the agent is built
   agent._critic_network_1 is agent._critic_network_2   # -> False
   agent._critic_network_1.name, agent._critic_network_2.name
   # -> ('CriticNetwork', 'CriticNetwork2')
   ```

---

## 6. Common misconception

"tf-agents SAC only has one critic" is a natural conclusion from the
**constructor signature** — `SacAgent(critic_network=...)` takes a single
critic. But that is only the *input*: the constructor's body (§1) copies it into
`CriticNetwork2` and builds two target networks, so the *agent* runs the full
twin-critic (clipped double-Q) algorithm. Both interpretations are consistent —
you supply one critic, SAC instantiates the rest.

---

## References

- Haarnoja et al., "Soft Actor-Critic" (2018) — entropy-regularized off-policy
  actor-critic with two Q-functions.
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic
  Methods" (TD3, 2018) — clipped double-Q (min of two critics).
- Nair et al., "AWAC: Accelerating Online RL with Offline Datasets" (2020) —
  advantage-weighted actor update used by `AwacSacAgent`.
- Code: `rl_agent/robotaxi.py` (network build + agent construction),
  `rl_agent/awac_sac_agent.py` (AWAC term + `_critic_min`),
  tf-agents `agents/sac/sac_agent.py` (twin/target critic construction).
