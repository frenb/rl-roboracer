"""Teacher-forced parity of the Unity duck task (DuckTask.cs) against mjlab.

Rolls out Mjlab-Velocity-Flat-MicroDuck in mjlab with random actions, records each step's
states (the last physics substep's input state, the post-step state rewards see, the state
the observations see) plus every per-env quantity the terms read, then has the Unity server
evaluate rewards, terminations and observations on exactly those states (protocol command
"eval") and compares term by term. Randomization that changes the model (mass, CoM, foot
friction, armature) and observation noise and delays are switched off on both sides; encoder
bias, IMU misalignment and BAM friction stay on and are fed through.

Per-foot contact forces are the one thing compared against MuJoCo C (Python bindings, on
mjlab's own compiled model) rather than mjlab: MuJoCo Warp, which mjlab trains with, splits
the load among a foot's contact points differently from MuJoCo C on identical states (same
contacts found, similar total force). That gap is the engine's, not the port's, and is
reported for information.

Run in the duck-trainer container, with the Linux server build mounted:
    python test_task_parity.py            # prints a per-term report, exits 1 on a mismatch
    pytest test_task_parity.py
"""
import copy
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from duck import unity_env  # noqa: E402

NUM_ENVS, STEPS, SEED = 64, 200, 3
# Past the first curriculum stages so every reward term has a non-zero weight; no stage
# boundary is crossed during the rollout.
GLOBAL_STEP = 20000
ACTION_STD = 0.25

# max |unity - mjlab| per reward term (weight * term, before dt scaling) and per obs term.
# mjlab runs MuJoCo Warp in float32; Unity runs MuJoCo C in float64.
REWARD_ATOL = 2e-3
REWARD_RTOL = 2e-3
OBS_ATOL_DEFAULT = 2e-3
# Contact flags come out of collision detection in different engines, and contact forces out
# of near-degenerate solves in differently compiled MuJoCo C builds; a few may disagree.
CONTACT_TERMS = {"foot_slip", "foot_swing_height", "air_time", "foot_contact", "self_collisions"}
ENGINE_TERMS = {"foot_contact_forces"}   # compared against MuJoCo C, see the module docstring
MAX_CONTACT_MISMATCH = 0.01

CONTACT_SENSOR, HEIGHT_SENSOR = "feet_ground_contact", "foot_height_scan"


def parity_cfg(cfg):
    """mjlab cfg with model-changing randomization, obs noise and obs delays off."""
    for e in cfg.events.values():
        name = getattr(e.func, "__name__", type(e.func).__name__)
        p = e.params
        if name == "pseudo_inertia":
            p["alpha_range"] = (0.0, 0.0)
        elif name == "body_com_offset":
            p["ranges"] = (0.0, 0.0)
        elif name == "geom_friction":
            p["ranges"] = (1.0, 1.0)
        elif name == "joint_armature":
            p["ranges"] = (1.0, 1.0)
    for k in [k for k, c in cfg.curriculum.items() if "com" in getattr(c.func, "__name__", "")]:
        cfg.curriculum.pop(k)
    for group in ("actor", "critic"):
        for t in cfg.observations[group].terms.values():
            t.noise = None
            t.delay_min_lag = t.delay_max_lag = 0
    return cfg


def parity_spec(spec):
    """The exported task with the same switches as parity_cfg."""
    ev = spec["events"]
    ev["mass_inertia"]["alpha"] = [0.0, 0.0]
    for c in ev.get("com", []):
        c["range"] = [0.0, 0.0]
    ev["foot_friction"]["range"] = [1.0, 1.0]
    ev["armature"]["range"] = [1.0, 1.0]
    spec["curriculum"] = [c for c in spec["curriculum"] if c["kind"] != "com_range"]
    for group in ("actor", "critic"):
        for t in spec[group]:
            t["noise"] = [0.0, 0.0]
            t["delay"].update(min_lag=0, max_lag=0)
    return spec


def np64(x):
    return x.detach().double().cpu().numpy()


class Recorder:
    """Hooks sim.step, sim.forward and reward_manager.compute of an mjlab env to capture the
    states and term inputs of one env.step."""

    def __init__(self, env):
        self.env = env
        sim, rm = env.sim, env.reward_manager
        self._sim_step, self._sim_forward, self._rewards = sim.step, sim.forward, rm.compute
        sim.step, sim.forward, rm.compute = self.sim_step, self.sim_forward, self.rewards
        names = list(rm._term_names)
        self.swing = rm._term_cfgs[names.index("foot_swing_height")].func
        self.joint_ids = env.action_manager._terms["joint_pos"]._target_ids
        sites = unity_env.load_task_spec()["foot_sites"]
        self.foot_sites = [env.sim.mj_model.site("robot/" + s).id for s in sites]
        self.max_height = env.scene[HEIGHT_SENSOR].cfg.max_distance
        self.rec = {}

    def sim_step(self):
        d, m = self.env.sim.data, self.env.sim.model
        n = self.env.num_envs
        self.rec.update(qpos_prev=np64(d.qpos), qvel_prev=np64(d.qvel),
                        qacc_warmstart=np64(d.qacc_warmstart), ctrl=np64(d.ctrl),
                        dof_frictionloss=np.broadcast_to(np64(m.dof_frictionloss), (n, d.qvel.shape[1])),
                        dof_damping=np.broadcast_to(np64(m.dof_damping), (n, d.qvel.shape[1])))
        return self._sim_step()

    def sim_forward(self):
        out = self._sim_forward()
        d = self.env.sim.data
        self.rec.update(qpos_obs=np64(d.qpos), qvel_obs=np64(d.qvel))
        return out

    def rewards(self, dt):
        e = self.env
        d, cm, am = e.sim.data, e.command_manager, e.action_manager
        contact = e.scene[CONTACT_SENSOR].data
        ema = getattr(e, "_head_bias_ema", None)
        self.rec.update(
            qpos=np64(d.qpos), qvel=np64(d.qvel), action=np64(am.action), prev_action=np64(am.prev_action),
            twist=np64(cm.get_command("twist"))[:, :3], head=np64(cm.get_command("head_pose")),
            air_time=np64(contact.current_air_time), contact_time=np64(contact.current_contact_time),
            foot_height=np64(e.scene[HEIGHT_SENSOR].data.heights),
            peak_height=np64(self.swing.peak_heights),
            head_ema=np.zeros((e.num_envs, 4)) if ema is None else np64(ema),
            episode_length=np64(e.episode_length_buf)[:, None],
            terminated=np64(e.termination_manager.terminated), time_out=np64(e.termination_manager.time_outs))
        out = self._rewards(dt)
        self.rec.update(reward=np64(out), reward_terms=np64(e.reward_manager._step_reward))
        return out

    def after_step(self):
        e = self.env
        cm = e.command_manager
        self.rec.update(
            twist_obs=np64(cm.get_command("twist"))[:, :3], head_obs=np64(cm.get_command("head_pose")),
            body_obs=np64(cm.get_command("body_pose")),
            encoder_bias=np64(e.scene["robot"].data.encoder_bias[:, self.joint_ids]),
            imu_quat=np64(e._imu_misalign_quat),
            site_height=np.clip(np64(e.sim.data.site_xpos[:, self.foot_sites, 2]), 0, self.max_height))
        done = (self.rec["terminated"] > 0) | (self.rec["time_out"] > 0)
        self.rec["sensors_refreshed"] = np.full((e.num_envs, 1), float(done.any()))
        rec, self.rec = self.rec, {}
        return rec


class MujocoC:
    """The contact force observation from MuJoCo C on a snapshot, the way DuckAgent.Eval
    computes it: forward on the last substep's input state (reward time), then, if the batch's
    sensor caches were refreshed, forward on the observation state."""

    def __init__(self, env, spec):
        import mujoco
        self.mujoco = mujoco
        self.m = copy.deepcopy(env.sim.mj_model)
        self.d = mujoco.MjData(self.m)
        friction = spec["events"]["foot_friction"]
        for g in friction["geoms"]:
            self.m.geom("robot/" + g).friction[0] = friction["range"][0]
        self.adr = [self.m.sensor(f"{CONTACT_SENSOR}_{g}_force").adr[0] for g in friction["geoms"]]

    def contact_forces(self, r):
        m, d, mj = self.m, self.d, self.mujoco
        n = r["qpos"].shape[0]
        out = np.zeros((n, 6))
        for i in range(n):
            d.qpos[:], d.qvel[:] = r["qpos_prev"][i], r["qvel_prev"][i]
            d.qacc_warmstart[:], d.ctrl[:] = r["qacc_warmstart"][i], r["ctrl"][i]
            m.dof_frictionloss[:], m.dof_damping[:] = r["dof_frictionloss"][i], r["dof_damping"][i]
            mj.mj_forward(m, d)
            if r["sensors_refreshed"][i, 0]:
                d.qpos[:], d.qvel[:] = r["qpos_obs"][i], r["qvel_obs"][i]
                mj.mj_forward(m, d)
            out[i] = np.concatenate([d.sensordata[a:a + 3] for a in self.adr])
        return np.sign(out) * np.log1p(np.abs(out))


def term_slices(terms):
    out, k = {}, 0
    for name, width in terms:
        out[name] = slice(k, k + width)
        k += width
    return out


def run(num_envs=NUM_ENVS, steps=STEPS, seed=SEED, verbose=True):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

    cfg = parity_cfg(make_microduck_velocity_env_cfg())
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = GLOBAL_STEP
    env.reset()
    rec = Recorder(env)
    rng = np.random.default_rng(seed)

    spec = parity_spec(unity_env.load_task_spec())
    server = unity_env.DuckServer(num_envs=num_envs, seed=seed, task=spec, global_step=GLOBAL_STEP)
    failures = []
    try:
        layout = dict(server.eval_layout)
        nq = env.sim.data.qpos.shape[1]
        if layout["qpos"] != nq or layout["qvel"] != env.sim.data.qvel.shape[1]:
            raise AssertionError(f"Unity model nq/nv {layout['qpos']}/{layout['qvel']} != mjlab "
                                 f"{nq}/{env.sim.data.qvel.shape[1]}")
        names = server.reward_names
        actor_sl, critic_sl = term_slices(server.header["actor_terms"]), term_slices(server.header["critic_terms"])
        err = {n: [] for n in names}
        rel = {n: [] for n in names}
        obs_err = {("actor", n): [] for n in actor_sl} | {("critic", n): [] for n in critic_sl}
        flags = {"terminated": 0, "time_out": 0, "height_lag": []}
        mismatched = {n: 0 for n in names} | {k: 0 for k in obs_err}
        counted = 0
        prev_site_height = None
        engine = MujocoC(env, spec)
        engine_gap = []

        for step in range(steps):
            a = np.clip(rng.normal(0, ACTION_STD, (num_envs, server.act_dim)), -1, 1).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(torch.as_tensor(a, device=device))
            r = rec.after_step()
            out = server.evaluate(r)
            done = (r["terminated"] > 0) | (r["time_out"] > 0)
            keep = ~done
            counted += int(keep.sum())

            flags["terminated"] += int((out["terminated"] != (r["terminated"] > 0)).sum())
            unity_timeout = out["truncated"] > 0
            flags["time_out"] += int((unity_timeout != (r["time_out"] > 0)).sum())
            if prev_site_height is not None:
                flags["height_lag"].append(np.abs(r["foot_height"] - prev_site_height).max())
            prev_site_height = r["site_height"]

            for i, n in enumerate(names):
                u, m = out["reward_terms"][:, i].astype(np.float64), r["reward_terms"][:, i]
                diff = np.abs(u - m)
                bad = diff > REWARD_ATOL + REWARD_RTOL * np.abs(m)
                mismatched[n] += int(bad.sum())
                err[n].append(diff.max())
                rel[n].append((diff / np.maximum(np.abs(m), 1e-6)).max())

            # Observations of reset envs come after a reset the snapshot doesn't carry.
            c_forces = engine.contact_forces(r)
            for group, sl, uo in (("actor", actor_sl, out["obs/policy"]), ("critic", critic_sl, out["obs/critic"])):
                mo = np64(obs[group])
                for n, s in sl.items():
                    ref = mo[:, s]
                    if n in ENGINE_TERMS:
                        gap = np.abs(ref - c_forces)[keep]
                        if gap.size:
                            engine_gap.append(gap)
                        ref = c_forces
                    diff = np.abs(uo[keep, s] - ref[keep])
                    if diff.size == 0:
                        continue
                    bad = np.flatnonzero(diff.max(1) > OBS_ATOL_DEFAULT)
                    if verbose and bad.size and not mismatched[(group, n)]:
                        i = np.flatnonzero(keep)[bad[0]]
                        print(f"step {step} env {i} {group}/{n}:\n  reference {np.round(ref[i], 4)}\n"
                              f"  unity     {np.round(uo[i, s], 4)}")
                    mismatched[(group, n)] += bad.size
                    obs_err[(group, n)].append(diff.max())

        total = steps * num_envs
        if verbose:
            print(f"{steps} steps x {num_envs} envs at global step {GLOBAL_STEP}, "
                  f"{total - counted} env-steps with a reset left out of the obs comparison")
            print(f"{'reward term':26s} {'max abs err':>12s} {'max rel err':>12s} {'mismatched':>11s}")
        for n in names:
            frac = mismatched[n] / total
            limit = MAX_CONTACT_MISMATCH if n in CONTACT_TERMS else 0
            ok = frac <= limit
            if verbose:
                print(f"{n:26s} {max(err[n]):12.2e} {max(rel[n]):12.2e} {frac:10.2%}{'' if ok else '  FAIL'}")
            if not ok:
                failures.append(f"reward {n}: {frac:.2%} env-steps off, max err {max(err[n]):.3g}")
        if verbose:
            print(f"{'obs term':26s} {'max abs err':>12s} {'':12s} {'mismatched':>11s}")
        for (group, n), e in obs_err.items():
            frac = mismatched[(group, n)] / max(counted, 1)
            limit = MAX_CONTACT_MISMATCH if n in CONTACT_TERMS | ENGINE_TERMS else 0
            ok = frac <= limit
            if verbose:
                note = "  (vs MuJoCo C)" if n in ENGINE_TERMS else ""
                print(f"{group + '/' + n:26s} {max(e, default=0):12.2e} {'':12s} {frac:10.2%}"
                      f"{'' if ok else '  FAIL'}{note}")
            if not ok:
                failures.append(f"obs {group}/{n}: {frac:.2%} env-steps off, max err {max(e, default=0):.3g}")
        if verbose and engine_gap:
            g = np.concatenate(engine_gap)
            print(f"MuJoCo Warp vs MuJoCo C, foot contact forces (sign * log1p|F|): "
                  f"median {np.median(g):.3f}, 95th percentile {np.percentile(g, 95):.3f}, max {g.max():.3f}")
        for k in ("terminated", "time_out"):
            if verbose:
                print(f"{k:26s} {flags[k]} env-steps disagree")
            if flags[k] > MAX_CONTACT_MISMATCH * total:
                failures.append(f"{k}: {flags[k]} env-steps disagree")
        lag = max(flags["height_lag"], default=0)
        if verbose:
            print(f"reward-time foot heights vs clamped foot site z after the previous step: max diff {lag:.2e}")
        if lag > 1e-5:
            failures.append(f"rewards read foot heights other than clamp(site z) at the previous sense "
                            f"(diff {lag:.3g}); DuckAgent.Sense doesn't model the height scan")
    finally:
        server.close()
        env.close()
    return failures


needs_build = os.path.exists(os.path.join(unity_env.SERVER_DIR, unity_env.EXECUTABLE))


def test_unity_task_matches_mjlab():
    import pytest
    if not needs_build:
        pytest.skip(f"no duck server build at {unity_env.SERVER_DIR}")
    pytest.importorskip("mjlab_microduck")
    failures = run(verbose=False)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    fails = run()
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)
