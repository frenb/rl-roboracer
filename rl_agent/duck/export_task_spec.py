"""Export the resolved mjlab Velocity-Flat MicroDuck task as JSON for the Unity duck server.

Every number the C# task uses (reward weights and parameters, observation layout, noise and
delays, command ranges, domain randomization, curriculum stages, BAM actuator constants,
default pose and soft joint limits) is read from make_microduck_velocity_env_cfg() and the
compiled scene here, so the server never carries hand-copied values. Any term whose function
the C# side doesn't implement fails the export instead of being silently dropped; terms with
weight 0 that no curriculum touches are exported as disabled.

Run in duck-trainer:
    cd /microduck_rl && python /python_ws/src/duck/export_task_spec.py out.json
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys

import numpy as np

TASK = "Mjlab-Velocity-Flat-MicroDuck"


def qualname(f) -> str:
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}"


def short_name(f) -> str:
    return getattr(f, "__qualname__", getattr(f, "__name__", repr(f))).split(".")[-1]


def pair(r) -> list[float]:
    lo, hi = r
    return [float(lo), float(hi)]


def resolve(patterns: dict[str, float], names: list[str], what: str) -> list[float]:
    out = []
    for n in names:
        hits = [v for p, v in patterns.items() if re.fullmatch(p, n)]
        if len(hits) != 1:
            raise SystemExit(f"{what}: joint {n} matches {len(hits)} patterns in {patterns}")
        out.append(float(hits[0]))
    return out


def match(patterns, names: list[str]) -> list[str]:
    if isinstance(patterns, str):
        patterns = (patterns,)
    return [n for n in names if any(re.fullmatch(p, n) for p in patterns)]


def stages(entries: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps = [int(e["step"]) for e in entries]
    if steps != sorted(steps):
        raise SystemExit(f"curriculum stages out of order: {steps}")
    vals = []
    for e in entries:
        v = e[key]
        vals.extend(float(x) for x in np.asarray(v, dtype=float).ravel())
    return steps, vals


def export(out_path: str) -> dict:
    import mujoco
    from bam.model import load_model
    from mjlab.scene import Scene
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

    cfg = make_microduck_velocity_env_cfg()
    cfg.scene.num_envs = 1
    robot_cfg = cfg.scene.entities["robot"]
    scene = Scene(cfg.scene, device="cpu")
    model = scene.spec.compile()

    def jname(j: int) -> str:
        return model.joint(j).name.removeprefix("robot/")

    # Action order: mjlab's JointPositionAction over actuators '.*' in actuator (= entity) order.
    act_joints = [jname(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    if any(n.startswith("passive_") for n in act_joints):
        raise SystemExit(f"passive joints are actuated: {act_joints}")
    default_pos = resolve(robot_cfg.init_state.joint_pos, act_joints, "init_state.joint_pos")
    # The C# side applies target = default + action - encoder_bias.
    if list(cfg.actions) != ["joint_pos"] or type(cfg.actions["joint_pos"]).__name__ != "JointPositionActionCfg":
        raise SystemExit(f"expected one JointPositionAction, got {list(cfg.actions)}")
    act = cfg.actions["joint_pos"]
    if (not act.use_default_offset or act.scale not in (1, 1.0) or act.offset not in (0, 0.0)
            or act.clip is not None or match(act.actuator_names, act_joints) != act_joints):
        raise SystemExit(f"joint_pos action not ported: scale {act.scale}, offset {act.offset}, "
                         f"clip {act.clip}, default offset {act.use_default_offset}")
    factor = float(robot_cfg.articulation.soft_joint_pos_limit_factor)
    soft_lo, soft_hi = [], []
    for n in act_joints:
        j = model.joint("robot/" + n).id
        lo, hi = model.jnt_range[j]
        if not model.jnt_limited[j]:
            lo, hi = -math.inf, math.inf
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
        soft_lo.append(float(mid - half))
        soft_hi.append(float(mid + half))

    # ---------------------------------------------------------------- actuator
    acts = robot_cfg.articulation.actuators
    if len(acts) != 1 or type(acts[0]).__name__ not in ("FrictionDRBamActuatorCfg", "BamActuatorCfg"):
        raise SystemExit(f"expected one BAM actuator group, got {[type(a).__name__ for a in acts]}")
    a = acts[0]
    if match(a.target_names_expr, act_joints) != act_joints:
        raise SystemExit(f"BAM actuator targets {a.target_names_expr} don't cover {act_joints}")
    bam = load_model(a._resolved_json_path)
    ba = bam.actuator
    if a.kp_fw is not None:
        ba.kp = a.kp_fw
    if a.vin is not None:
        ba.vin = a.vin
    if type(ba).__name__ != "VoltageControlledActuator" and not hasattr(ba, "error_gain"):
        raise SystemExit(f"BAM actuator {type(ba).__name__} has no voltage control law")
    for flag in ("stribeck", "load_dependent", "directional", "quadratic"):
        if not getattr(bam, flag):
            raise SystemExit(f"C# BAM implements m6 only; model lacks {flag}")
    max_current = getattr(ba, "max_current", None)
    vin_range = pair(a.vin_range) if a.vin_range is not None else [float(ba.vin)] * 2
    actuator = {
        "model": bam.name,
        "friction_dr": type(a).__name__ == "FrictionDRBamActuatorCfg",
        "kt": bam.kt.value, "R": bam.R.value,
        "friction_base": bam.friction_base.value,
        "friction_stribeck": bam.friction_stribeck.value,
        "load_friction_motor": bam.load_friction_motor.value,
        "load_friction_external": bam.load_friction_external.value,
        "load_friction_motor_stribeck": bam.load_friction_motor_stribeck.value,
        "load_friction_external_stribeck": bam.load_friction_external_stribeck.value,
        "load_friction_motor_quad": bam.load_friction_motor_quad.value,
        "load_friction_external_quad": bam.load_friction_external_quad.value,
        "dtheta_stribeck": bam.dtheta_stribeck.value,
        "alpha": bam.alpha.value,
        "friction_viscous": bam.friction_viscous.value,
        "kp": float(ba.kp),
        "error_gain": float(ba.error_gain),
        "max_pwm": float(ba.max_pwm),
        "max_current": float(max_current) if max_current is not None else 0.0,
        "vin_range": vin_range,
        "vin_drop_gain_range": pair(a.vin_drop_gain_range) if a.vin_drop_gain_range is not None else [0.0, 0.0],
        "vin_min": float(a.vin_min) if a.vin_min is not None else 0.0,
        "delay": {
            "min_lag": a.delay_min_lag, "max_lag": a.delay_max_lag,
            "hold_prob": a.delay_hold_prob, "update_period": a.delay_update_period,
            "per_env_phase": a.delay_per_env_phase,
        },
    }

    # ---------------------------------------------------------------- commands
    tw = cfg.commands["twist"]
    if tw.rel_heading_envs > 0 or tw.rel_world_envs > 0 or tw.init_velocity_prob > 0:
        raise SystemExit("heading / world-frame / init-velocity twist envs are not ported")
    twist = {
        "resampling_time": pair(tw.resampling_time_range),
        "lin_vel_x": pair(tw.ranges.lin_vel_x),
        "lin_vel_y": pair(tw.ranges.lin_vel_y),
        "ang_vel_z": pair(tw.ranges.ang_vel_z),
        "rel_standing_envs": float(tw.rel_standing_envs),
        "rel_forward_envs": float(tw.rel_forward_envs),
        "rel_turn_in_place_envs": float(getattr(tw, "rel_turn_in_place_envs", 0.0)),
        "turn_in_place_min_frac": float(getattr(tw, "turn_in_place_min_frac", 0.4)),
    }

    def pose_cmd(name: str) -> dict:
        c = cfg.commands[name]
        if type(c).__name__ != "UniformPoseCommandCfg":
            raise SystemExit(f"command {name}: {type(c).__name__} not ported")
        return {"resampling_time": pair(c.resampling_time_range),
                "lo": [float(r[0]) for r in c.ranges], "hi": [float(r[1]) for r in c.ranges],
                "zero_command_prob": float(c.zero_command_prob)}

    head_pose, body_pose = pose_cmd("head_pose"), pose_cmd("body_pose")
    if len(head_pose["lo"]) != 4 or len(body_pose["lo"]) != 6:
        raise SystemExit("head_pose must be 4-D and body_pose 6-D")
    from mjlab_microduck.tasks import mdp as duck_mdp
    neck_joints = [match(p, act_joints)[0] for p in duck_mdp._NECK_JOINT_PATTERNS]

    # ---------------------------------------------------------------- observations
    obs_kinds = {
        "base_ang_vel_imu_misaligned": "imu_gyro", "projected_gravity_imu_misaligned": "imu_gravity",
        "joint_pos_rel": "joint_pos", "joint_vel_rel": "joint_vel", "last_action": "last_action",
        "base_lin_vel": "base_lin_vel", "projected_gravity": "gravity",
        "foot_height_safe": "foot_height", "foot_air_time_safe": "foot_air_time",
        "foot_contact": "foot_contact", "foot_contact_forces_safe": "foot_contact_forces",
    }

    def obs_group(name: str) -> tuple[list[dict], float]:
        g = cfg.observations[name]
        if g.history_length:
            raise SystemExit(f"obs group {name}: history not ported")
        terms, misalign = [], 0.0
        for tname, t in g.terms.items():
            fname = short_name(t.func)
            if fname == "generated_commands":
                kind = {"twist": "twist_command", "head_pose": "head_command",
                        "body_pose": "body_command"}[t.params["command_name"]]
            elif fname == "builtin_sensor":
                if t.params["sensor_name"] != "robot/imu_ang_vel":
                    raise SystemExit(f"obs {name}/{tname}: sensor {t.params['sensor_name']} not ported")
                kind = "gyro"
            elif fname in obs_kinds:
                kind = obs_kinds[fname]
            else:
                raise SystemExit(f"obs {name}/{tname}: {qualname(t.func)} not ported")
            if t.clip is not None or (t.scale not in (None, 1.0)) or t.history_length:
                raise SystemExit(f"obs {name}/{tname}: clip/scale/history not ported")
            if kind in ("joint_pos", "joint_vel"):
                jn = t.params["asset_cfg"].joint_names
                if match(jn, act_joints) != act_joints:
                    raise SystemExit(f"obs {name}/{tname}: joints {jn} are not the actuated joints")
            if kind in ("imu_gyro", "imu_gravity"):
                misalign = max(misalign, float(t.params.get("max_angle_deg", 0.0)))
            noise = [0.0, 0.0]
            if g.enable_corruption and t.noise is not None:
                if type(t.noise).__name__ != "UniformNoiseCfg" or t.noise.operation != "add":
                    raise SystemExit(f"obs {name}/{tname}: noise {t.noise} not ported")
                noise = [float(t.noise.n_min), float(t.noise.n_max)]
            terms.append({
                "name": tname, "kind": kind,
                "biased": bool(t.params.get("biased", False)),
                "noise": noise,
                "delay": {"min_lag": t.delay_min_lag, "max_lag": t.delay_max_lag,
                          "hold_prob": t.delay_hold_prob, "update_period": t.delay_update_period,
                          "per_env_phase": t.delay_per_env_phase},
            })
        return terms, misalign

    actor, misalign = obs_group("actor")
    critic, _ = obs_group("critic")

    # ---------------------------------------------------------------- curriculum
    def stage_gate(func) -> bool:
        """True when a stage applies at common_step_counter >= step, False for > step."""
        src = inspect.getsource(func)
        ge, gt = "common_step_counter >= stage" in src, "common_step_counter > stage" in src
        if ge == gt:
            raise SystemExit(f"{qualname(func)}: can't tell how stages gate on common_step_counter")
        return ge

    curriculum, touched_rewards = [], set()
    for cname, c in cfg.curriculum.items():
        fname, p = short_name(c.func), c.params
        if fname == "reward_weight":
            steps, vals = stages(p["weight_stages"], "weight")
            curriculum.append({"name": cname, "kind": "reward_weight", "target": p["reward_name"],
                               "steps": steps, "values": vals, "width": 1})
            touched_rewards.add(p["reward_name"])
        elif fname == "standing_envs_curriculum":
            steps, vals = stages(p["standing_stages"], "rel_standing_envs")
            curriculum.append({"name": cname, "kind": "standing_envs", "target": p["command_name"],
                               "steps": steps, "values": vals, "width": 1})
        elif fname == "pose_command_range_curriculum":
            entries = p["range_stages"]
            steps = [int(e["step"]) for e in entries]
            vals = [float(x) for e in entries for r in e["ranges"] for x in r]
            curriculum.append({"name": cname, "kind": "pose_range", "target": p["command_name"],
                               "steps": steps, "values": vals, "width": 2 * len(entries[0]["ranges"])})
        elif fname == "com_range_curriculum":
            steps, vals = stages(p["range_stages"], "range")
            curriculum.append({"name": cname, "kind": "com_range", "target": p["event_name"],
                               "steps": steps, "values": vals, "width": 1})
        else:
            raise SystemExit(f"curriculum {cname}: {qualname(c.func)} not ported")
        curriculum[-1]["inclusive"] = stage_gate(c.func)

    # ---------------------------------------------------------------- rewards
    reward_kinds = {
        "track_linear_velocity": "track_linear_velocity",
        "track_angular_velocity": "track_angular_velocity",
        "upright": "upright", "variable_posture": "variable_posture",
        "body_angular_velocity_penalty": "body_ang_vel", "angular_momentum_penalty": "angular_momentum",
        "joint_pos_limits": "joint_pos_limits", "action_rate_l2": "action_rate_l2",
        "feet_air_time": "feet_air_time", "feet_clearance": "feet_clearance",
        "feet_swing_height": "feet_swing_height", "feet_slip": "feet_slip",
        "self_collision_cost": "self_collisions", "head_pose_tracking": "head_pose_tracking",
        "head_pose_bias_penalty": "head_pose_bias",
    }
    rewards = []
    for rname, r in cfg.rewards.items():
        fname, p = short_name(r.func), dict(r.params)
        kind = reward_kinds.get(fname)
        if kind is None:
            if r.weight == 0.0 and rname not in touched_rewards:
                rewards.append({"name": rname, "kind": "disabled", "weight": 0.0})
                continue
            raise SystemExit(f"reward {rname}: {qualname(r.func)} not ported")
        term = {"name": rname, "kind": kind, "weight": float(r.weight)}
        for k in ("std", "threshold_min", "threshold_max", "command_threshold", "target_height",
                  "tau_s", "walking_threshold", "running_threshold"):
            if k in p:
                term[k] = float(p[k])
        if p.get("fine_std") is not None or p.get("gate_height_low") is not None:
            raise SystemExit(f"reward {rname}: fine_std / height gate not ported")
        if "asset_cfg" in p and p["asset_cfg"].body_names not in (None, ("trunk_base",)):
            raise SystemExit(f"reward {rname}: bodies {p['asset_cfg'].body_names} not ported")
        if "asset_cfg" in p and p["asset_cfg"].site_names not in (None, ["left_foot", "right_foot"],
                                                                  ("left_foot", "right_foot")):
            raise SystemExit(f"reward {rname}: sites {p['asset_cfg'].site_names} not ported")
        if kind == "variable_posture":
            joints = match(p["asset_cfg"].joint_names, act_joints)
            term["joint_mask"] = [1 if n in joints else 0 for n in act_joints]
            for regime in ("standing", "walking", "running"):
                std = resolve(p[f"std_{regime}"], joints, f"{rname}.std_{regime}")
                full = dict(zip(joints, std))
                term[f"std_{regime}"] = [full.get(n, 1.0) for n in act_joints]
        if kind == "joint_pos_limits":
            jn = p.get("asset_cfg").joint_names if "asset_cfg" in p else None
            if jn not in (None, (".*",)) and match(jn, act_joints) != act_joints:
                raise SystemExit(f"reward {rname}: joint subset {jn} not ported")
        rewards.append(term)

    # ---------------------------------------------------------------- terminations
    terminations = []
    for tname, t in cfg.terminations.items():
        fname = short_name(t.func)
        if fname == "time_out":
            terminations.append({"name": tname, "kind": "time_out", "time_out": True})
        elif fname == "bad_orientation":
            terminations.append({"name": tname, "kind": "bad_orientation", "time_out": bool(t.time_out),
                                 "limit_angle": float(t.params["limit_angle"])})
        elif fname == "robot_state_is_nan":
            terminations.append({"name": tname, "kind": "nan_state", "time_out": bool(t.time_out)})
        elif fname == "out_of_terrain_bounds":
            if cfg.scene.terrain.terrain_type != "plane":
                raise SystemExit("out_of_terrain_bounds only ported for the infinite plane")
            continue  # never fires on an infinite plane
        else:
            raise SystemExit(f"termination {tname}: {qualname(t.func)} not ported")

    # ---------------------------------------------------------------- events
    ev = {}
    for ename, e in cfg.events.items():
        fname, p = short_name(e.func), e.params
        if fname == "reset_root_state_uniform":
            pr = p["pose_range"]
            if p.get("velocity_range"):
                raise SystemExit("reset velocity_range not ported")
            ev["reset_base"] = {k: pair(pr.get(k, (0.0, 0.0))) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        elif fname == "reset_joints_by_offset":
            ev["reset_joints"] = {"position_range": pair(p["position_range"]),
                                  "velocity_range": pair(p["velocity_range"])}
        elif fname == "push_by_setting_velocity":
            vr = p["velocity_range"]
            extra = set(vr) - {"x", "y"}
            if extra:
                raise SystemExit(f"push axes {extra} not ported")
            ev["push"] = {"interval": pair(e.interval_range_s), "x": pair(vr.get("x", (0, 0))),
                          "y": pair(vr.get("y", (0, 0)))}
            if e.is_global_time or e.min_step_count_between_reset:
                raise SystemExit("push: global time / min step count not ported")
        elif fname == "geom_friction":
            geoms = list(p["asset_cfg"].geom_names)
            if p["operation"] != "abs" or p.get("axes") not in (None, [0]):
                raise SystemExit(f"geom_friction {p['operation']} axes {p.get('axes')} not ported")
            ev["foot_friction"] = {"geoms": geoms, "range": pair(p["ranges"]),
                                   "shared": bool(p.get("shared_random", False)), "mode": e.mode}
        elif fname == "encoder_bias":
            ev["encoder_bias"] = {"range": pair(p["bias_range"]), "mode": e.mode}
        elif fname == "body_com_offset":
            bodies = list(p["asset_cfg"].body_names or ())
            if not bodies:
                continue  # base_com: no bodies selected
            if p["operation"] != "add":
                raise SystemExit(f"{ename}: operation {p['operation']} not ported")
            ranges = p["ranges"]
            if isinstance(ranges, dict):
                raise SystemExit(f"{ename}: per-axis ranges not ported")
            ev.setdefault("com", []).append({"name": ename, "bodies": bodies, "range": pair(ranges),
                                             "mode": e.mode})
        elif fname == "pseudo_inertia":
            nonzero = [k for k, v in p.items() if k.endswith("_range") and k != "alpha_range"
                       and v is not None and tuple(v) != (0.0, 0.0)]
            if nonzero or p.get("distribution", "uniform") != "uniform":
                raise SystemExit(f"pseudo_inertia: {nonzero} / {p.get('distribution')} not ported")
            ev["mass_inertia"] = {"bodies": list(p["asset_cfg"].body_names), "alpha": pair(p["alpha_range"]),
                                  "mode": e.mode}
        elif fname == "randomize_bam_friction":
            ev["bam_friction"] = {"range": pair(p["scale_range"]), "mode": e.mode}
        elif fname == "joint_armature":
            if p["operation"] != "scale" or match(p["asset_cfg"].joint_names, act_joints) != act_joints:
                raise SystemExit(f"joint_armature {p['operation']} {p['asset_cfg'].joint_names} not ported")
            ev["armature"] = {"range": pair(p["ranges"]), "mode": e.mode}
        elif fname in ("expand_bam_friction_fields", "reset_action_history"):
            continue
        else:
            raise SystemExit(f"event {ename}: {qualname(e.func)} not ported")
    # Head-CoM body patterns like '(bottom_head_shell|jaw_soft)' resolve against model bodies.
    body_names = [model.body(b).name.removeprefix("robot/") for b in range(model.nbody)]
    for c in ev.get("com", []):
        c["bodies"] = [b for pat in c["bodies"] for b in body_names if re.fullmatch(pat, b)]
    ev["mass_inertia"]["bodies"] = [b for pat in ev["mass_inertia"]["bodies"]
                                    for b in body_names if re.fullmatch(pat, b)]
    for c in curriculum:
        if c["kind"] == "com_range" and c["target"] not in [x["name"] for x in ev.get("com", [])]:
            raise SystemExit(f"curriculum {c['name']}: event {c['target']} missing")

    # Foot height scan: on the infinite plane every downward ray from the foot site hits z = 0,
    # so the C# side reduces it to clamp(site_z, 0, max_distance); refuse anything else.
    scans = [s for s in cfg.scene.sensors if s.name == "foot_height_scan"]
    if len(scans) != 1:
        raise SystemExit("expected one foot_height_scan sensor")
    scan = scans[0]
    if (cfg.scene.terrain.terrain_type != "plane" or scan.ray_alignment != "yaw"
            or type(scan).__name__ != "TerrainHeightSensorCfg"):
        raise SystemExit("foot_height_scan: only yaw-aligned scans over the plane are ported")
    foot_sites = [f.name for f in scan.frame]
    if any(f.type != "site" for f in scan.frame):
        raise SystemExit("foot_height_scan frames must be sites")
    for r in rewards:
        if r["kind"] in ("feet_clearance", "feet_slip"):
            sites = cfg.rewards[r["name"]].params["asset_cfg"].site_names
            if list(sites) != foot_sites:
                raise SystemExit(f"reward {r['name']}: sites {sites} differ from the scan {foot_sites}")

    o = cfg.sim.mujoco
    spec = {
        "task": TASK,
        "decimation": int(cfg.decimation),
        "physics_dt": float(o.timestep),
        "episode_length_s": float(cfg.episode_length_s),
        "scale_rewards_by_dt": bool(getattr(cfg, "scale_rewards_by_dt", True)),
        "joint_names": act_joints,
        "default_joint_pos": default_pos,
        "soft_limit_lo": soft_lo,
        "soft_limit_hi": soft_hi,
        "neck_joints": neck_joints,
        "foot_sites": foot_sites,
        "foot_height_max_distance": float(scan.max_distance),
        "imu_misalign_deg": misalign,
        "actuator": actuator,
        "twist": twist,
        "head_pose": head_pose,
        "body_pose": body_pose,
        "actor": actor,
        "critic": critic,
        "rewards": rewards,
        "terminations": terminations,
        "events": ev,
        "curriculum": curriculum,
    }
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=1)
    return spec


if __name__ == "__main__":
    s = export(sys.argv[1] if len(sys.argv) > 1 else "flat_task_spec.json")
    print(f"{s['task']}: actor {len(s['actor'])} terms, critic {len(s['critic'])}, "
          f"rewards {len(s['rewards'])}, curriculum {len(s['curriculum'])}")
