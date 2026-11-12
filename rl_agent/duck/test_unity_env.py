"""Tests for the Unity duck env server client (duck/unity_env.py).

The protocol tests use an in-process fake server. The server tests run the real Linux
build and are skipped when it isn't mounted at DUCK_UNITY_SERVER_DIR.
"""
import json
import os
import pathlib
import socket
import struct
import sys
import tempfile
import threading

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from duck import unity_env  # noqa: E402

FIELDS = [{"name": "obs/policy", "dtype": "f32", "width": 3},
          {"name": "reward", "dtype": "f32", "width": 1},
          {"name": "terminated", "dtype": "u8", "width": 1},
          {"name": "episode_length", "dtype": "i32", "width": 1}]


def fake_server(path, num_envs=2, act_dim=2, version=unity_env.PROTOCOL_VERSION):
    """Serves one client: every reply's obs are the summed actions plus the step count."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(1)

    def run():
        conn, _ = listener.accept()
        header = json.dumps({"num_envs": num_envs, "act_dim": act_dim, "control_dt": 0.02,
                             "episode_steps": 500, "fields": FIELDS}).encode()
        conn.sendall(b"DUCK" + struct.pack("<ii", version, len(header)) + header)
        steps = 0
        while True:
            try:
                cmd = conn.recv(1)
            except ConnectionResetError:
                break
            if not cmd or cmd[0] == unity_env.CLOSE:
                break
            if cmd[0] == unity_env.GLOBAL_STEP:
                conn.recv(8)
                a = np.zeros((num_envs, act_dim), np.float32)
            elif cmd[0] == unity_env.STEP:
                raw = b""
                while len(raw) < num_envs * act_dim * 4:
                    raw += conn.recv(num_envs * act_dim * 4 - len(raw))
                a = np.frombuffer(raw, np.float32).reshape(num_envs, act_dim)
                steps += 1
            else:
                if cmd[0] == unity_env.RESET:
                    conn.recv(8)
                    steps = 0
                a = np.zeros((num_envs, act_dim), np.float32)
            obs = np.repeat(a.sum(1, keepdims=True), 3, 1).astype(np.float32)
            conn.sendall(obs.tobytes() + np.full(num_envs, steps, np.float32).tobytes()
                         + np.arange(num_envs, dtype=np.uint8).tobytes()
                         + np.full(num_envs, 7 + steps, np.int32).tobytes())
        conn.close()
        listener.close()

    threading.Thread(target=run, daemon=True).start()


def test_client_parses_header_and_typed_fields():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s")
        fake_server(path)
        c = unity_env.DuckClient(unix=path)
        assert (c.num_envs, c.act_dim, c.control_dt) == (2, 2, 0.02)
        assert c.fields == {"obs/policy": 3, "reward": 1, "terminated": 1, "episode_length": 1}
        out = c.reset(5)
        assert out["obs/policy"].shape == (2, 3) and out["reward"].shape == (2,)
        out = c.step(np.array([[1, 2], [3, 4]], np.float32))
        np.testing.assert_array_equal(out["obs/policy"][:, 0], [3, 7])
        np.testing.assert_array_equal(out["terminated"], [0, 1])
        assert out["episode_length"].dtype == np.int32 and out["episode_length"][0] == 8
        with pytest.raises(ValueError):
            c.step(np.zeros((3, 2)))
        out = c.set_global_step(12345)
        assert out["obs/policy"].shape == (2, 3)
        c.close()


def test_client_rejects_other_protocol_versions():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s")
        fake_server(path, version=2)
        with pytest.raises(ConnectionError, match="v2"):
            unity_env.DuckClient(unix=path)


def test_task_spec_is_the_exported_velocity_flat_task():
    spec = unity_env.load_task_spec()
    assert spec["task"] == "Mjlab-Velocity-Flat-MicroDuck"
    assert len(spec["joint_names"]) == 14 and len(spec["rewards"]) == 16


needs_build = pytest.mark.skipif(
    not os.path.exists(os.path.join(unity_env.SERVER_DIR, unity_env.EXECUTABLE)),
    reason=f"no duck server build at {unity_env.SERVER_DIR}")


def rollout(env, steps, seed=3):
    rng = np.random.default_rng(seed)
    out = env.reset()
    trace = [out["obs/policy"].copy()]
    for _ in range(steps):
        out = env.step(rng.normal(0, 0.3, (env.num_envs, env.act_dim)))
        trace.append(np.concatenate([out["obs/policy"], out["obs/critic"], out["reward"][:, None],
                                     out["reward_terms"], out["terminated"][:, None],
                                     out["truncated"][:, None]], 1).copy())
    return trace


@needs_build
def test_results_do_not_depend_on_thread_count():
    with unity_env.DuckServer(num_envs=24, seed=11, threads=1) as one:
        a = rollout(one, 60)
    with unity_env.DuckServer(num_envs=24, seed=11, threads=5) as many:
        b = rollout(many, 60)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@needs_build
def test_reset_with_the_same_seed_replays_and_another_seed_differs():
    with unity_env.DuckServer(num_envs=8, seed=1) as env:
        first = rollout(env, 20)
        again = rollout(env, 20)
        env.seed = 2
        other = rollout(env, 20)
    for x, y in zip(first, again):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(first[0], other[0])


@needs_build
def test_header_and_reply_layout():
    spec = unity_env.load_task_spec()
    with unity_env.DuckServer(num_envs=4, seed=1) as env:
        assert env.act_dim == 14 and env.control_dt == pytest.approx(0.02)
        assert env.fields == {"obs/policy": 61, "obs/critic": 76, "reward": 1, "terminated": 1,
                              "truncated": 1, "episode_length": 1, "reward_terms": 16,
                              "terminations": len(spec["terminations"])}
        assert env.reward_names == [r["name"] for r in spec["rewards"]]
        assert env.termination_names == [t["name"] for t in spec["terminations"]]
        assert [t[0] for t in env.header["actor_terms"]] == [t["name"] for t in spec["actor"]]
        out = env.reset()
        # Standing at reset: gravity points down the trunk z axis (the actor's copy is seen
        # through an IMU misaligned by up to imu_misalign_deg, plus noise).
        names, widths = zip(*env.header["critic_terms"])
        gravity_z = int(np.cumsum((0,) + widths)[names.index("projected_gravity")]) + 2
        np.testing.assert_allclose(out["obs/critic"][:, gravity_z], -1, atol=1e-3)
        np.testing.assert_allclose(out["obs/policy"][:, 5], -1, atol=0.03)


@needs_build
def test_randomized_physics_changes_the_rollout():
    with unity_env.DuckServer(num_envs=8, seed=4) as env:
        nominal = rollout(env, 30)
    spec = unity_env.load_task_spec()
    spec["events"]["bam_friction"]["range"] = [3.0, 3.0]
    with unity_env.DuckServer(num_envs=8, seed=4, task=spec) as env:
        heavy = rollout(env, 30)
    np.testing.assert_array_equal(nominal[0], heavy[0])
    assert np.abs(nominal[-1] - heavy[-1]).max() > 1e-3


@needs_build
def test_curriculum_follows_the_global_step():
    spec = unity_env.load_task_spec()
    rate = next(c for c in spec["curriculum"] if c["name"] == "action_rate_weight")
    i = [r["name"] for r in spec["rewards"]].index(rate["target"])
    a = np.full((4, 14), 0.1, np.float32)
    weights = []
    for step in (0, rate["steps"][-1] + 1):
        with unity_env.DuckServer(num_envs=4, seed=1, global_step=step) as env:
            env.reset()
            env.step(a)
            out = env.step(-a)
            # action_rate_l2 is sum((0.2)^2) over 14 joints = 0.56 before the weight.
            weights.append(out["reward_terms"][:, i] / 0.56)
    np.testing.assert_allclose(weights[1], rate["values"][-1], rtol=1e-4)
    assert not np.allclose(weights[0], weights[1])


@needs_build
def test_episodes_truncate_and_reset():
    spec = unity_env.load_task_spec()
    spec["episode_length_s"] = 10 * 0.02
    spec["terminations"] = [t for t in spec["terminations"] if t["kind"] != "bad_orientation"]
    with unity_env.DuckServer(num_envs=4, seed=1, task=spec) as env:
        env.reset()
        zeros = np.zeros((4, 14), np.float32)
        for k in range(1, 10):
            out = env.step(zeros)
            assert not out["truncated"].any() and (out["episode_length"] == k).all()
        out = env.step(zeros)
        assert out["truncated"].all() and not out["terminated"].any()
        assert (out["episode_length"] == 10).all()
        time_out = env.termination_names.index("time_out")
        assert out["terminations"][:, time_out].all() and out["terminations"].sum() == 4
        out = env.step(zeros)
        assert (out["episode_length"] == 1).all() and not out["terminations"].any()


@needs_build
def test_episode_lengths_can_be_set():
    spec = unity_env.load_task_spec()
    spec["episode_length_s"] = 10 * 0.02
    spec["terminations"] = [t for t in spec["terminations"] if t["kind"] != "bad_orientation"]
    with unity_env.DuckServer(num_envs=4, seed=1, task=spec) as env:
        env.reset()
        out = env.set_episode_lengths([0, 3, 8, 9])
        np.testing.assert_array_equal(out["episode_length"], [0, 3, 8, 9])
        out = env.step(np.zeros((4, 14), np.float32))
        np.testing.assert_array_equal(out["truncated"], [0, 0, 0, 1])
        np.testing.assert_array_equal(out["episode_length"], [1, 4, 9, 10])


@needs_build
def test_server_exits_when_the_client_disconnects():
    env = unity_env.DuckServer(num_envs=2, seed=1)
    env.reset()
    env.sock.close()
    env.sock = None
    assert env.proc.wait(30) == 0
    env.close()
