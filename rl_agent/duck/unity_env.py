"""Client and launcher for the Unity duck env server (DuckEnvServer.cs, protocol v4).

The server is the MicroDuck Linux player run headless with ``-server``: N MuJoCo ducks
running the mjlab Velocity-Flat task, stepped across CPU threads, one binary request/reply
per control step. ``DuckServer`` starts it as a child process of the duck worker and
connects over a Unix socket::

    with DuckServer(num_envs=4096, seed=1) as env:
        out = env.reset()
        out = env.step(actions)          # float32 [num_envs, act_dim]
        out["obs/policy"], out["obs/critic"], out["reward"], out["terminated"], out["truncated"]
        out["reward_terms"]              # [num_envs, len(env.reward_names)]
        out["terminations"]              # [num_envs, len(env.termination_names)], 0/1

The task is the JSON written by export_task_spec.py (flat_task_spec.json next to this file,
also shipped inside the build); pass ``task=`` to run a modified copy (see load_task_spec).

The build is bind-mounted read-only at ``DUCK_UNITY_SERVER_DIR`` (compose/duck.yml) and
copied to local disk before it runs, since it can't execute from the Windows mount.
"""
import ctypes
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np

PROTOCOL_VERSION = 4
STEP, RESET, ECHO, CLOSE, GLOBAL_STEP, EVAL, EPISODE_LENGTH = 1, 2, 3, 4, 5, 6, 7
DTYPES = {"f32": np.float32, "u8": np.uint8, "i32": np.int32}
TASK_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flat_task_spec.json")


def load_task_spec(path=TASK_SPEC):
    """The exported task as a dict, to modify and pass as DuckServer(task=...)."""
    with open(path) as f:
        return json.load(f)

SERVER_DIR = os.environ.get("DUCK_UNITY_SERVER_DIR", "/unity_builds/duck_server")
INSTALL_DIR = os.environ.get("DUCK_UNITY_INSTALL_DIR", "/opt/duckserver")
EXECUTABLE = "DuckServer.x86_64"
# Build output that never needs to run: IL2CPP symbols, several times the size of the player.
SKIP_DIRS = ("_BackUpThisFolder_ButDontShipItWithYourGame",)


class DuckClient:
    """One connection to a running server. Replies are dicts of numpy arrays shaped
    [num_envs, width] (width-1 fields are flattened to [num_envs]); they are views into a
    buffer the next call overwrites, so copy anything kept across calls."""

    def __init__(self, tcp=None, unix=None, timeout=600.0):
        if unix:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(unix)
        else:
            host, port = tcp.rsplit(":", 1)
            self.sock = socket.create_connection((host, int(port)), timeout=timeout)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)

        magic, version, length = struct.unpack("<4sii", self._read(12))
        if magic != b"DUCK":
            raise ConnectionError(f"not a duck env server (magic {magic!r})")
        if version != PROTOCOL_VERSION:
            raise ConnectionError(f"server speaks protocol v{version}, this client v{PROTOCOL_VERSION}")
        self.header = json.loads(self._read(length).decode("utf-8"))
        self.num_envs = self.header["num_envs"]
        self.act_dim = self.header["act_dim"]
        self.control_dt = self.header["control_dt"]
        self.reward_names = self.header.get("reward_names", [])
        self.termination_names = self.header.get("termination_names", [])
        self.episode_steps = self.header["episode_steps"]

        self._layout = []
        offset = 0
        for f in self.header["fields"]:
            dtype = np.dtype(DTYPES[f["dtype"]])
            count = self.num_envs * f["width"]
            self._layout.append((f["name"], dtype, f["width"], offset, count))
            offset += count * dtype.itemsize
        self._reply = bytearray(offset)
        self._view = memoryview(self._reply)

    @property
    def fields(self):
        return {name: width for name, _, width, _, _ in self._layout}

    def _read(self, size):
        buf = bytearray(size)
        self._read_into(memoryview(buf))
        return bytes(buf)

    def _read_into(self, view):
        got = 0
        while got < len(view):
            n = self.sock.recv_into(view[got:])
            if n == 0:
                raise ConnectionError("duck env server closed the connection")
            got += n

    def _call(self, payload):
        self.sock.sendall(payload)
        self._read_into(self._view)
        out = {}
        for name, dtype, width, offset, count in self._layout:
            a = np.frombuffer(self._reply, dtype, count, offset)
            out[name] = a if width == 1 else a.reshape(self.num_envs, width)
        return out

    def step(self, actions):
        a = np.ascontiguousarray(actions, dtype=np.float32)
        if a.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"actions must be [{self.num_envs}, {self.act_dim}], got {list(a.shape)}")
        return self._call(bytes([STEP]) + a.tobytes())

    def reset(self, seed):
        """Reseed every duck's random stream from ``seed`` and reset them all."""
        return self._call(bytes([RESET]) + struct.pack("<Q", seed))

    def echo(self):
        return self._call(bytes([ECHO]))

    def set_global_step(self, step):
        """Set the curriculum's global step counter (mjlab's common_step_counter), to resume."""
        return self._call(bytes([GLOBAL_STEP]) + struct.pack("<q", step))

    def set_episode_lengths(self, lengths):
        """Set every duck's episode step counter (rsl_rl's init_at_random_ep_len)."""
        a = np.ascontiguousarray(lengths, dtype=np.int32).reshape(self.num_envs)
        return self._call(bytes([EPISODE_LENGTH]) + a.tobytes())

    @property
    def eval_layout(self):
        return [(name, width) for name, width in self.header.get("eval_layout", [])]

    def evaluate(self, states):
        """Rewards, terminations and observations on given states, without stepping.
        ``states`` maps every eval_layout name to an array [num_envs, width]."""
        cols = [np.asarray(states[name], np.float64).reshape(self.num_envs, width)
                for name, width in self.eval_layout]
        return self._call(bytes([EVAL]) + np.ascontiguousarray(np.concatenate(cols, 1)).tobytes())

    def close(self):
        """Tell the server to exit, then drop the connection."""
        try:
            self.sock.sendall(bytes([CLOSE]))
        except OSError:
            pass
        self.sock.close()


def install(src=SERVER_DIR, dst=INSTALL_DIR):
    """Copy the server build to local disk unless the copy there is already current.
    Returns the executable's path."""
    stamp_src = os.path.join(src, "GameAssembly.so")
    if not os.path.exists(os.path.join(src, EXECUTABLE)) or not os.path.exists(stamp_src):
        raise FileNotFoundError(f"no duck server build in {src} (expected {EXECUTABLE} and GameAssembly.so)")
    st = os.stat(stamp_src)
    stamp = f"{st.st_size} {int(st.st_mtime)}"
    stamp_file = os.path.join(dst, ".installed_from")
    if os.path.exists(stamp_file) and open(stamp_file).read() == stamp:
        return os.path.join(dst, EXECUTABLE)

    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(*(f"*{d}" for d in SKIP_DIRS)))
    os.chmod(os.path.join(tmp, EXECUTABLE), 0o755)
    with open(os.path.join(tmp, ".installed_from"), "w") as f:
        f.write(stamp)
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)
    return os.path.join(dst, EXECUTABLE)


def _die_with_parent():
    # Linux PR_SET_PDEATHSIG: the server gets SIGKILL if the worker dies without closing it.
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)


class DuckServer(DuckClient):
    """Starts a server process, connects to it, and stops it on close() or context exit."""

    def __init__(self, num_envs=4096, seed=1, threads=None, task=None, log_file=None,
                 executable=None, start_timeout=180.0, global_step=0, extra_args=()):
        self.executable = executable or install()
        self._tmpdir = tempfile.mkdtemp(prefix="duckserver-")
        self.socket_path = os.path.join(self._tmpdir, "env.sock")
        self.log_file = log_file or os.path.join(self._tmpdir, "server.log")
        self.seed = seed
        argv = [self.executable, "-batchmode", "-nographics", "-server",
                "-ducks", str(num_envs), "-seed", str(seed),
                "-unix", self.socket_path, "-exit-on-disconnect",
                "-accept-timeout", str(int(start_timeout)), "-logFile", self.log_file]
        if threads:
            argv += ["-threads", str(threads)]
        if task is not None:
            task_path = os.path.join(self._tmpdir, "task.json")
            with open(task_path, "w") as f:
                json.dump(task, f)
            argv += ["-task", task_path]
        if global_step:
            argv += ["-global-step", str(int(global_step))]
        argv += list(extra_args)
        self.proc = subprocess.Popen(argv, cwd=os.path.dirname(self.executable),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     preexec_fn=_die_with_parent)
        try:
            self._wait_for_socket(start_timeout)
            super().__init__(unix=self.socket_path)
        except BaseException:
            self._stop()
            raise

    def _wait_for_socket(self, timeout):
        deadline = time.monotonic() + timeout
        while not os.path.exists(self.socket_path):
            if self.proc.poll() is not None:
                raise RuntimeError(f"duck server exited with code {self.proc.returncode} before listening:\n"
                                   + self.log_tail())
            if time.monotonic() > deadline:
                raise TimeoutError(f"duck server not listening after {timeout:.0f} s:\n" + self.log_tail())
            time.sleep(0.2)

    def log_tail(self, lines=30):
        try:
            with open(self.log_file, errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return "(no server log)"

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        return super().reset(self.seed)

    def _stop(self, grace=10.0):
        # The Unity player doesn't act on SIGTERM promptly, so after the close command (or
        # the disconnect, given -exit-on-disconnect) it gets a grace period, then SIGKILL.
        if self.proc.poll() is None:
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def close(self):
        if getattr(self, "sock", None) is not None:
            super().close()
            self.sock = None
        self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
