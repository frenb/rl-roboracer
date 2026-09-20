"""Publish the fly brain's overlay to Unity: geometry once, activity at 20 Hz.

Step 8 of docs/flybrain-driver-plan.md. Two topics, deliberately split:

  fly_brain_geometry  static soma positions + edge pairs, ~120 KB
  fly_brain_activity  per-neuron intensity as uint8, ~2 KB a frame

Both must also appear as ``RosSubscriber`` entries in
``docker/ros_server/ROS/src/niryo_moveit/scripts/unity_node.py``. That routing
table is static -- the embedded ROS-TCP connector never registers subscribers
dynamically -- so an unlisted topic publishes fine here and silently never
arrives in Unity.

Every numeric field is base64 of a little-endian buffer rather than a JSON
number array. 2,034 neurons and 8,000 edges as JSON text is about 207 KB
against 120 KB packed, and Unity's JsonUtility would have to allocate and parse
24,000 floats every time geometry is resent; Convert.FromBase64String plus a
Buffer.BlockCopy is one allocation.

WHY A BACKGROUND THREAD: the publish is a blocking gRPC round trip, and the
policy calls into this from inside its action loop, which is on the sim's
critical path. The thread holds only the newest activity frame, so a slow or
dead ros-server costs the driving loop nothing and stale frames are dropped
rather than queued (same rationale as rollout_viz).
"""
import base64
import json
import os
import threading
import time

import numpy as np

GEOMETRY_TOPIC = "fly_brain_geometry"
ACTIVITY_TOPIC = "fly_brain_activity"

# Roles and sides travel as small ints, not strings: one byte per neuron
# instead of a repeated type name, and Unity switches on them directly.
ROLE_CODES = {"interneuron": 0, "sensory": 1, "command": 2, "descending": 3}
SIDE_CODES = {"L": 1, "R": 2}


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def get_config():
    def _float(key, default):
        try:
            return float(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": os.environ.get("FLY_VIZ_ENABLED", "1").lower()
        in ("1", "true", "yes", "on"),
        # EVAL and the fly courses are single-env on actor 0, so unlike
        # rollout_viz there is no per-actor fan-out to do here.
        "addr": os.environ.get("FLY_VIZ_ADDR", "ros-server-0:50051"),
        "hz": max(1.0, _float("FLY_VIZ_HZ", 20.0)),
        # Geometry is static, but Unity may connect (or reconnect after a
        # scene reload) long after the first send, and the static routing
        # table gives it no way to ask for a resend. A slow heartbeat is the
        # cheap fix: ~12 KB/s averaged, against 54 KB/s for the activity.
        "geometry_period": _float("FLY_VIZ_GEOMETRY_S", 10.0),
        # ---- Placement, published in the geometry payload so the overlay can
        # be moved and resized WITHOUT a Unity rebuild (set the env var and
        # restart the trainer). A build has no inspector, so without this every
        # "it's off screen" costs an Editor round trip. Unity falls back to its
        # own defaults when displaySize is absent/0.
        "display_size": _float("FLY_VIZ_DISPLAY_SIZE", 12.0),
        "point_size": _float("FLY_VIZ_POINT_SIZE", 0.10),
        "offset": os.environ.get("FLY_VIZ_OFFSET", "0,40,0"),
        "spin": _float("FLY_VIZ_SPIN", 8.0),
        "edge_alpha": _float("FLY_VIZ_EDGE_ALPHA", 0.22),
    }


def _offset(text):
    try:
        x, y, z = (float(v) for v in str(text).split(","))
        return x, y, z
    except (TypeError, ValueError):
        return 0.0, 40.0, 0.0


class FlyBrainViz(object):
    """Owns the publish thread. Construct once per policy; call submit() per step."""

    def __init__(self, client, config=None):
        self.cfg = config or get_config()
        self._client = client
        self._pub = None
        self._geometry_json = None
        self._last_geometry = 0.0
        self._slot = None           # newest (activity, step, spikes), or None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._err_count = 0
        self._sent = 0
        if self.cfg["enabled"]:
            self._thread = threading.Thread(
                target=self._run, name="fly-brain-viz", daemon=True)
            self._thread.start()
            print("[fly_viz] publishing %s + %s -> %s at %.0f Hz"
                  % (GEOMETRY_TOPIC, ACTIVITY_TOPIC, self.cfg["addr"],
                     self.cfg["hz"]), flush=True)

    @property
    def enabled(self):
        return self._thread is not None

    def submit(self, activity, step=0, spikes=0):
        """Hand the thread the newest activity frame. Never raises, never blocks."""
        if self._thread is None or activity is None:
            return
        with self._lock:
            self._slot = (np.asarray(activity, np.uint8), int(step), int(spikes))

    def stop(self):
        self._stop.set()

    # ---------------------------------------------------------------- internals
    def _build_geometry(self):
        """Fetch the display subset once and pack it for Unity."""
        g = self._client.geometry()
        pos = np.asarray(g["positions"], np.float32)

        # Normalize to a unit box centred on the origin so the Unity side is a
        # single scale factor rather than hard-coded connectome coordinates
        # (which are raw MaleCNS nanometre-ish soma positions).
        centre = pos.mean(axis=0)
        pos = pos - centre
        extent = float(np.abs(pos).max())
        if extent > 0:
            pos = pos / extent

        labels = json.loads(g["labels_json"])
        role = np.array([ROLE_CODES.get(l.get("role"), 0) for l in labels], np.uint8)
        side = np.array([SIDE_CODES.get(l.get("side"), 0) for l in labels], np.uint8)

        ox, oy, oz = _offset(self.cfg["offset"])
        return json.dumps({
            "kind": "geometry",
            "stamp": time.time(),
            "n": int(g["n_display"]),
            "nEdges": int(g["n_edges"]),
            "displaySize": self.cfg["display_size"],
            "pointSize": self.cfg["point_size"],
            "offsetX": ox, "offsetY": oy, "offsetZ": oz,
            "spin": self.cfg["spin"],
            "edgeAlpha": self.cfg["edge_alpha"],
            "pos": _b64(pos.reshape(-1)),            # float32[n*3], xyz interleaved
            "edgeSrc": _b64(g["edge_src"]),          # int32[nEdges], index into pos
            "edgeDst": _b64(g["edge_dst"]),          # int32[nEdges]
            "edgeWeight": _b64(g["edge_weight"]),    # float32[nEdges], signed
            "role": _b64(role),                      # uint8[n], see ROLE_CODES
            "side": _b64(side),                      # uint8[n], see SIDE_CODES
        })

    def _publisher(self):
        if self._pub is not None:
            if not self._pub.is_healthy():
                self._pub.reconnect()
            return self._pub
        from rollout_viz import _DirectPublisher
        self._pub = _DirectPublisher(self.cfg["addr"])
        return self._pub

    def _run(self):
        period = 1.0 / self.cfg["hz"]
        while not self._stop.is_set():
            t0 = time.time()
            try:
                pub = self._publisher()
                if self._geometry_json is None:
                    self._geometry_json = self._build_geometry()
                if (t0 - self._last_geometry) >= self.cfg["geometry_period"]:
                    pub.publish(GEOMETRY_TOPIC, self._geometry_json)
                    self._last_geometry = t0

                with self._lock:
                    slot, self._slot = self._slot, None
                if slot is not None:
                    activity, step, spikes = slot
                    pub.publish(ACTIVITY_TOPIC, json.dumps({
                        "kind": "activity",
                        "stamp": t0,
                        "step": step,
                        "n": int(activity.size),
                        "spikes": spikes,
                        "act": _b64(activity),       # uint8[n], 0..255 intensity
                    }))
                    self._sent += 1
                if self._err_count:
                    print("[fly_viz] recovered after %d error(s)"
                          % self._err_count, flush=True)
                    self._err_count = 0
            except Exception as e:  # noqa: BLE001 - the overlay must never stop a run
                self._err_count += 1
                if self._err_count <= 3 or self._err_count % 200 == 0:
                    print("[fly_viz] publish error #%d: %s"
                          % (self._err_count, e), flush=True)
            self._stop.wait(max(0.0, period - (time.time() - t0)))
