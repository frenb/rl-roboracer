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
        # Defaults tuned against the sim's top-down camera.
        "display_size": _float("FLY_VIZ_DISPLAY_SIZE", 50.0),
        "point_size": _float("FLY_VIZ_POINT_SIZE", 0.16),
        # Parks the overlay in the empty area left of the track, under the ROS
        # HUD, which also puts it over the camera's flat background instead of
        # grass. Note the axes are the overlay parent's LOCAL ones and that
        # parent is rotated: measured against this camera, +z moves the brain
        # left at 3.1 px/m and +x moves it up at 2.5 px/m (at 1250 px wide).
        "offset": os.environ.get("FLY_VIZ_OFFSET", "-8,40,130"),
        # 0, not a slow turntable: the sim camera looks straight down, so a
        # spin about Unity's y would swing the brain in the screen plane and
        # left/right would stop meaning left/right.
        "spin": _float("FLY_VIZ_SPIN", 0.0),
        "edge_alpha": _float("FLY_VIZ_EDGE_ALPHA", 0.40),
        # Draw the connections at all. See the note in _build_geometry.
        "edges": os.environ.get("FLY_VIZ_EDGES", "0").lower()
        in ("1", "true", "yes", "on"),
        # Which connectome axis goes on which Unity axis, as signed names for
        # Unity x,y,z. The camera looks down Unity y, so whatever lands there
        # is the axis we lose. Measured on the display subset (n=3225):
        #   connectome x  left-right   (L +14116 vs R -14182, 1.77 sd apart)
        #   connectome y  sensory -5618 -> descending +7683, the flow axis
        #   connectome z  thinnest by sd, and the long descending projections
        # So x stays horizontal, the flow axis becomes screen-vertical, and z
        # is spent on depth. The sign on y is +, not -, because the sim
        # camera's up maps to -Z: unnegated is what puts sensory at the top of
        # the screen with the flow running down to the descending neurons.
        "axes": os.environ.get("FLY_VIZ_AXES", "x,z,y"),
        # How much of the camera-facing axis to keep. 1.0 is anatomically
        # honest but perspective-smears the overlay; see _build_geometry.
        "depth_scale": _float("FLY_VIZ_DEPTH_SCALE", 0.12),
        # Floor under the published activity byte, 0-255. Unity maps that byte
        # onto a rest->full ramp whose resting end is a hardcoded 0.18 of the
        # role colour at 0.20 alpha, which is near-invisible against the sim's
        # background; the connectome is quiet most of the time, so most of the
        # cloud sits at that floor. Lifting it here keeps the knob on the side
        # that does not need an Editor rebuild. Activity still varies, it just
        # starts somewhere you can see: 140 lands a resting neuron at ~40%
        # brightness after Unity's 1.6 gamma.
        "act_floor": int(_float("FLY_VIZ_ACT_FLOOR", 140.0)),
    }


def _axis_map(spec):
    """Parse "x,z,-y" into (source index per Unity axis, sign per Unity axis)."""
    names = {"x": 0, "y": 1, "z": 2}
    try:
        parts = [p.strip().lower() for p in str(spec).split(",")]
        if len(parts) != 3:
            raise ValueError(spec)
        idx, sign = [], []
        for p in parts:
            s = -1.0 if p.startswith("-") else 1.0
            idx.append(names[p.lstrip("+-")])
            sign.append(s)
        return idx, np.array(sign, np.float32)
    except (KeyError, ValueError):
        return [0, 2, 1], np.array([1.0, 1.0, -1.0], np.float32)


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

        # Reorient before normalizing so the sim's top-down camera sees the
        # informative plane rather than looking down the sensory->descending
        # axis. See the "axes" note in get_config.
        idx, sign = _axis_map(self.cfg["axes"])
        pos = pos[:, idx] * sign

        # Normalize to a unit box centred on the origin so the Unity side is a
        # single scale factor rather than hard-coded connectome coordinates
        # (which are raw MaleCNS nanometre-ish soma positions). The median, not
        # the mean, so the long descending projections down the nerve cord do
        # not drag the centre off the brain.
        pos = pos - np.median(pos, axis=0)

        # Scale the two on-screen axes together, by their shared max, so the
        # anatomy keeps its true aspect and nothing needs clipping. Taking the
        # max over all three axes is what used to shrink the brain to a dot --
        # the culprit was depth (z spans 106k against a 7.6k sd), and depth is
        # handled separately below, so within the screen plane the max is only
        # about 1.4x the 99th percentile and costs nothing.
        plane = float(np.abs(pos[:, [0, 2]]).max())
        if plane > 0:
            pos = pos / plane
        # Depth is the one axis a top-down camera throws away, and it is where
        # the extreme outliers live. Clip it, then flatten it: left free, a
        # neuron at 3.5 units sits ~88 m up at displaySize 25, close enough to
        # a perspective camera that it projects far off to the side and smears
        # the whole structure into a radial fan.
        pos[:, 1] = np.clip(pos[:, 1], -1.0, 1.0) * self.cfg["depth_scale"]

        labels = json.loads(g["labels_json"])
        role = np.array([ROLE_CODES.get(l.get("role"), 0) for l in labels], np.uint8)
        side = np.array([SIDE_CODES.get(l.get("side"), 0) for l in labels], np.uint8)

        # Edges are off by default. 8000 lines over a bright sim background
        # read as a dark scribble that buries the neurons; the reference
        # renders this connectome as a bare point cloud for the same reason.
        # Sending none (rather than alpha 0) also drops ~94 KB off the payload.
        edges_on = self.cfg["edges"]
        n_edges = int(g["n_edges"]) if edges_on else 0
        empty32 = np.zeros(0, np.int32)

        ox, oy, oz = _offset(self.cfg["offset"])
        return json.dumps({
            "kind": "geometry",
            "stamp": time.time(),
            "n": int(g["n_display"]),
            "nEdges": n_edges,
            "displaySize": self.cfg["display_size"],
            "pointSize": self.cfg["point_size"],
            "offsetX": ox, "offsetY": oy, "offsetZ": oz,
            "spin": self.cfg["spin"],
            "edgeAlpha": self.cfg["edge_alpha"],
            "pos": _b64(pos.reshape(-1)),            # float32[n*3], xyz interleaved
            # Always a string, never absent: Unity base64-decodes these
            # unconditionally and null would throw in BuildMeshes.
            "edgeSrc": _b64(g["edge_src"] if edges_on else empty32),
            "edgeDst": _b64(g["edge_dst"] if edges_on else empty32),
            "edgeWeight": _b64(g["edge_weight"] if edges_on else empty32),
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
                    floor = self.cfg["act_floor"]
                    if floor > 0:
                        # Rescale into [floor, 255] rather than clamping, so a
                        # firing neuron still separates from a resting one.
                        activity = (
                            floor + activity.astype(np.uint16)
                            * (255 - floor) // 255).astype(np.uint8)
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
