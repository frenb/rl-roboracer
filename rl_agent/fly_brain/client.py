"""Trainer-side client for the fly-brain gRPC service.

Step 3 of docs/flybrain-driver-plan.md. The connectome runs in its own
container because it needs Python 3.10 and this process is Python 3.8; this
class is the only thing that should know that.

Smoke test (proves the cross-container, cross-interpreter path):

    docker compose exec -w /python_ws/src sim-controller python -m fly_brain.client
"""
import os
import time

import grpc
import numpy as np

from fly_brain.proto import fly_brain_pb2 as pb
from fly_brain.proto import fly_brain_pb2_grpc as pb_grpc

DEFAULT_TARGET = os.environ.get("FLY_BRAIN_TARGET", "fly-brain:50061")

# The gym publishes scene data every 0.1 s and the brain's dt is 0.020 s, so
# one control step is five brain steps. See the plan's verified interface facts.
SUBSTEPS = 5
# inject.py holds the photoreceptors here; without it the brain is too quiet.
EYE_DRIVE = 0.45


class FlyBrainClient(object):
    def __init__(self, target=DEFAULT_TARGET, timeout=10.0):
        self.target = target
        self.timeout = timeout
        self._channel = grpc.insecure_channel(target)
        self._stub = pb_grpc.FlyBrainStub(self._channel)
        self.info = self._stub.Info(pb.InfoRequest(), timeout=timeout)

    def close(self):
        self._channel.close()

    def reset(self, seed=0):
        self._stub.Reset(pb.ResetRequest(seed=int(seed)), timeout=self.timeout)

    def step(self, inject=(), substeps=SUBSTEPS, eye_drive=EYE_DRIVE,
             want_snapshot=False):
        """Advance the brain one control step. Returns (trace, snapshot, spikes).

        `inject` is a sequence of (indices, amount) pairs, one per driven
        population, mirroring FlyBrain.step. The amount is a scalar; see the
        Injection message for why it cannot be per-neuron.

        `trace` is float32[info.trace_len]; `snapshot` is uint8[info.n_display]
        or None.
        """
        groups = [
            pb.Injection(idx=np.ascontiguousarray(idx, np.int32).tobytes(),
                         amount=float(amount))
            for idx, amount in inject
        ]
        reply = self._stub.Step(
            pb.StepRequest(inject=groups, substeps=int(substeps),
                           eye_drive=float(eye_drive),
                           want_snapshot=bool(want_snapshot)),
            timeout=self.timeout,
        )
        trace = np.frombuffer(reply.trace, np.float32)
        snap = np.frombuffer(reply.snapshot, np.uint8) if reply.snapshot else None
        return trace, snap, reply.spikes

    def snapshot(self):
        reply = self._stub.Snapshot(pb.SnapshotRequest(), timeout=self.timeout)
        return np.frombuffer(reply.activity, np.uint8)

    def cells(self, types=None, side=None, group=None):
        reply = self._stub.Cells(
            pb.CellsRequest(types=types or [], side=side or "", group=group or ""),
            timeout=self.timeout,
        )
        return np.frombuffer(reply.idx, np.int32)

    def geometry(self):
        """Static overlay data. Fetch once; it never changes."""
        r = self._stub.Geometry(pb.GeometryRequest(), timeout=60.0)
        return {
            "positions": np.frombuffer(r.positions, np.float32).reshape(-1, 3),
            "edge_src": np.frombuffer(r.edge_src, np.int32),
            "edge_dst": np.frombuffer(r.edge_dst, np.int32),
            "edge_weight": np.frombuffer(r.edge_weight, np.float32),
            "labels_json": r.labels_json,
            "n_display": r.n_display,
            "n_edges": r.n_edges,
        }


def _smoke_test():
    c = FlyBrainClient()
    i = c.info
    print("connected to %s" % c.target)
    print("  neurons=%d connections=%d device=%s" % (i.n_neurons, i.n_connections, i.device))
    print("  dt=%.3f tonic=%.2f gain=%.1f trace_len=%d tau=%.2f display=%d"
          % (i.dt, i.tonic, i.gain, i.trace_len, i.trace_tau, i.n_display))

    lc4 = c.cells(types=["LC4", "LPLC2"], side="L")
    print("  LC4+LPLC2 left: %d cells" % len(lc4))
    inject = [(lc4, 0.8)]

    # CuPy compiles its kernels on first use, which costs about 30 ms/step for
    # the first hundred steps after the container starts. Burn that off before
    # timing, and note that the first episode of a fresh container pays it.
    c.reset(seed=0)
    t0 = time.time()
    for _ in range(100):
        c.step(inject)
    warmup = (time.time() - t0) / 100

    c.reset(seed=0)
    t0 = time.time()
    n = 100
    for k in range(n):
        trace, snap, spikes = c.step(inject, want_snapshot=(k == n - 1))
    dt = (time.time() - t0) / n
    print("  warmup pass: %.1f ms/step" % (warmup * 1000.0))

    assert trace.shape == (i.trace_len,), trace.shape
    assert snap is not None and snap.shape == (i.n_display,), snap.shape
    print("  %d steps of %d substeps: %.1f ms/step (%.1f Hz), last spikes=%d"
          % (n, SUBSTEPS, dt * 1000.0, 1.0 / dt, spikes))
    print("  trace: shape=%s nonzero=%d max=%.3f" % (trace.shape, int((trace > 0).sum()), trace.max()))
    print("  snapshot: nonzero=%d max=%d" % (int((snap > 0).sum()), snap.max()))

    # Same seed, same injection must give the same trace, or nothing downstream
    # is reproducible.
    c.reset(seed=0)
    for _ in range(n):
        trace2, _, _ = c.step(inject)
    print("  determinism: max|trace - trace2| = %.3e" % np.abs(trace - trace2).max())

    g = c.geometry()
    print("  geometry: %d neurons, %d edges, positions %s"
          % (g["n_display"], g["n_edges"], g["positions"].shape))
    c.close()
    print("OK")


if __name__ == "__main__":
    _smoke_test()
