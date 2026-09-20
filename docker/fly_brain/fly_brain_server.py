"""gRPC front end for the frozen fly.ai connectome.

Step 3 of docs/flybrain-driver-plan.md. Holds one FlyBrain, exposes Reset /
Step / Snapshot / Geometry / Info, and keeps a decaying descending-neuron trace
using flybrain's own `Trace` so the features here match what `Readout.fit` sees
offline in step 5.

Run:  docker compose exec -d fly-brain python /fly_brain/fly_brain_server.py
"""
import json
import os
import sys
import threading
import time
from concurrent import futures

import grpc
import numpy as np
from flybrain import FlyBrain, Trace

# Stubs live under gen/ so the generated "from fly_brain.proto import ..."
# resolves the same way here as it does on the trainer. Run gen_protos.sh if
# this import fails.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import display_subset
from fly_brain.proto import fly_brain_pb2 as pb
from fly_brain.proto import fly_brain_pb2_grpc as pb_grpc

PORT = int(os.environ.get("FLY_BRAIN_PORT", "50061"))
TRACE_TAU = float(os.environ.get("FLY_TRACE_TAU", "0.1"))
# Snapshot intensity is a decaying trace too, so the overlay shows a fading
# flash rather than a one-frame flicker that is invisible at 20 Hz.
SNAPSHOT_TAU = float(os.environ.get("FLY_SNAPSHOT_TAU", "0.15"))


class FlyBrainService(pb_grpc.FlyBrainServicer):
    def __init__(self):
        print("loading connectome ...", flush=True)
        t0 = time.time()
        self.brain = FlyBrain(device="auto")
        self.lock = threading.Lock()

        self.trace = Trace(self.brain, types=["descending_neuron"], tau=TRACE_TAU)
        self.trace_len = len(self.trace.idx)

        (self.display_idx, self.positions, self.edge_src, self.edge_dst,
         self.edge_weight, self.labels) = display_subset.build(self.brain)
        self.n_display = len(self.display_idx)
        # full-brain index -> display row, for scattering spikes each step
        self._display_slot = np.full(self.brain.n, -1, np.int64)
        self._display_slot[self.display_idx] = np.arange(self.n_display)
        self._snap = np.zeros(self.n_display, np.float32)
        self._snap_decay = float(np.exp(-self.brain.dt / SNAPSHOT_TAU))

        self._eye = np.zeros(len(self.brain.visual), np.float32)
        self._labels_json = json.dumps(self.labels)

        print("ready in %.1f s: n=%d device=%s trace_len=%d display=%d edges=%d"
              % (time.time() - t0, self.brain.n, self.brain.device,
                 self.trace_len, self.n_display, len(self.edge_src)),
              flush=True)

    # ------------------------------------------------------------------ rpcs
    def Info(self, request, context):
        return pb.InfoReply(
            n_neurons=int(self.brain.n),
            n_connections=int(len(self.brain.indices)),
            device=str(self.brain.device),
            dt=float(self.brain.dt),
            tonic=float(self.brain.tonic),
            gain=float(self.brain.gain),
            trace_len=int(self.trace_len),
            trace_tau=float(TRACE_TAU),
            batch=int(self.brain.batch),
            n_display=int(self.n_display),
        )

    def Reset(self, request, context):
        with self.lock:
            self.brain.reset(request.seed)
            self.trace.reset()
            self._snap[:] = 0.0
        return pb.ResetReply()

    def Step(self, request, context):
        inject = [(np.frombuffer(g.idx, np.int32), float(g.amount))
                  for g in request.inject]
        for idx, _ in inject:
            if len(idx) and (idx.min() < 0 or idx.max() >= self.brain.n):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              "inject index out of range [0, %d)" % self.brain.n)
        substeps = request.substeps or 1

        eye = self._eye
        if request.eye_drive:
            eye = np.full(len(self.brain.visual), request.eye_drive, np.float32)

        total = 0
        with self.lock:
            for _ in range(substeps):
                fired = self.brain.step(eye, inject=inject)
                self.trace.observe(fired)
                total += len(fired)

                self._snap *= self._snap_decay
                rows = self._display_slot[fired]
                rows = rows[rows >= 0]
                if len(rows):
                    self._snap[rows] += 1.0

            trace = np.asarray(self.trace.features(), np.float32).ravel()
            snap = self._snapshot_bytes() if request.want_snapshot else b""

        return pb.StepReply(trace=trace.tobytes(), snapshot=snap, spikes=total)

    def Snapshot(self, request, context):
        with self.lock:
            return pb.SnapshotReply(activity=self._snapshot_bytes())

    def Geometry(self, request, context):
        return pb.GeometryReply(
            positions=self.positions.tobytes(),
            edge_src=self.edge_src.tobytes(),
            edge_dst=self.edge_dst.tobytes(),
            edge_weight=self.edge_weight.tobytes(),
            labels_json=self._labels_json,
            n_display=int(self.n_display),
            n_edges=int(len(self.edge_src)),
        )

    def Cells(self, request, context):
        if request.group:
            if request.types:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              "pass types or group, not both")
            try:
                idx = self.brain.groups[request.group]
            except KeyError:
                context.abort(grpc.StatusCode.NOT_FOUND,
                              "no group %r; have %s"
                              % (request.group, sorted(self.brain.groups)))
        else:
            if not request.types:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              "pass types or group")
            idx = self.brain.cells(list(request.types),
                                   side=request.side or None)
        return pb.CellsReply(idx=np.asarray(idx, np.int32).tobytes())

    # ----------------------------------------------------------------- utils
    def _snapshot_bytes(self):
        # Normalize against a full trace (1/(1-decay)) so the scale is stable
        # across frames rather than auto-ranging on the current maximum.
        ceiling = 1.0 / (1.0 - self._snap_decay)
        v = np.clip(self._snap / ceiling, 0.0, 1.0) * 255.0
        return v.astype(np.uint8).tobytes()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_FlyBrainServicer_to_server(FlyBrainService(), server)
    server.add_insecure_port("[::]:%d" % PORT)
    server.start()
    print("fly-brain gRPC listening on :%d" % PORT, flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
