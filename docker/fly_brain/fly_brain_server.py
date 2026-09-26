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
# Accumulated trace that counts as full brightness, in spikes. A fresh spike
# contributes 1.0, so 1.8 means one spike reads a little over half-bright and
# two in quick succession saturate. See _snapshot_bytes for why this is not
# the 1/(1-decay) it looks like it should be.
SNAPSHOT_FULL = float(os.environ.get("FLY_SNAPSHOT_FULL", "1.8"))


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

        self._warmup()

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
        xp = self.brain.xp
        inject, dense = [], []
        for g in request.inject:
            idx = np.frombuffer(g.idx, np.int32)
            if len(idx) and (idx.min() < 0 or idx.max() >= self.brain.n):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              "inject index out of range [0, %d)" % self.brain.n)
            if g.amounts:
                amounts = np.frombuffer(g.amounts, np.float32)
                if len(amounts) != len(idx):
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                  "amounts (%d) and idx (%d) differ in length"
                                  % (len(amounts), len(idx)))
                # Converted once, outside the substep loop.
                dense.append((xp.asarray(idx),
                              xp.asarray(amounts).reshape(-1, 1)))
            else:
                inject.append((idx, float(g.amount)))
        substeps = request.substeps or 1

        eye = self._eye
        if request.eye_drive:
            eye = np.full(len(self.brain.visual), request.eye_drive, np.float32)

        total = 0
        with self.lock:
            for _ in range(substeps):
                # Per-neuron drive goes straight into the voltage array:
                # FlyBrain._amount reshapes any array to (1, batch) and reads
                # it as one value per fly, so step(inject=...) cannot do this.
                for gi, ga in dense:
                    self.brain.v[gi] += ga
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

    def Sectors(self, request, context):
        k = int(request.k)
        if k < 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "k must be >= 1")
        pos = np.asarray(self.brain.positions)
        per_sector = [[] for _ in range(k)]
        for t in request.types:
            idx = np.asarray(self.brain.cells([t], side=request.side or None))
            if not len(idx):
                continue
            # Sort each type along its own principal axis, then take matching
            # slices, so a sector holds comparable positions from every type
            # instead of one type filling it.
            c = pos[idx] - pos[idx].mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            order = np.argsort(c @ axis)
            for s, part in enumerate(np.array_split(idx[order], k)):
                per_sector[s].append(part)
        out = [np.concatenate(p).astype(np.int32) if p else np.zeros(0, np.int32)
               for p in per_sector]
        return pb.SectorsReply(idx=[a.tobytes() for a in out])

    # ----------------------------------------------------------------- utils
    def _warmup(self):
        # CuPy compiles each kernel on its first call. On a fresh container
        # that made the first client Step outlast the trainer's 10 s deadline,
        # so both inject paths are driven here before the port opens.
        t0 = time.time()
        idx = np.arange(64, dtype=np.int32).tobytes()
        request = pb.StepRequest(
            inject=[pb.Injection(idx=idx, amount=0.5),
                    pb.Injection(idx=idx,
                                 amounts=np.full(64, 0.5, np.float32).tobytes())],
            substeps=5, eye_drive=0.45, want_snapshot=True)
        for _ in range(20):
            self.Step(request, None)
        self.Reset(pb.ResetRequest(seed=0), None)
        print("warmup: 20 steps in %.1f s" % (time.time() - t0), flush=True)

    def _snapshot_bytes(self):
        # Fixed scale, not auto-ranged on the current maximum, so brightness
        # means the same thing from frame to frame.
        #
        # The scale used to be 1/(1-decay), i.e. full brightness required
        # firing on EVERY substep. Almost nothing does that except a handful of
        # tonically-driven cells, so they pinned at 255 while real bursts --
        # which accumulate one or two spikes' worth -- landed around a byte of
        # 40 and then all but vanished under the overlay's 1.6 gamma. The
        # measured result was a static bright core supplying 42% of the light
        # with the actual spiking invisible underneath it.
        #
        # Calibrating to a couple of spikes instead puts a single spike near
        # half brightness and anything brisker at full, which is what makes
        # bursts read as bursts. Tonic cells still saturate; they just no
        # longer own the whole dynamic range.
        v = np.clip(self._snap / SNAPSHOT_FULL, 0.0, 1.0) * 255.0
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
