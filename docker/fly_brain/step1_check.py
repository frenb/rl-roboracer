"""Step 1 of docs/flybrain-driver-plan.md: is the brain up, on the GPU, and fast?

Checks three things and prints the API surface so later steps do not have to
guess at it:

  1. the build has exactly 166,700 neurons and 25,582,938 connections
  2. FlyBrain is actually on CUDA (1-2 ms/step) and not silently on CPU (9-15 ms)
  3. what attributes FlyBrain exposes, since the README only documents a few

Run:  docker compose exec fly-brain python /fly_brain/step1_check.py
"""
import os
import time

import numpy as np

EXPECT_NEURONS = 166_700
EXPECT_CONNECTIONS = 25_582_938
N_STEPS = 100

print("=" * 70)
print("FLY_DATA =", os.environ.get("FLY_DATA"))
print("FLY_DEVICE =", os.environ.get("FLY_DEVICE"))

import flybrain  # noqa: E402
from flybrain import FlyBrain  # noqa: E402

print("flybrain", getattr(flybrain, "__version__", "(no __version__)"))

# ---------------------------------------------------------------- data files
# The .npz files are the authoritative counts: they are what `flybrain build`
# wrote, independent of whatever the object chooses to expose.
fly_data = os.environ.get("FLY_DATA", os.path.expanduser("~/fly-data"))
print()
print("=== files in FLY_DATA ===")
for root, _dirs, files in os.walk(fly_data):
    for f in sorted(files):
        p = os.path.join(root, f)
        print("  %-52s %8.1f MB" % (p, os.path.getsize(p) / 1e6))

# ---------------------------------------------------------------- build
print()
print("=== constructing FlyBrain(device='auto') ===")
t0 = time.time()
brain = FlyBrain(device="auto")
print("constructed in %.1f s" % (time.time() - t0))

# Probe for the device / size attributes rather than assuming names.
def probe(obj, names):
    for n in names:
        if hasattr(obj, n):
            try:
                return n, getattr(obj, n)
            except Exception:
                pass
    return None, None

dev_name, dev = probe(brain, ["device", "_device", "backend"])
print("device attr: %s = %r" % (dev_name, dev))

n_name, n_neurons = probe(brain, ["n", "n_neurons", "num_neurons", "size"])
print("neuron-count attr: %s = %r" % (n_name, n_neurons))

# ---------------------------------------------------------------- counts
print()
print("=== counts ===")
neurons = conns = None
try:
    w = np.load(os.path.join(fly_data, "weights.npz"))
    print("weights.npz keys:", list(w.keys()))
    # CSC/CSR: indices length is the connection count; indptr length is n+1.
    if "indices" in w:
        conns = int(w["indices"].shape[0])
    if "indptr" in w:
        neurons = int(w["indptr"].shape[0]) - 1
    if "shape" in w:
        print("weights shape:", w["shape"])
except Exception as exc:
    print("could not read weights.npz:", exc)

if neurons is None:
    neurons = n_neurons

print("neurons:     %s  (expected %s)" % (neurons, EXPECT_NEURONS))
print("connections: %s  (expected %s)" % (conns, EXPECT_CONNECTIONS))
ok_counts = (neurons == EXPECT_NEURONS) and (conns == EXPECT_CONNECTIONS)
print("counts match:", ok_counts)
if not ok_counts:
    print("  -> if these differ, the MaleCNS data changed; do not proceed.")

# ---------------------------------------------------------------- timing
print()
print("=== timing %d steps ===" % N_STEPS)
brain.step()  # warm up any JIT / kernel compilation
t0 = time.time()
for _ in range(N_STEPS):
    brain.step()
dt = (time.time() - t0) / N_STEPS
print("%.2f ms/step  (%.0f steps/s)" % (dt * 1000.0, 1.0 / dt))
if dt < 0.004:
    print("  -> GPU-class timing. fly.ai measures 1.4 ms on an RTX 4060 laptop.")
elif dt < 0.020:
    print("  -> WARNING: slower than expected for a 4090. Check FLY_DEVICE=cuda.")
else:
    print("  -> LOOKS LIKE CPU (fly.ai measures 9-15 ms on 24 threads).")

print()
print("At K=5 substeps per 10 Hz control step, one env step costs %.1f ms."
      % (dt * 5 * 1000.0))

# ---------------------------------------------------------------- api surface
print()
print("=== public FlyBrain attributes (for steps 2-4) ===")
print("  " + ", ".join(sorted(a for a in dir(brain) if not a.startswith("_"))))

print()
print("=== flybrain module exports ===")
print("  " + ", ".join(sorted(a for a in dir(flybrain) if not a.startswith("_"))))

# The encoder in step 4 needs these to resolve; fail early if they don't.
print()
print("=== can we resolve the neuron types the plan needs? ===")
for t in ["LC4", "LPLC2", "LPLC1", "LC10a", "DNa02", "DNp01", "DNg100", "MDN",
          "descending_neuron"]:
    try:
        left = brain.cells([t], side="L")
        right = brain.cells([t], side="R")
        print("  %-18s L=%-6d R=%-6d" % (t, len(left), len(right)))
    except Exception as exc:
        print("  %-18s FAILED: %s" % (t, exc))
