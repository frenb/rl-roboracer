"""The neurons and edges the Unity overlay draws.

166,700 neurons and 25.6M edges cannot be drawn at frame rate and would be
unreadable anyway. This picks a legible few thousand: the feature detectors the
encoder drives, every descending neuron it reads, the named command neurons,
and the strongest edges among them.

Step 7 of docs/flybrain-driver-plan.md refines what is shown; the shape of the
output is fixed by the Geometry RPC and should not change.
"""
import os

import numpy as np

# How many neurons to draw purely for the silhouette. See _context_sample.
CONTEXT_N = int(os.environ.get("FLY_CONTEXT_N", "16000"))

# Driven by the encoder (step 4).
SENSORY_TYPES = ["LC4", "LPLC2", "LPLC1", "LC10a"]
# Called out individually in the overlay so they can be labelled.
COMMAND_TYPES = ["DNa02", "DNp01", "DNg100", "MDN", "DNp10", "DNg13", "pIP10"]
# Everything the readout sees.
READOUT_SUPERCLASS = "descending_neuron"

MAX_EDGES = 8000
# How many relay interneurons to show between the sensory and descending
# layers. Without these the overlay draws inputs and outputs with nothing in
# between, and step 9's "the looming cluster and its path to DNp01 flare" has
# no path to light up.
MAX_INTERNEURONS = 1200


def _relay_interneurons(brain, sensory, readout, exclude, k=MAX_INTERNEURONS):
    """The k neurons carrying the most sensory -> descending two-hop weight.

    Scored as (total |weight| received from `sensory`) x (total |weight| sent to
    `readout`), so a neuron must both listen to the feature detectors and talk
    to the motor output to appear. Candidates are restricted to actual targets
    of `sensory`, which keeps this to a few thousand row scans instead of a
    pass over all 25.6M connections.
    """
    indptr = np.asarray(brain.indptr)
    indices = np.asarray(brain.indices)
    weights = np.asarray(brain.weights)

    # Hop 1: everything the sensory populations project onto.
    cols, ws = [], []
    for s in sensory:
        lo, hi = indptr[s], indptr[s + 1]
        if hi > lo:
            cols.append(indices[lo:hi])
            ws.append(np.abs(weights[lo:hi]))
    if not cols:
        return np.zeros(0, np.int64)
    from_sensory = np.bincount(np.concatenate(cols),
                               weights=np.concatenate(ws).astype(np.float64),
                               minlength=brain.n)

    candidates = np.flatnonzero(from_sensory > 0)
    candidates = candidates[~np.isin(candidates, exclude)]
    if not len(candidates):
        return np.zeros(0, np.int64)

    # Hop 2: of those, how strongly each drives the descending population.
    is_readout = np.zeros(brain.n, bool)
    is_readout[readout] = True
    to_readout = np.zeros(len(candidates), np.float64)
    for j, c in enumerate(candidates):
        lo, hi = indptr[c], indptr[c + 1]
        if hi > lo:
            sel = is_readout[indices[lo:hi]]
            if sel.any():
                to_readout[j] = np.abs(weights[lo:hi][sel]).sum()

    score = from_sensory[candidates] * to_readout
    keep = np.flatnonzero(score > 0)
    if len(keep) > k:
        keep = keep[np.argsort(-score[keep])[:k]]
    return candidates[keep]


def _context_sample(brain, exclude, k=CONTEXT_N):
    """Neurons drawn only so the nervous system has a recognizable shape.

    Every neuron in the driving circuit is a brain neuron: the feature
    detectors, the relays and the descending somas all sit in the head. Drawn
    alone they are a blob with no body, which is why the overlay read as an
    amorphous cloud. fly.ai's own render shows the whole central nervous
    system, brain and ventral nerve cord, with the active cells picked out
    against it, and the nerve cord is most of that picture.

    A uniform draw, deliberately: neuron density varies enormously between
    neuropils, and preserving that is what makes the optic lobes read as dense
    and the cord as sparse, the way they do in the reference. Seeded, so the
    display subset is identical across server restarts and any recorded run
    stays comparable.

    These carry real activity like every other drawn neuron; they are just not
    part of the readout path.
    """
    pos = np.asarray(brain.positions)
    ok = np.flatnonzero(np.isfinite(pos).all(axis=1))
    ok = ok[~np.isin(ok, exclude)]
    if len(ok) <= k:
        return ok
    return np.sort(np.random.default_rng(0).choice(ok, size=k, replace=False))


def build(brain, max_edges=MAX_EDGES):
    """Return (idx, positions, edge_src, edge_dst, edge_weight, labels).

    `idx` indexes into the full brain. `edge_src`/`edge_dst` index into `idx`,
    so Unity can treat the subset as a standalone graph.
    """
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)

    sensory = np.flatnonzero(np.isin(ct, SENSORY_TYPES))
    command = np.flatnonzero(np.isin(ct, COMMAND_TYPES))
    readout = np.asarray(brain.cells([READOUT_SUPERCLASS]))

    core = np.unique(np.concatenate([sensory, command, readout]))
    relay = _relay_interneurons(brain, sensory, readout, exclude=core)
    circuit = np.unique(np.concatenate([core, relay]))
    context = _context_sample(brain, exclude=circuit)
    idx = np.unique(np.concatenate([circuit, context]))

    # A few neurons carry no soma position in brain.npz (6 descending ones in
    # the current build). They cannot be drawn, and leaving them in makes every
    # downstream centre/extent NaN, so drop them here - before the slot map and
    # the edge list are built - instead of asking each consumer to defend
    # itself. n_display then matches what the overlay can actually show.
    all_positions = np.asarray(brain.positions)
    idx = idx[np.isfinite(all_positions[idx]).all(axis=1)]

    # Context first, so anything that is also part of the circuit is relabelled
    # by the lines below and keeps its brighter role.
    role = np.full(len(idx), "context", dtype=object)
    role[np.isin(idx, relay)] = "interneuron"
    role[np.isin(idx, readout)] = "descending"
    role[np.isin(idx, sensory)] = "sensory"
    role[np.isin(idx, command)] = "command"

    labels = [
        {"type": str(ct[i]), "side": str(sd[i]), "role": str(r)}
        for i, r in zip(idx, role)
    ]

    positions = all_positions[idx].astype(np.float32)

    # CSR slice restricted to the subset, then the strongest edges by |weight|.
    indptr = np.asarray(brain.indptr)
    indices = np.asarray(brain.indices)
    weights = np.asarray(brain.weights)

    # Map full-brain index -> subset position, -1 for everything excluded.
    slot = np.full(brain.n, -1, np.int64)
    slot[idx] = np.arange(len(idx))

    src_list, dst_list, w_list = [], [], []
    for row, neuron in enumerate(idx):
        lo, hi = indptr[neuron], indptr[neuron + 1]
        if hi <= lo:
            continue
        cols = indices[lo:hi]
        keep = slot[cols] >= 0
        if not keep.any():
            continue
        src_list.append(np.full(int(keep.sum()), row, np.int32))
        dst_list.append(slot[cols[keep]].astype(np.int32))
        w_list.append(weights[lo:hi][keep].astype(np.float32))

    if src_list:
        edge_src = np.concatenate(src_list)
        edge_dst = np.concatenate(dst_list)
        edge_weight = np.concatenate(w_list)
        if len(edge_weight) > max_edges:
            keep = np.argsort(-np.abs(edge_weight))[:max_edges]
            keep.sort()
            edge_src, edge_dst = edge_src[keep], edge_dst[keep]
            edge_weight = edge_weight[keep]
    else:
        edge_src = np.zeros(0, np.int32)
        edge_dst = np.zeros(0, np.int32)
        edge_weight = np.zeros(0, np.float32)

    return idx, positions, edge_src, edge_dst, edge_weight, labels
