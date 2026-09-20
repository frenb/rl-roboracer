"""The neurons and edges the Unity overlay draws.

166,700 neurons and 25.6M edges cannot be drawn at frame rate and would be
unreadable anyway. This picks a legible few thousand: the feature detectors the
encoder drives, every descending neuron it reads, the named command neurons,
and the strongest edges among them.

Step 7 of docs/flybrain-driver-plan.md refines what is shown; the shape of the
output is fixed by the Geometry RPC and should not change.
"""
import numpy as np

# Driven by the encoder (step 4).
SENSORY_TYPES = ["LC4", "LPLC2", "LPLC1", "LC10a"]
# Called out individually in the overlay so they can be labelled.
COMMAND_TYPES = ["DNa02", "DNp01", "DNg100", "MDN", "DNp10", "DNg13", "pIP10"]
# Everything the readout sees.
READOUT_SUPERCLASS = "descending_neuron"

MAX_EDGES = 8000


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

    idx = np.unique(np.concatenate([sensory, command, readout]))

    role = np.full(len(idx), "interneuron", dtype=object)
    role[np.isin(idx, readout)] = "descending"
    role[np.isin(idx, sensory)] = "sensory"
    role[np.isin(idx, command)] = "command"

    labels = [
        {"type": str(ct[i]), "side": str(sd[i]), "role": str(r)}
        for i, r in zip(idx, role)
    ]

    positions = np.asarray(brain.positions)[idx].astype(np.float32)

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
