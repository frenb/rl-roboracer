"""Where the 1,314 descending neurons sit in the wiring, and how many the cues move.

Backs the "Descending neurons" section of docs/fly-brain-driver-how-it-works.md.
Two parts:

  1. Wiring. Share of the descending neurons' input and output synaptic weight
     by partner superclass, and how much of the motor neurons' input comes
     straight from them. flybrain/build.py stores the matrix as
     csr_matrix((w, (post, pre))), so row i lists the inputs *to* neuron i.
  2. Activity. Firing at rest, then how many descending and motor neurons change
     rate when each of the encoder's four cues is injected at 0.8, the encoder's
     MAX_AMOUNT. A change counts if it is at least 1 spike/s and more than
     three standard errors across seeds.

Builds its own FlyBrain, so it never touches the served brain a training job is
stepping.

Run:  docker compose exec fly-brain python /fly_brain/probe_descending.py
"""
import collections
import os

import numpy as np
from flybrain import FlyBrain

SEEDS, SECONDS, WARMUP = 4, 2.0, 0.5
EYE_DRIVE, AMOUNT = 0.45, 0.8
STIMULI = {"loom_L": ("L", ["LC4", "LPLC2"]), "loom_R": ("R", ["LC4", "LPLC2"]),
           "chase_L": ("L", ["LC10a"]), "chase_R": ("R", ["LC10a"])}

d = os.environ["FLY_DATA"]
z = np.load(os.path.join(d, "brain.npz"), allow_pickle=True)
sc, ct, sd = z["superclass"], z["cell_type"], z["side"]
w = np.load(os.path.join(d, "weights.npz"), allow_pickle=True)
indptr, indices, data = w["indptr"], w["indices"], w["data"]
post_of = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
pre_of = indices
aw = np.abs(data)

dn = np.flatnonzero(sc == "descending_neuron")
mot = np.flatnonzero(sc == "vnc_motor")
isdn = np.zeros(len(sc), bool); isdn[dn] = True
ismot = np.zeros(len(sc), bool); ismot[mot] = True


def shares(mask, keys):
    k, inv = np.unique(keys[mask], return_inverse=True)
    s = np.bincount(inv, weights=aw[mask])
    s /= s.sum()
    return ", ".join("%s %.1f%%" % (k[i], 100 * s[i]) for i in np.argsort(-s)[:8])


print("descending neurons: %d %s, %d types" % (
    len(dn), dict(collections.Counter(sd[dn])), len(set(ct[dn]))))
print("input by source:  " + shares(isdn[post_of], sc[pre_of]))
print("output by target: " + shares(isdn[pre_of], sc[post_of]))
out = isdn[pre_of]
print("output weight into vnc_*: %.1f%%" % (100 * aw[out & np.char.startswith(
    sc[post_of].astype(str), "vnc_")].sum() / aw[out].sum()))
print("descending neurons with a direct motor-neuron target: %d"
      % len(np.unique(pre_of[out & ismot[post_of]])))
print("motor-neuron input by source: " + shares(ismot[post_of], sc[pre_of]))

brain = FlyBrain(device="auto")
brain.tonic, brain.gain = 0.14, 3.0
eye = np.full(len(brain.visual), EYE_DRIVE, np.float32)


def rates(target):
    out = []
    for s in range(SEEDS):
        brain.reset(s)
        cnt = np.zeros(brain.n)
        steps, warm = int(SECONDS / brain.dt), int(WARMUP / brain.dt)
        for k in range(steps):
            if target is not None:
                brain.stimulate(target, AMOUNT)
            fired = brain.step(eye)
            if k >= warm:
                cnt[fired] += 1
        out.append(cnt / ((steps - warm) * brain.dt))
    return np.array(out)


base = rates(None)
print("\nrest: descending mean %.2f spikes/s, %d silent; motor mean %.2f, %d of %d silent"
      % (base[:, dn].mean(), (base[:, dn].mean(0) == 0).sum(),
         base[:, mot].mean(), (base[:, mot].mean(0) == 0).sum(), len(mot)))
for name, (side, types) in STIMULI.items():
    diff = rates(np.flatnonzero(np.isin(ct, types) & (sd == side))) - base
    m = diff.mean(0)
    se = diff.std(0, ddof=1) / np.sqrt(SEEDS) + 1e-9
    moved = (np.abs(m) >= 1.0) & (np.abs(m / se) > 3)
    dmov = dn[moved[dn]]
    top = sorted(dmov, key=lambda i: -abs(m[i]))[:6]
    print("%-8s descending moved %3d %s in %d types; motor moved %d; top: %s" % (
        name, len(dmov), dict(collections.Counter(sd[dmov])), len(set(ct[dmov])),
        moved[mot].sum(), ", ".join("%s_%s %+.1f" % (ct[i], sd[i], m[i]) for i in top)))
