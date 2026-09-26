"""Which fly populations could carry speed and lane position to the readout?

The four-cue encoder hands the brain no speed, and its cos(angle) weighting
deletes the +-90 deg rays that give lane position. A ridge on the cues measures
those as most of what is missing: accel R2 0.002 -> 0.202 with speed, steer
0.703 -> 0.799 with the side walls. This probes where to inject them.

"Drives some descending neurons" is not the bar. Step 5 found the brain
collapses much of its input into a common-mode descending response, so a new
cue only helps if the response it evokes is:

  reliable  the same across two independent halves of the seeds
  graded    weaker at 0.3 than at 0.8 but pointing the same way
  new       not expressible as a mix of the loom/chase responses the encoder
            already produces

Speed is not side-specific, so it is judged on bilateral injection. Lane
position is a left/right contrast, so it is judged on the L-minus-R response.

"new" is measured across halves so noise cannot pass for novelty: the
candidate from one half of the seeds is projected onto the existing responses
from the other half, and `new_rel` is how well the leftover part replicates
between halves. The floor comes from a null "response" -- resting rates on
fresh seeds minus the baseline's -- which is pure spiking noise with the right
statistics. (The existing cues cannot serve as the floor: they are in the
basis, which makes their leftovers anti-correlated by construction.)

Run:  docker compose exec fly-brain python /fly_brain/probe_candidates.py
"""
import numpy as np

from flybrain import FlyBrain

SECONDS = 2.0
WARMUP = 0.5
SEEDS = 8
EYE_DRIVE = 0.45
TONIC, GAIN = 0.14, 3.0  # the live service's regime
STRENGTHS = (0.3, 0.8)

EXISTING = {"loom": ["LC4", "LPLC2"], "chase": ["LC10a"]}
CANDIDATES = {
    # Elementary motion detectors tuned to front-to-back motion, which forward
    # translation produces on both eyes.
    "T4a+T5a": ["T4a", "T5a"],
    # Lobula plate tangential cells integrating horizontal wide-field motion.
    "HS": ["HSE", "HSN", "HSS"],
    # Lobula plate projection neurons reported to respond to translational flow.
    "LPC1": ["LPC1"],
    "LLPC1": ["LLPC1"],
}


def dn_rates(brain, target, strength, slot, n_dn, eye, seed):
    """Spikes/s for every descending neuron over the scored window."""
    brain.reset(seed)
    steps = int(SECONDS / brain.dt)
    warm = int(WARMUP / brain.dt)
    counts = np.zeros(n_dn, np.float64)
    for s in range(steps):
        if target is not None:
            brain.stimulate(target, strength)
        fired = brain.step(eye)
        if s >= warm:
            k = slot[np.asarray(fired)]
            counts += np.bincount(k[k >= 0], minlength=n_dn)
    return counts / ((steps - warm) * brain.dt)


def _resid(v, basis):
    coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
    return v - basis @ coef


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def novelty(dA, dB, basisA, basisB):
    """(new_frac, new_rel) of a response against the existing-cue basis.

    new_frac  share of the response's energy outside the other half's basis
    new_rel   correlation of those leftovers between halves; ~0 for noise
    """
    rA, rB = _resid(dA, basisB), _resid(dB, basisA)
    frac = 0.5 * (rA @ rA / (dA @ dA + 1e-12) + rB @ rB / (dB @ dB + 1e-12))
    return float(frac), _corr(rA, rB)


def main():
    brain = FlyBrain(device="auto")
    brain.tonic, brain.gain = TONIC, GAIN
    ct = np.asarray(brain.cell_type).astype(str)
    sd = np.asarray(brain.side).astype(str)
    eye = np.full(len(brain.visual), EYE_DRIVE, np.float32)

    dn = np.asarray(brain.cells(["descending_neuron"]))
    slot = np.full(brain.n, -1, np.int64)
    slot[dn] = np.arange(len(dn))
    n_dn = len(dn)
    print("descending neurons:", n_dn, flush=True)

    def pop(types, side):
        m = np.isin(ct, types)
        if side != "both":
            m &= sd == side
        return np.flatnonzero(m)

    base = np.stack([dn_rates(brain, None, 0, slot, n_dn, eye, s) for s in range(SEEDS)])
    print("resting mean DN rate %.2f Hz" % base.mean(), flush=True)

    def delta(target, strength):
        """Per-seed change in every DN's rate, paired with the same-seed baseline."""
        return np.stack([dn_rates(brain, target, strength, slot, n_dn, eye, s)
                         for s in range(SEEDS)]) - base

    A, B = slice(0, SEEDS, 2), slice(1, SEEDS, 2)
    resp = {}
    for name, types in list(EXISTING.items()) + list(CANDIDATES.items()):
        strengths = (0.8,) if name in EXISTING else STRENGTHS
        for side in ("L", "R", "both"):
            if name in EXISTING and side == "both":
                continue
            tgt = pop(types, side)
            for st in strengths:
                resp[(name, side, st)] = delta(tgt, st)
            print("  measured %-8s %-4s (%d cells)" % (name, side, len(tgt)), flush=True)

    ex = [(n, s, 0.8) for n in EXISTING for s in "LR"]
    basisA = np.stack([resp[k][A].mean(0) for k in ex], 1)
    basisB = np.stack([resp[k][B].mean(0) for k in ex], 1)

    def summarise(d):
        """Metrics for a per-seed response matrix (seeds x DNs)."""
        m = d.mean(0)
        t = m / (d.std(0, ddof=1) / np.sqrt(len(d)) + 1e-9)
        n_sig = int(((np.abs(m) >= 1.0) & (np.abs(t) > 3)).sum())
        rel = _corr(d[A].mean(0), d[B].mean(0))
        frac, nrel = novelty(d[A].mean(0), d[B].mean(0), basisA, basisB)
        return m, n_sig, float(np.linalg.norm(m)), rel, frac, nrel

    hdr = "%-18s %6s %9s %6s %9s %8s" % ("", "n_sig", "|resp| Hz", "rel", "new_frac", "new_rel")

    print("\n=== existing cues, strength 0.8 ===")
    print("%-18s %6s %9s %6s" % ("", "n_sig", "|resp| Hz", "rel"))
    for k in ex:
        _, n_sig, norm, rel, _, _ = summarise(resp[k])
        print("%-18s %6d %9.1f %6.2f" % ("%s_%s" % k[:2], n_sig, norm, rel))
    for n in EXISTING:
        _, n_sig, norm, rel, _, _ = summarise(resp[(n, "L", 0.8)] - resp[(n, "R", 0.8)])
        print("%-18s %6d %9.1f %6.2f" % (n + " L-R", n_sig, norm, rel))

    # Fresh seeds, not a permutation of `base`: permuted differences sum to
    # exactly zero over the seeds and would report a floor of nothing.
    null = np.stack([dn_rates(brain, None, 0, slot, n_dn, eye, SEEDS + s)
                     for s in range(SEEDS)]) - base
    _, n_sig, norm, rel, frac, nrel = summarise(null)
    print("\n=== null: resting rates on fresh seeds (the floor) ===")
    print(hdr)
    print("%-18s %6d %9.1f %6.2f %9.2f %8.2f" % ("null", n_sig, norm, rel, frac, nrel))

    names = [(dn[i], ct[dn[i]], sd[dn[i]]) for i in range(n_dn)]

    def top(m, k=5):
        o = np.argsort(-np.abs(m))[:k]
        return ", ".join("%s_%s %+.1f" % (names[i][1], names[i][2], m[i]) for i in o)

    print("\n=== SPEED candidates: bilateral injection ===")
    print(hdr + "  graded(0.3/0.8, cos)")
    for n in CANDIDATES:
        hi = summarise(resp[(n, "both", 0.8)])
        lo = summarise(resp[(n, "both", 0.3)])
        print("%-18s %6d %9.1f %6.2f %9.2f %8.2f   %.2f, %.2f"
              % (n, hi[1], hi[2], hi[3], hi[4], hi[5], lo[2] / (hi[2] + 1e-9), _corr(lo[0], hi[0])))
        print("    top DNs: " + top(hi[0]))

    print("\n=== LANE candidates: left minus right response, strength 0.8 ===")
    print(hdr + "  cos(L,R)")
    for n in CANDIDATES:
        dL, dR = resp[(n, "L", 0.8)], resp[(n, "R", 0.8)]
        s = summarise(dL - dR)
        print("%-18s %6d %9.1f %6.2f %9.2f %8.2f   %.2f"
              % (n, s[1], s[2], s[3], s[4], s[5], _corr(dL.mean(0), dR.mean(0))))
        print("    top DNs: " + top(s[0]))


if __name__ == "__main__":
    main()
