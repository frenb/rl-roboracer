"""Step 2 of docs/flybrain-driver-plan.md: does the wiring do the right thing
on the correct side?

A faithful port of fly.ai's inject.py, with two changes:

  * Sides come from `brain.side` instead of re-reading
    body-annotations-male-cns-v1.0.feather. That file only exists after
    `flybrain build`; we used `flybrain download`. Verified identical for
    LC4/LPLC2/LC10a/DNp01/DNa02 (see _probe_api.py).
  * Both sides are stimulated, not just left. inject.py only tests L, but our
    encoder drives both, so side-specificity is checked in both directions.

Expected (fly.ai's finding 3, tonic 0.14 / gain 3.0):
    loom  on one side -> that side's DNp01 +17..25 spikes/s, other side flat
    chase on one side -> that side's DNa02 +1.4..3.7 spikes/s, other side flat

Run:  docker compose exec fly-brain python /fly_brain/step2_inject.py
"""
import numpy as np

from flybrain import FlyBrain

SECONDS = 2.0
WARMUP = 0.5
SEEDS = 6
EYE_DRIVE = 0.45  # inject.py holds the photoreceptors at a constant 0.45
STRENGTHS = (0.3, 0.8)
READOUT_TYPES = ["DNp01", "DNp10", "DNa02", "DNg13", "MDN", "DNg100", "DNg11", "pIP10"]
STIMULI = {"loom": ["LC4", "LPLC2"], "chase": ["LC10a"]}
# The regime fly.ai reports finding 3 under, plus Fly64's for contrast.
REGIMES = [(0.14, 3.0), (0.18, 1.5)]


def rates(brain, target, strength, readout, eye, seed):
    """Mean spikes/s per neuron for each readout group over the scored window."""
    brain.reset(seed)
    steps = int(SECONDS / brain.dt)
    warm = int(WARMUP / brain.dt)
    counts = {k: 0 for k in readout}
    hit = np.zeros(brain.n, bool)
    for s in range(steps):
        if target is not None:
            brain.stimulate(target, strength)
        fired = brain.step(eye)
        if s >= warm:
            hit[:] = False
            hit[fired] = True
            for k, idx in readout.items():
                counts[k] += hit[idx].sum()
    window = (steps - warm) * brain.dt
    return {k: counts[k] / len(readout[k]) / window for k in readout}


def main():
    brain = FlyBrain(device="auto")
    ct = np.asarray(brain.cell_type)
    sd = np.asarray(brain.side)
    eye = np.full(len(brain.visual), EYE_DRIVE, np.float32)

    readout = {}
    for t in READOUT_TYPES:
        for s in "LR":
            idx = np.flatnonzero((ct == t) & (sd == s))
            if len(idx):
                readout[f"{t}_{s}"] = idx

    targets = {}
    for name, types in STIMULI.items():
        for s in "LR":
            idx = np.flatnonzero(np.isin(ct, types) & (sd == s))
            if len(idx):
                targets[f"{name}_{s}"] = idx

    print("target sizes:", {k: len(v) for k, v in targets.items()})
    print("readout sizes:", {k: len(v) for k, v in readout.items()})

    for tonic, gain in REGIMES:
        brain.tonic, brain.gain = tonic, gain
        base = [rates(brain, None, 0, readout, eye, s) for s in range(SEEDS)]
        rest = np.mean([np.mean([base[i][k] for k in readout]) for i in range(SEEDS)])
        print(f"\n=== tonic {tonic} gain {gain} "
              f"(mean readout rate at rest {rest:.1f} Hz) ===")

        for name, target in targets.items():
            for strength in STRENGTHS:
                stim = [rates(brain, target, strength, readout, eye, s)
                        for s in range(SEEDS)]
                parts = []
                for k in readout:
                    d = np.array([stim[i][k] - base[i][k] for i in range(SEEDS)])
                    t = d.mean() / (d.std(ddof=1) / np.sqrt(SEEDS) + 1e-9)
                    mark = "*" if abs(t) > 3 and abs(d.mean()) >= 1 else " "
                    parts.append(f"{k}:{d.mean():+5.1f}{mark}")
                print(f"  stim {name:8s} x{strength}: " + " ".join(parts),
                      flush=True)

    # The pass/fail the plan actually cares about, at the working regime.
    print("\n=== side-specificity verdict (tonic 0.14 gain 3.0, strength 0.8) ===")
    brain.tonic, brain.gain = 0.14, 3.0
    base = [rates(brain, None, 0, readout, eye, s) for s in range(SEEDS)]
    for stim_name, out_type, lo, hi in (("loom", "DNp01", 17.0, 25.0),
                                        ("chase", "DNa02", 1.4, 3.7)):
        for s, other in (("L", "R"), ("R", "L")):
            tgt = targets.get(f"{stim_name}_{s}")
            if tgt is None:
                continue
            run = [rates(brain, tgt, 0.8, readout, eye, sd_) for sd_ in range(SEEDS)]
            same = np.mean([run[i][f"{out_type}_{s}"] - base[i][f"{out_type}_{s}"]
                            for i in range(SEEDS)])
            opp = np.mean([run[i][f"{out_type}_{other}"] - base[i][f"{out_type}_{other}"]
                           for i in range(SEEDS)])
            ok = (same > lo * 0.5) and (abs(opp) < max(1.0, abs(same) * 0.25))
            print(f"  {stim_name}_{s:1s} -> {out_type}_{s}:{same:+6.1f}  "
                  f"{out_type}_{other}:{opp:+6.1f}  "
                  f"(fly.ai reports +{lo:.1f}..{hi:.1f} same side, ~0 other)  "
                  f"{'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
