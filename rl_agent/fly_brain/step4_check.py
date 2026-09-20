"""Step 4 acceptance: synthetic scans must move the expected side.

From docs/flybrain-driver-plan.md: "a wall approaching on the left raises left
DNp01, open space on the right raises right DNa02."

Builds observations by hand rather than replaying the corpus, so each cue is
isolated. Compares every scenario against a symmetric baseline, the way step 2
compared stimulated runs against an unstimulated one.

    docker compose exec -w /python_ws/src sim-controller python -m fly_brain.step4_check
"""
import numpy as np

from fly_brain.client import FlyBrainClient, SUBSTEPS
from fly_brain.encoder import N_RAYS, RayEncoder, resolve_cells

STEPS = 40          # control steps per scenario
SETTLE = 20         # ignore this many while the trace fills
SEED = 0
SPEED = 4.0         # m/s, near the corpus median
MID = 20.0          # a neutral range, close to the corpus median


def make_obs(rays, speed=SPEED):
    obs = np.empty(2 + N_RAYS, np.float32)
    obs[0] = speed
    obs[1] = 0.0      # sideslip, unused by the encoder
    obs[2:] = rays
    return obs


def scenario_symmetric():
    """Baseline: nothing approaching, both sides equally open."""
    return [make_obs(np.full(N_RAYS, MID)) for _ in range(STEPS)]


def scenario_wall_closing(side):
    """A wall rushing in on one side; the other side holds steady.

    It must still be closing during the measurement window: looming is a rate,
    so a wall that arrives and stops is silent by construction. Closing 0.5 m
    per step from 25 m leaves 5.5 m at the last step, and stays under the
    encoder's speed cap of 1.5 * 4.0 m/s.
    """
    from fly_brain.encoder import LEFT, RIGHT
    mask = LEFT if side == "L" else RIGHT
    out = []
    for t in range(STEPS):
        rays = np.full(N_RAYS, MID, np.float32)
        rays[mask] = 25.0 - 0.5 * t
        out.append(make_obs(rays))
    return out


def scenario_open(side):
    """One side wide open, the other tight. Static, so looming stays silent."""
    from fly_brain.encoder import LEFT, RIGHT
    mask = LEFT if side == "L" else RIGHT
    rays = np.full(N_RAYS, 8.0, np.float32)
    rays[mask] = 45.0
    return [make_obs(rays) for _ in range(STEPS)]


def run(client, cells, frames):
    """Replay frames through the brain, return the mean settled trace."""
    enc = RayEncoder()
    enc.reset()
    client.reset(seed=SEED)
    traces, cues = [], []
    for i, obs in enumerate(frames):
        cue, inject = enc.encode(obs, cells)
        cues.append(cue)
        trace, _, _ = client.step(inject, substeps=SUBSTEPS)
        if i >= SETTLE:
            traces.append(trace)
    return np.mean(traces, axis=0), cues[-1]


def main():
    c = FlyBrainClient()
    cells = resolve_cells(c)
    print("populations: " + ", ".join("%s=%d" % (k, len(v))
                                      for k, v in sorted(cells.items())))

    # Trace column for a given cell type and side. Trace.idx is
    # brain.cells(["descending_neuron"]), so the same call reproduces it.
    desc = c.cells(types=["descending_neuron"])
    slot = {n: i for i, n in enumerate(desc)}

    def cols(cell_type, side):
        idx = c.cells(types=[cell_type], side=side)
        return np.array([slot[i] for i in idx if i in slot], np.int64)

    readouts = {}
    for t in ("DNp01", "DNa02"):
        for s in ("L", "R"):
            readouts["%s_%s" % (t, s)] = cols(t, s)
    print("readouts: " + ", ".join("%s=%d" % (k, len(v))
                                   for k, v in sorted(readouts.items())))

    base, _ = run(c, cells, scenario_symmetric())

    scenarios = [
        ("wall closing LEFT", scenario_wall_closing("L"), "DNp01", "L"),
        ("wall closing RIGHT", scenario_wall_closing("R"), "DNp01", "R"),
        ("open space LEFT", scenario_open("L"), "DNa02", "L"),
        ("open space RIGHT", scenario_open("R"), "DNa02", "R"),
    ]

    print()
    print("%-20s %-22s %9s %9s %9s" % ("scenario", "cue", "dn_L", "dn_R", "verdict"))
    ok = True
    for name, frames, dn, expect in scenarios:
        trace, cue = run(c, cells, frames)
        d = trace - base
        l = float(d[readouts["%s_L" % dn]].mean())
        r = float(d[readouts["%s_R" % dn]].mean())
        # A tie must not pass: two silent neurons would otherwise "agree" with
        # whichever side the scenario happened to name.
        margin = (l - r) if expect == "L" else (r - l)
        good = margin > 1e-6
        ok = ok and good
        cue_s = " ".join("%s=%.2f" % (k.split("_")[0][:2] + k[-1], v)
                         for k, v in sorted(cue.items()) if v > 0.01)
        print("%-20s %-22s %+9.4f %+9.4f %9s"
              % (name, cue_s or "(none)", l, r,
                 ("OK %s" % dn) if good else "WRONG SIDE"))

    print()
    print("STEP 4 %s" % ("PASS" if ok else "FAIL"))
    c.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
