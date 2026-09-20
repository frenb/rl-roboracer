"""Turn 29 raycast distances into drive for four fly neuron populations.

Step 4 of docs/flybrain-driver-plan.md. This is the whole translation between
the car's world and the fly's, and the only part of the plan with no reference
implementation to copy.

Four scalars per control step: looming and chase, left and right. Looming goes
to LC4 + LPLC2 on the matching side, chase to LC10a. Step 2 verified that those
populations drive DNp01 and DNa02 side-specifically, which is the signal the
readout will try to recover.

Pure NumPy on purpose: it runs on the trainer, but nothing stops it running
inside the fly-brain container.
"""
import numpy as np

# Ray angles in degrees, in observation order, from CarController's
# `directions` enum (the declaration order is the wire order) cross-checked
# against SetUpDirectionToAngle(). Negative is left.
#
# NOTE the two transpositions the plan calls out: entries 1/2 are -30/-60 and
# entries 26/27 are +60/+30. This list is the reason the encoder can weight a
# ray by its angle at all -- do not assume monotonicity.
RAY_ANGLES_DEG = np.array([
    -90.0, -30.0, -60.0,
    -27.5, -25.0, -22.5, -20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5,
    0.0,
    2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5,
    60.0, 30.0, 90.0,
], dtype=np.float32)

N_RAYS = len(RAY_ANGLES_DEG)
RAY_SLICE = slice(2, 2 + N_RAYS)  # into the 31-D donut_no_hint observation

# cos(angle) as the per-ray weight. This is what stops the encoder reading the
# track wall instead of the road: measured on the demo corpus, the +-90 rays sit
# at a median 6.3 m and the +-60 rays at 7.4 m against 25 m straight ahead, so
# an unweighted per-side minimum is the wall, every frame, on both sides.
# cos(90 deg) is exactly 0 and cos(60 deg) is 0.5, which suppresses them.
RAY_WEIGHT = np.cos(np.radians(RAY_ANGLES_DEG)).astype(np.float32)

LEFT = RAY_ANGLES_DEG < 0.0
RIGHT = RAY_ANGLES_DEG > 0.0
# The 0 deg ray belongs to neither side; it would cancel out of the chase
# contrast anyway and gives looming no left/right information.

DT = 0.1  # one env step in simulated seconds (SceneDataPublisher, 10 Hz)

# Ranges are in Unity metres. p50 is ~18 m and p90 ~47 m over the demo corpus,
# so 30 m reads as "fully open" without clipping most of the useful variation.
R_OPEN = 30.0
# Guards 1/r when a ray reports a near-zero range.
R_FLOOR = 0.5

# A ray reporting exactly 0 never hit anything since the car was created:
# CarController.DrawRay leaves distToClosestObjects[d] untouched on a miss, and
# the array starts zeroed. Treat it as "no return", i.e. maximally open, rather
# than as an obstacle parked on the bumper. 0.03% of readings in the corpus.
R_NO_RETURN = 0.0

# The rays rotate with the car, so a ray sliding off a wall edge reports a jump
# that is not motion: 23.5% of per-ray steps in the corpus move more than 1.0 m
# while the 6.7 m/s top speed allows only 0.67 m per step. Every obstacle in the
# course is static, so true closing along a ray cannot exceed the car's own
# speed; clamp to that (with slack for sideslip and yaw) and the artefacts drop
# out without a filter that would also lag the real approach.
CLOSING_SLACK = 1.5
SPEED_FLOOR = 0.2

# Gains mapping a raw cue to injected voltage. Step 2 found 0.3 a weak
# injection and 0.8 a strong one, so MAX_AMOUNT sits at the strong end.
#
# Both gains put the cue's 99th percentile over the demo corpus at MAX_AMOUNT,
# so the common frame uses the dynamic range instead of clipping: measured
# p99 was 0.77 for looming and 0.56 for chase.
MAX_AMOUNT = 0.8
LOOM_GAIN = 1.04
CHASE_GAIN = 1.44

POPULATIONS = ("loom_L", "loom_R", "chase_L", "chase_R")
# What each cue drives. Resolve with FlyBrainClient.cells(types, side).
POPULATION_CELLS = {
    "loom_L": (["LC4", "LPLC2"], "L"),
    "loom_R": (["LC4", "LPLC2"], "R"),
    "chase_L": (["LC10a"], "L"),
    "chase_R": (["LC10a"], "R"),
}


class RayEncoder(object):
    """Stateful: looming needs the previous frame's ranges. Reset per episode."""

    def __init__(self, dt=DT, r_open=R_OPEN, loom_gain=LOOM_GAIN,
                 chase_gain=CHASE_GAIN, max_amount=MAX_AMOUNT):
        self.dt = float(dt)
        self.r_open = float(r_open)
        self.loom_gain = float(loom_gain)
        self.chase_gain = float(chase_gain)
        self.max_amount = float(max_amount)
        self._prev = None

    def reset(self):
        self._prev = None

    def cues(self, obs):
        """Raw, ungained cues for one observation. Returns a dict of 4 floats.

        `obs` is the full 31-D donut_no_hint vector. Speed (index 0) is needed
        to bound the closing rate, so the rays alone are not enough.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != 2 + N_RAYS:
            raise ValueError("expected the %d-D observation, got %d"
                             % (2 + N_RAYS, obs.shape[-1]))
        speed = float(obs[0])
        rays = obs[RAY_SLICE]

        # A missed ray means nothing is out there, not something at zero range.
        r = np.where(rays <= R_NO_RETURN, self.r_open, rays)
        r = np.maximum(r, R_FLOOR)

        # Looming as inverse time-to-contact, -rdot/r, approach only. This is
        # deliberately not a proximity term: LPLC2 is a *looming* detector, so
        # a wall parked at 1 m and a wall rushing in from 1 m must not look
        # alike. On the first step of an episode there is no previous frame and
        # looming is zero.
        if self._prev is None:
            inv_tau = np.zeros(N_RAYS, np.float32)
        else:
            closing = (self._prev - r) / self.dt      # +ve when approaching
            cap = CLOSING_SLACK * max(abs(speed), SPEED_FLOOR)
            inv_tau = np.clip(closing, 0.0, cap) / r
        self._prev = r

        w_inv_tau = RAY_WEIGHT * inv_tau
        loom_l = float(w_inv_tau[LEFT].max())
        loom_r = float(w_inv_tau[RIGHT].max())

        # Chase is a contrast, not a level: only the *more* open side is driven,
        # so straight road drives neither and the asymmetry is what steers.
        open_ = np.clip(r / self.r_open, 0.0, 1.0)
        wl, wr = RAY_WEIGHT[LEFT], RAY_WEIGHT[RIGHT]
        open_l = float((wl * open_[LEFT]).sum() / wl.sum())
        open_r = float((wr * open_[RIGHT]).sum() / wr.sum())
        d = open_l - open_r

        return {"loom_L": loom_l, "loom_R": loom_r,
                "chase_L": max(d, 0.0), "chase_R": max(-d, 0.0)}

    # cues() is the ONLY stateful call. Everything below is a pure function of
    # a cue dict, so there is no way to advance the frame twice by asking for
    # both the cues and the injection -- which silently zeroes looming, since
    # the second call sees no change in range.

    def amounts(self, cues):
        """Cues scaled to injection voltages, as a dict of 4 floats in 0..max."""
        g = {"loom_L": self.loom_gain, "loom_R": self.loom_gain,
             "chase_L": self.chase_gain, "chase_R": self.chase_gain}
        return {k: float(np.clip(v * g[k], 0.0, self.max_amount))
                for k, v in cues.items()}

    def inject(self, cues, cell_idx):
        """Ready to hand to FlyBrainClient.step.

        `cell_idx` maps each population name to its neuron indices, e.g.
        {"loom_L": client.cells(["LC4","LPLC2"], side="L"), ...}. Populations
        whose cue is zero are dropped rather than injected with 0.0, which
        keeps the request small on the common straight-road frame.
        """
        return [(cell_idx[k], a)
                for k, a in self.amounts(cues).items() if a > 0.0]

    def encode(self, obs, cell_idx):
        """One frame: advance, and return (cues, injection). The usual entry."""
        cues = self.cues(obs)
        return cues, self.inject(cues, cell_idx)


def resolve_cells(client):
    """Look up the four populations' neuron indices over gRPC, once."""
    return {name: client.cells(types=types, side=side)
            for name, (types, side) in POPULATION_CELLS.items()}


# --------------------------------------------------------------------------
# Retinotopic encoder
# --------------------------------------------------------------------------
# Step 5 measured the four-scalar encoder above as the binding constraint: a
# ridge fit on its four cues scores R2 0.687 for steering against 0.747 on the
# raw rays, and the brain then returns 0.620. No readout recovers what the
# encoder discarded, so raising that 0.687 is the only change with room to pay
# off.
#
# This keeps the same two cues but stops collapsing each side to one number.
# The rays are split into angular sectors and every sector drives its own slice
# of the LC populations, which is what "retinotopic" means here: nearby
# directions excite nearby cells instead of the whole eye at once.
#
# Two deliberate differences from the four-scalar version:
#   - No cos(angle) weighting. That existed to stop the +-90 wall returns
#     swamping a single per-side aggregate. With one channel per sector the
#     readout can simply down-weight the lateral sectors, and cos weighting
#     would instead delete them.
#   - Openness is a left-right contrast between mirrored sectors, not an
#     absolute level. Absolute levels were tried first and cost 0.13 of
#     steering R2 against the flat encoder (0.488 vs 0.620) despite handing the
#     brain a far better input. The reason is measured: driving any single
#     sector produces a largely common-mode descending response, with
#     within-eye similarity 0.476 against across-eye 0.460, so the brain barely
#     distinguishes sectors but reads left/right contrast well. Absolute
#     openness is ~0.6 on both sides and buries that contrast in common mode;
#     mirroring restores it at full amplitude while keeping sector resolution.

SECTORS_PER_SIDE = 7  # 14 rays per side -> 2 rays per sector


def _sector_ray_groups(k=SECTORS_PER_SIDE):
    """Ray indices per sector, ordered lateral -> medial, for each side.

    Returns [(side, ray_indices), ...]. The 0 deg ray joins the innermost
    sector of both sides: it is the single most important direction and
    belongs to neither eye exclusively.
    """
    groups = []
    for side, mask in (("L", LEFT), ("R", RIGHT)):
        rays = np.flatnonzero(mask)
        # Sort by |angle| descending so sector 0 is the most lateral.
        rays = rays[np.argsort(-np.abs(RAY_ANGLES_DEG[rays]))]
        for s, part in enumerate(np.array_split(rays, k)):
            part = list(part)
            if s == k - 1:
                part.append(int(np.flatnonzero(RAY_ANGLES_DEG == 0.0)[0]))
            groups.append((side, np.array(part, np.int64)))
    return groups


SECTOR_GROUPS = _sector_ray_groups()
N_SECTORS = len(SECTOR_GROUPS)

# Calibrated the same way as the four-scalar gains: corpus p99 -> MAX_AMOUNT.
# Measured p99 was 0.74 for per-sector looming and 0.72 for the openness
# contrast, whose median is 0 -- only the more open side of each mirrored pair
# is driven at all.
RETINO_LOOM_GAIN = 1.09
RETINO_OPEN_GAIN = 1.11


class RetinotopicEncoder(object):
    """Per-sector looming and openness. Same cue definitions, finer resolution."""

    def __init__(self, dt=DT, r_open=R_OPEN, max_amount=MAX_AMOUNT,
                 loom_gain=RETINO_LOOM_GAIN, open_gain=RETINO_OPEN_GAIN):
        self.dt = float(dt)
        self.r_open = float(r_open)
        self.max_amount = float(max_amount)
        self.loom_gain = float(loom_gain)
        self.open_gain = float(open_gain)
        self._prev = None

    def reset(self):
        self._prev = None

    def cues(self, obs):
        """Returns (loom, openness), each float32[N_SECTORS]."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != 2 + N_RAYS:
            raise ValueError("expected the %d-D observation, got %d"
                             % (2 + N_RAYS, obs.shape[-1]))
        speed = float(obs[0])
        rays = obs[RAY_SLICE]

        r = np.where(rays <= R_NO_RETURN, self.r_open, rays)
        r = np.maximum(r, R_FLOOR)

        if self._prev is None:
            inv_tau = np.zeros(N_RAYS, np.float32)
        else:
            cap = CLOSING_SLACK * max(abs(speed), SPEED_FLOOR)
            inv_tau = np.clip((self._prev - r) / self.dt, 0.0, cap) / r
        self._prev = r

        open_ray = np.clip(r / self.r_open, 0.0, 1.0)
        loom = np.empty(N_SECTORS, np.float32)
        raw_open = np.empty(N_SECTORS, np.float32)
        for i, (_, group) in enumerate(SECTOR_GROUPS):
            loom[i] = inv_tau[group].max()
            raw_open[i] = open_ray[group].mean()

        # Mirror sector s of one eye against sector s of the other, and keep
        # only the positive part on each side. Exactly the flat encoder's chase
        # contrast, computed per sector instead of per side.
        k = N_SECTORS // 2
        d = raw_open[:k] - raw_open[k:]
        openness = np.concatenate([np.maximum(d, 0.0), np.maximum(-d, 0.0)])
        return loom, openness.astype(np.float32)

    def features(self, obs):
        """Both cue vectors concatenated, for fitting a readout directly."""
        loom, openness = self.cues(obs)
        return np.concatenate([loom, openness])

    def inject(self, cues, sector_cells):
        """Dense per-neuron injections, one per cue type.

        `sector_cells` maps "loom"/"open" to a list of N_SECTORS index arrays.
        Every sector's drive is packed into a single dense injection so the
        brain pays one scatter per cue type instead of one per sector, which
        measured 7.75 ms against 21.66 ms per control step.
        """
        loom, openness = cues
        out = []
        for name, vals, gain in (("loom", loom, self.loom_gain),
                                 ("open", openness, self.open_gain)):
            idx, amt = [], []
            for s, cells in enumerate(sector_cells[name]):
                a = min(float(vals[s]) * gain, self.max_amount)
                if a <= 0.0 or not len(cells):
                    continue
                idx.append(cells)
                amt.append(np.full(len(cells), a, np.float32))
            if idx:
                out.append((np.concatenate(idx), np.concatenate(amt)))
        return out

    def encode(self, obs, sector_cells):
        cues = self.cues(obs)
        return cues, self.inject(cues, sector_cells)


def resolve_sector_cells(client, k=SECTORS_PER_SIDE):
    """Neuron indices per sector, ordered to match SECTOR_GROUPS (L then R)."""
    out = {"loom": [], "open": []}
    for side in ("L", "R"):
        out["loom"] += client.sectors(["LC4", "LPLC2"], side=side, k=k)
        out["open"] += client.sectors(["LC10a"], side=side, k=k)
    return out
