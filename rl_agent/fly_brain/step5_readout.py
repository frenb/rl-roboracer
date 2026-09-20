"""Step 5: fit a ridge readout on the descending trace and report held-out R2.

From docs/flybrain-driver-plan.md, this is the go/no-go before any Unity work:
does the frozen connectome's descending activity contain enough about the scene
to recover the expert's action? Behaviour cloning with no new data collection,
no live loop and no RL.

    docker compose exec -w /python_ws/src sim-controller python -m fly_brain.step5_readout

The replay is the expensive part (~10 ms per frame, serial -- FlyBrain(batch=N)
runs N independent flies, not N sequential frames), so traces are cached and
any refit afterwards is free.
"""
import glob
import os
import time

import numpy as np

# Episodes are exactly 1000 steps: speed drops to ~0 a few rows after every
# multiple of 1000, and at 385 of those boundaries nearly every ray jumps at
# once. The brain is stateful and the encoder holds the previous frame's
# ranges, so both must reset here -- otherwise 500 teleports are replayed as
# real motion.
EPISODE_LEN = 1000
N_EPISODES = int(os.environ.get("FLY_EPISODES", "100"))
HELDOUT_EPISODES = max(1, N_EPISODES // 5)

CORPUS = "/tfrecords/job_64168c1b58d4d8ccdb76e721"
CACHE = "/tmp/fly_step5_trace_%dep.npz" % N_EPISODES
OUT = "/tmp/fly_readout_%dep.npz" % N_EPISODES

LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)


def load_corpus(n_rows):
    import tensorflow as tf

    import collect_training_data as ctd
    ctd.set_observation_size(31)

    obs, act = [], []
    for rec in tf.data.TFRecordDataset(sorted(glob.glob(os.path.join(CORPUS, "*")))):
        p = tf.io.parse_single_example(rec, ctd.feature_description)
        obs.append(p["observation"].numpy())
        act.append(p["action"].numpy())
        if len(obs) >= n_rows:
            break
    return np.stack(obs)[:, 1:], np.stack(act)


def replay(obs):
    """Push every observation through the frozen brain, one episode at a time."""
    from fly_brain.client import FlyBrainClient, SUBSTEPS
    from fly_brain.encoder import RayEncoder, resolve_cells

    client = FlyBrainClient()
    cells = resolve_cells(client)
    enc = RayEncoder()

    n = len(obs)
    traces = np.empty((n, client.info.trace_len), np.float32)
    t0 = time.time()
    for i in range(n):
        if i % EPISODE_LEN == 0:
            # Seed per episode so the brain's noise is reproducible.
            client.reset(seed=i // EPISODE_LEN)
            enc.reset()
        _, inject = enc.encode(obs[i], cells)
        traces[i], _, _ = client.step(inject, substeps=SUBSTEPS)

        if i and i % 5000 == 0:
            el = time.time() - t0
            print("  %6d/%d  %.1f min elapsed, %.1f min left"
                  % (i, n, el / 60.0, el / i * (n - i) / 60.0), flush=True)
    client.close()
    print("  replay done in %.1f min" % ((time.time() - t0) / 60.0))
    return traces


def ridge_r2(xtr, ytr, xte, yte, lambdas=LAMBDAS):
    """Standardised ridge. Returns (best_lambda, r2_per_channel, weights, mu, sd)."""
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-8] = 1.0
    a = ((xtr - mu) / sd).astype(np.float64)
    b = ((xte - mu) / sd).astype(np.float64)
    ym = ytr.mean(0)
    yc = (ytr - ym).astype(np.float64)

    # Gram once, reused for every lambda.
    g = a.T @ a
    rhs = a.T @ yc
    eye = np.eye(g.shape[0])

    # Hold out the last fifth of the training episodes to pick lambda, so the
    # test split is never touched during selection.
    cut = int(len(a) * 0.8)
    gv = a[:cut].T @ a[:cut]
    rv = a[:cut].T @ yc[:cut]
    best, best_lam = -np.inf, lambdas[0]
    for lam in lambdas:
        w = np.linalg.solve(gv + lam * eye, rv)
        pred = a[cut:] @ w + ym
        s = _r2(ytr[cut:], pred).mean()
        if s > best:
            best, best_lam = s, lam

    w = np.linalg.solve(g + best_lam * eye, rhs)
    return best_lam, _r2(yte, b @ w + ym), w, mu, sd, ym


def _r2(y, pred):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    n_rows = N_EPISODES * EPISODE_LEN
    print("step 5: %d episodes (%d rows), %d held out"
          % (N_EPISODES, n_rows, HELDOUT_EPISODES))

    if os.path.exists(CACHE):
        print("loading cached traces from %s" % CACHE)
        z = np.load(CACHE)
        traces, obs, act = z["traces"], z["obs"], z["act"]
    else:
        print("reading corpus ...")
        obs, act = load_corpus(n_rows)
        print("  obs %s act %s" % (obs.shape, act.shape))
        print("replaying through the frozen brain ...")
        traces = replay(obs)
        np.savez(CACHE, traces=traces, obs=obs, act=act)
        print("cached to %s" % CACHE)

    # Split by EPISODE, never by row: consecutive rows are 0.1 s apart and
    # highly correlated, so a random row split would leak the answer across it.
    split = (N_EPISODES - HELDOUT_EPISODES) * EPISODE_LEN
    tr, te = slice(0, split), slice(split, n_rows)
    print("train rows %d, held-out rows %d" % (split, n_rows - split))

    # Controls. Without these the trace's R2 is uninterpretable: the question
    # is not "is it above zero" but "does the brain add anything to what the
    # encoder already hands it".
    from fly_brain.encoder import POPULATIONS, RayEncoder
    enc = RayEncoder()
    cues = np.empty((len(obs), len(POPULATIONS)), np.float32)
    for i, o in enumerate(obs):
        if i % EPISODE_LEN == 0:
            enc.reset()
        c = enc.cues(o)
        cues[i] = [c[k] for k in POPULATIONS]
    chase = (cues[:, POPULATIONS.index("chase_L")]
             - cues[:, POPULATIONS.index("chase_R")]).reshape(-1, 1)

    feature_sets = [
        ("descending trace, 1314 features", traces),
        ("the 4 encoder cues = the brain's whole input", cues),
        ("chase asymmetry alone, 1 feature", chase),
        ("raw 31-D observation (upper reference)", obs),
    ]

    print()
    print("%-46s %8s %10s %10s" % ("features", "lambda", "R2 accel", "R2 steer"))
    results = {}
    for name, x in feature_sets:
        lam, r2, w, mu, sd, ym = ridge_r2(x[tr], act[tr], x[te], act[te])
        results[name] = (r2, w, mu, sd, ym, lam)
        print("%-46s %8.3g %10.4f %10.4f" % (name, lam, r2[0], r2[1]))

    r2, w, mu, sd, ym, lam = results[feature_sets[0][0]]
    np.savez(OUT, w=w, mu=mu, sd=sd, y_mean=ym, lam=lam, r2=r2)
    print()
    print("readout saved to %s (w %s)" % (OUT, w.shape))

    brain = results[feature_sets[0][0]][0]
    inp = results[feature_sets[1][0]][0]
    print()
    print("STEP 5 %s: steering R2 = %.4f, meaningfully above zero."
          % ("PASS" if brain[1] > 0.2 else "FAIL", brain[1]))
    # The threshold alone is not the useful reading. Compare each channel
    # against the brain's own input: a frozen reservoir is only worth its cost
    # where it returns more than it was given.
    print()
    for i, ch in enumerate(("accel", "steer")):
        delta = brain[i] - inp[i]
        verdict = ("the brain ADDS %.3f over its own input" % delta if delta > 0
                   else "the brain LOSES %.3f against its own input" % -delta)
        print("  %-6s trace %.4f vs encoder cues %.4f  ->  %s"
              % (ch, brain[i], inp[i], verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
