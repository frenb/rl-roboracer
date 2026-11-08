# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "pymongo>=4.6"]
# ///
"""
Compare the 31 policy inputs the real JetRacer produces against what the
donut_no_hint policy saw in simulation.

    uv run compare_obs.py --job <job id> --real real_obs.csv [--json out.json]

--job   a TRAIN or EVAL job whose per-step logs (MongoDB `logs`,
        extra_data.car) are the simulated baseline.
--real  N x 31 vectors exactly as the car-side bridge would feed the policy
        (after any clip or scale), in policy order. .csv (header optional)
        or .npy.

Per input it reports simulated and real percentiles, the real/sim median
ratio (a scale error shows up here), the share of real values outside the
simulated 1st-99th percentile range, missing (non-finite) values, the share
pinned at the real maximum (a range cap or missing-return fill), and a
two-sample KS statistic; rows are sorted by the sum of KS, outside and
missing, worst first.
"""
import argparse
import json
import sys

import numpy as np
from bson import ObjectId
from pymongo import MongoClient

# Policy order: DonutCourse.scene_data_array() minus dist_from_traj.
FIELDS = [
    ("speed", None), ("goal_2", None),
    ("left", -90), ("forward_left", -30), ("forward_left_left", -60),
    ("n_27_50", -27.5), ("n_25_00", -25), ("n_22_50", -22.5), ("n_20_00", -20), ("n_17_50", -17.5),
    ("n_15_00", -15), ("n_12_50", -12.5), ("n_10_00", -10), ("n_07_50", -7.5), ("n_05_00", -5), ("n_02_50", -2.5),
    ("forward", 0),
    ("p_02_50", 2.5), ("p_05_00", 5), ("p_07_50", 7.5), ("p_10_00", 10), ("p_12_50", 12.5),
    ("p_15_00", 15), ("p_17_50", 17.5), ("p_20_00", 20), ("p_22_50", 22.5), ("p_25_00", 25), ("p_27_50", 27.5),
    ("forward_right_right", 60), ("forward_right", 30), ("right", 90),
]
LABELS = {"speed": "speed (m/s)", "goal_2": "sideslip (deg/180)"}


def group(name, bearing):
    if bearing is None:
        return "speed" if name == "speed" else "sideslip"
    return "front rays" if abs(bearing) <= 27.5 else "side rays"


def load_sim(uri, job, limit):
    db = MongoClient(uri, serverSelectionTimeoutMS=5000).get_default_database("robotaxi")
    ids = [job]
    if ObjectId.is_valid(job):
        ids.append(ObjectId(job))
    proj = {f"extra_data.car.{n}": 1 for n, _ in FIELDS}
    cur = db.logs.find({"job_id": {"$in": ids}, "extra_data.car.forward": {"$exists": True}}, proj)
    rows = []
    for d in cur.sort("_id", -1).limit(limit):
        car = d["extra_data"]["car"]
        try:
            rows.append([float(car[n]) for n, _ in FIELDS])
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(rows, dtype=np.float64)


def load_real(path):
    if path.endswith(".npy"):
        a = np.load(path).astype(np.float64)
    else:
        with open(path) as f:
            first = f.readline()
        skip = 0 if all(_is_num(c) for c in first.strip().split(",")) else 1
        a = np.genfromtxt(path, delimiter=",", skip_header=skip, dtype=np.float64)
    a = np.atleast_2d(a)
    if a.shape[1] != len(FIELDS):
        sys.exit(f"--real has {a.shape[1]} columns; the policy takes {len(FIELDS)} (speed, sideslip, 29 rays)")
    return a


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return s.strip().lower() in ("nan", "inf", "-inf")


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    if not len(a) or not len(b):
        return None
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))


def pct(x):
    return [round(float(v), 3) for v in np.percentile(x, [1, 50, 99])] if len(x) else None


def compare(sim, real):
    rows = []
    for i, (name, bearing) in enumerate(FIELDS):
        s = sim[:, i][np.isfinite(sim[:, i])]
        r_all = real[:, i]
        r = r_all[np.isfinite(r_all)]
        lo, hi = np.percentile(s, [1, 99])
        out = float(np.mean((r < lo) | (r > hi))) if len(r) else None
        s_med = float(np.median(s))
        ratio = float(np.median(r)) / s_med if len(r) and abs(s_med) > 1e-6 else None
        k = ks(s, r) if len(r) else None
        nonfin = float(np.mean(~np.isfinite(r_all)))
        rows.append({
            "input": LABELS[name] if bearing is None else f"ray {bearing:+g} deg ({name})",
            "index": i,
            "group": group(name, bearing),
            "sim_p1_p50_p99": pct(s),
            "real_p1_p50_p99": pct(r),
            "median_ratio_real_to_sim": round(ratio, 3) if ratio is not None else None,
            "real_outside_sim_range": round(out, 3) if out is not None else None,
            "real_non_finite": round(nonfin, 3),
            "real_pinned_at_max": round(float(np.mean(r == r.max())), 3) if len(r) else None,
            "ks": round(k, 3) if k is not None else None,
            # Distribution shift, plus readings the policy never saw (outside range or missing).
            "gap_score": round((k or 0) + (out or 0) + nonfin, 3),
        })
    rows.sort(key=lambda x: -x["gap_score"])
    groups = {}
    for x in rows:
        g = groups.setdefault(x["group"], {"inputs": 0, "nonfin": [], "pinned": [], "ks": []})
        g["inputs"] += 1
        g["nonfin"].append(x["real_non_finite"])
        g["pinned"].append(x["real_pinned_at_max"] or 0)
        g["ks"].append(x["ks"] or 0)
    summary = {k: {"inputs": v["inputs"], "mean_ks": round(float(np.mean(v["ks"])), 3),
                   "mean_non_finite": round(float(np.mean(v["nonfin"])), 3),
                   "mean_pinned_at_max": round(float(np.mean(v["pinned"])), 3)} for k, v in groups.items()}
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]["mean_ks"]))
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--mongo", default="mongodb://root:example@127.0.0.1:27017/?directConnection=true")
    ap.add_argument("--limit", type=int, default=20000, help="most recent simulated steps to use")
    ap.add_argument("--json", help="also write the full report here")
    a = ap.parse_args()

    sim = load_sim(a.mongo, a.job, a.limit)
    if len(sim) < 100:
        sys.exit(f"only {len(sim)} simulated steps with all 31 inputs for job {a.job}; pick a job with more logs")
    real = load_real(a.real)
    rows, summary = compare(sim, real)
    report = {"job": a.job, "sim_steps": len(sim), "real_steps": len(real), "by_group": summary, "inputs": rows}

    print(f"sim steps {len(sim)}, real steps {len(real)}")
    for k, v in summary.items():
        print(f"  {k:<10} KS {v['mean_ks']:.2f}  missing {v['mean_non_finite']:.0%}  pinned at max {v['mean_pinned_at_max']:.0%}")
    print(f"\n{'input':<34}{'sim p1/p50/p99':<26}{'real p1/p50/p99':<26}"
          f"{'med r/s':>8}{'outside':>8}{'nonfin':>8}{'pinned':>8}{'KS':>6}")
    for x in rows:
        ratio = x["median_ratio_real_to_sim"]
        print(f"{x['input']:<34}{str(x['sim_p1_p50_p99']):<26}{str(x['real_p1_p50_p99']):<26}"
              f"{(f'{ratio:.2f}' if ratio is not None else '-'):>8}{(x['real_outside_sim_range'] or 0):>8.0%}"
              f"{x['real_non_finite']:>8.0%}{(x['real_pinned_at_max'] or 0):>8.0%}{(x['ks'] or 0):>6.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
