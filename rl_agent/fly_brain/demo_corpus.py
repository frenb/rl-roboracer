"""Expert demos for the fly courses: a scene corpus replayed through the brain.

The fly courses observe the descending-neuron trace, which depends on the
whole episode so far, so it cannot be recorded directly. It can be rebuilt:
the brain never sees the action, so pushing a recorded episode's scene frames
through the same encoder and a freshly reset brain, in order, yields exactly
the traces the policy would have observed. Step 5 established this offline.

``ensure(demo_job_id, course_type)`` turns a DEMO job's 32-wide scene records
into a trace corpus at ``/tfrecords/fly_trace_<course>_<job id>``, built once
and reused. Built from do_job at TRAIN start, where nothing else is stepping
the brain service.

    docker compose exec -w /python_ws/src sim-controller \\
        python -m fly_brain.demo_corpus <demo_job_id> fly_donut_flow
"""
import datetime
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

TFRECORD_ROOT = "/tfrecords"
SCENE_WIDTH = 32
# tf_agents' StepType.LAST, written as next_step_type on the final record of
# every episode by robotaxi.collect_expert_demos.
_LAST = 2
ROWS_PER_FILE = 10000


def corpus_dir(demo_job_id, course_type):
    return os.path.join(TFRECORD_ROOT, "fly_trace_%s_%s" % (course_type, demo_job_id))


def _meta_path(directory):
    # Beside the directory, not in it: read_files_from_directory parses every
    # file inside as a TFRecord.
    return directory.rstrip("/") + ".json"


def read_meta(directory):
    path = _meta_path(directory)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_trace_corpus(directory):
    return read_meta(directory) is not None


def trace_len(directory):
    return int(read_meta(directory)["trace_len"])


def _course_encoder(course_type):
    from environments.courses.fly_donut_course import FlyDonutCourse, FlyDonutFlowCourse
    courses = {c.COURSE_NAME: c for c in (FlyDonutCourse, FlyDonutFlowCourse)}
    if course_type not in courses:
        raise ValueError("no fly encoder for course %r" % course_type)
    return courses[course_type]._make_encoder()


def load_scene_episodes(source_dir):
    """Recorded episodes as a list of (obs (n, 31), action (n, 2)), in order.

    The leading dist_from_traj column is dropped, which is what the fly
    courses' scene_data_array hands the encoder. Episodes end at a record
    marked LAST and at every file end: collect_expert_demos only flushes
    between episodes, so no episode spans two files.
    """
    import tensorflow as tf

    schema = {
        "observation": tf.io.FixedLenFeature((SCENE_WIDTH,), tf.float32),
        "action": tf.io.FixedLenFeature((2,), tf.float32),
        "next_step_type": tf.io.FixedLenFeature((1,), tf.int64, default_value=[1]),
    }
    episodes, marked = [], 0
    for path in sorted(glob.glob(os.path.join(source_dir, "*"))):
        obs, act = [], []
        for rec in tf.data.TFRecordDataset(path):
            p = tf.io.parse_single_example(rec, schema)
            obs.append(p["observation"].numpy()[1:])
            act.append(p["action"].numpy())
            if int(p["next_step_type"].numpy()[0]) == _LAST:
                marked += 1
                episodes.append((np.stack(obs), np.stack(act)))
                obs, act = [], []
        if obs:
            episodes.append((np.stack(obs), np.stack(act)))
    if not marked:
        # Without boundaries every reset would be replayed as the car
        # teleporting, and the brain would carry one episode's state into the
        # next.
        raise RuntimeError(
            "%s has no episode-end markers (recorded before next_step_type was "
            "written); record a fresh DEMO job to build a fly corpus from."
            % source_dir)
    return episodes


def _write_rows(writer_path, traces, actions):
    import tensorflow as tf

    with tf.io.TFRecordWriter(writer_path) as w:
        for trace, action in zip(traces, actions):
            ex = tf.train.Example(features=tf.train.Features(feature={
                "observation": tf.train.Feature(float_list=tf.train.FloatList(value=trace)),
                "action": tf.train.Feature(float_list=tf.train.FloatList(value=action)),
            }))
            w.write(ex.SerializeToString())


def build(demo_job_id, course_type, out_dir):
    from fly_brain.client import FlyBrainClient, SUBSTEPS

    source_dir = os.path.join(TFRECORD_ROOT, "job_%s" % demo_job_id)
    episodes = load_scene_episodes(source_dir)
    n_frames = sum(len(o) for o, _ in episodes)
    print("fly demo corpus: %d episodes, %d frames from %s"
          % (len(episodes), n_frames, source_dir), flush=True)

    client = FlyBrainClient()
    enc, populations = _course_encoder(course_type)
    from fly_brain.encoder import resolve_cells
    cells = resolve_cells(client, populations)
    info = client.info
    width = int(info.trace_len)

    partial = out_dir + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    os.makedirs(partial)

    buf_t, buf_a, n_files, n_rows, done = [], [], 0, 0, 0
    t0 = time.time()
    for k, (obs, act) in enumerate(episodes):
        client.reset(seed=k)
        enc.reset()
        traces = np.empty((len(obs), width), np.float32)
        for i, o in enumerate(obs):
            _, inject = enc.encode(o, cells)
            traces[i], _, _ = client.step(inject, substeps=SUBSTEPS)
        # Record t holds the observation AFTER action t, so the expert's
        # response to that observation is action t+1. Re-pair accordingly;
        # the last frame of each episode has no response and is dropped.
        buf_t.append(traces[:-1])
        buf_a.append(act[1:])
        done += len(obs)
        if sum(len(b) for b in buf_t) >= ROWS_PER_FILE or k == len(episodes) - 1:
            t, a = np.concatenate(buf_t), np.concatenate(buf_a)
            _write_rows(os.path.join(partial, "%04dtrace.tfrecord" % n_files), t, a)
            n_files += 1
            n_rows += len(t)
            buf_t, buf_a = [], []
            el = time.time() - t0
            print("  %d/%d frames, %.1f min elapsed, %.1f min left"
                  % (done, n_frames, el / 60, el / done * (n_frames - done) / 60), flush=True)
    client.close()

    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(partial, out_dir)
    meta = {
        "source_demo_job_id": str(demo_job_id),
        "course_type": course_type,
        "encoder": type(enc).__name__,
        "trace_len": width,
        "dt": float(info.dt),
        "trace_tau": float(info.trace_tau),
        "substeps": SUBSTEPS,
        "episodes": len(episodes),
        "rows": n_rows,
        "pairing": "trace after frame t -> expert action t+1",
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "build_minutes": round((time.time() - t0) / 60, 1),
    }
    with open(_meta_path(out_dir), "w") as f:
        json.dump(meta, f, indent=2)
    print("fly demo corpus: wrote %d rows x %d to %s" % (n_rows, width, out_dir), flush=True)
    return out_dir


def ensure(demo_job_id, course_type):
    """Path of the trace corpus for this DEMO job and course, building it if needed.

    Rebuilt when the brain's trace width no longer matches, since a corpus at
    the wrong width cannot feed the live observation spec.
    """
    out_dir = corpus_dir(demo_job_id, course_type)
    meta = read_meta(out_dir)
    if meta is not None and os.path.isdir(out_dir):
        from fly_brain.client import FlyBrainClient
        client = FlyBrainClient()
        live_width = int(client.info.trace_len)
        client.close()
        if int(meta["trace_len"]) == live_width:
            print("fly demo corpus: reusing %s (%d rows, built %s)"
                  % (out_dir, meta["rows"], meta["built_at"]), flush=True)
            return out_dir
        print("fly demo corpus: %s is %d wide but the brain is %d; rebuilding"
              % (out_dir, meta["trace_len"], live_width), flush=True)
        os.remove(_meta_path(out_dir))
    return build(demo_job_id, course_type, out_dir)


if __name__ == "__main__":
    ensure(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fly_donut_flow")
