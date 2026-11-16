"""Training clips for the car: its Unity client's follow camera (ClipRecorder.cs,
top-down on the car, zoomed to the track's width), recorded during evals.

Every eval records the eval env's client, up to ``CLIP_SECONDS``, but only an
eval that sets a new best return (and so saves a new best model) keeps it. The
clip runs whole episodes until it covers at least ``MIN_CLIP_SECONDS``; a
shorter recording is dropped. It is written as
``<saved models>/<robot>/clips/<job id>/clip_<step>.mp4`` and ``.gif``
(rl_agent/media.py, the same files the duck makes), labelled with the step and
return, listed in the job's ``videos``, and stamped on the best model's record.
When training finishes, the first ``PROGRESS_CLIP_SECONDS`` of evenly spaced
clips are stitched into ``progress_<step>.mp4`` and ``.gif`` for the job's
``progress_video`` and ``progress_gif``.

The job turns these off with ``video: false`` (clips) and
``progress_video: false``; ``video_checkpoints`` caps the clips stitched.
Clients built before ClipRecorder send no frames, so their jobs get no clips.
"""
import io
import os
import threading
import time
from pathlib import Path

import media

CLIP_FPS = 10
CLIP_WIDTH = 1280
CLIP_SECONDS = 120
MIN_CLIP_SECONDS = 5.0
# Kept after the clip's last episode ends, so the clip shows how it ended.
EPISODE_TAIL_SECONDS = 1.0
PROGRESS_CLIP_SECONDS = 20
DEFAULT_PROGRESS_CLIPS = 10


def clip_dir(job_id):
    root = Path(os.getenv("SAVED_MODELS_DIR", "/saved_models"))
    return root / os.getenv("ROBOT_TYPE", "robotaxi") / "clips" / str(job_id)


def clip_path(job_id, step):
    return clip_dir(job_id) / f"clip_{step}.mp4"


def _env_call(env, method, *args):
    """``method`` on ``env``, or on its first actor when it's a ParallelPyEnvironment."""
    from tf_agents.environments import parallel_py_environment
    if isinstance(env, parallel_py_environment.ParallelPyEnvironment):
        return env._envs[0].call(method, *args)()
    return getattr(env, method)(*args)


def resample(frames, fps):
    """Timed JPEG frames as RGB arrays at a steady ``fps``, each slot showing the
    newest frame that had arrived by then, so a stall shows as a held frame.
    A generator: a two-minute clip is too big to hold decoded."""
    import numpy as np
    from PIL import Image

    if not frames:
        return
    start = frames[0][0]
    slots = int((frames[-1][0] - start) * fps) + 1
    decoded, i = {}, 0
    size = None
    for k in range(slots):
        due = start + k / fps
        while i + 1 < len(frames) and frames[i + 1][0] <= due:
            i += 1
        if i not in decoded:
            image = Image.open(io.BytesIO(frames[i][1])).convert("RGB")
            size = size or image.size
            if image.size != size:  # the window was resized mid-recording
                image = image.resize(size)
            decoded = {i: np.asarray(image)}
        yield decoded[i]


def whole_episodes(frames, episode_ends):
    """``frames`` up to the end of the first episode that ends at least
    ``MIN_CLIP_SECONDS`` into the recording (plus a short tail); all of them
    when none does."""
    if not frames:
        return frames
    start = frames[0][0]
    end = next((e for e in episode_ends if e - start >= MIN_CLIP_SECONDS), None)
    if end is None:
        return frames
    return [f for f in frames if f[0] <= end + EPISODE_TAIL_SECONDS]


def duration(frames):
    return frames[-1][0] - frames[0][0] if frames else 0.0


class CarClips:
    def __init__(self, env, job_id, db):
        self.env = env
        self.job_id = job_id
        self.db = db
        job = db.jobs.find_one({"_id": job_id}, {"video": 1, "progress_video": 1, "video_checkpoints": 1}) or {}
        self.clips_on = job.get("video") is not False
        self.progress_on = job.get("progress_video") is not False
        self.progress_clips = int(job.get("video_checkpoints") or DEFAULT_PROGRESS_CLIPS)
        self.dir = clip_dir(job_id)
        self._recording = False
        self._episode_ends = []
        self._last = []
        self._writers = []
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self.clips_on or self.progress_on

    def start(self):
        """Record the coming eval."""
        if not self.enabled:
            return
        self._episode_ends = []
        self._last = []
        try:
            _env_call(self.env, "start_clip", CLIP_FPS, CLIP_WIDTH, CLIP_SECONDS)
            self._recording = True
        except Exception as e:  # noqa: BLE001 - a clip must never break training
            print(f"car clip: start failed: {e!r}", flush=True)

    def observe(self, trajectory):
        """Eval-actor observer: notes when each recorded episode ends.
        Frames are stamped with the same clock (time.monotonic) in the env."""
        if self._recording:
            import numpy as np
            if np.any(trajectory.is_last()):
                self._episode_ends.append(time.monotonic())

    def stop(self):
        """End the eval's recording, keeping it until the eval's result is known."""
        if not self._recording:
            return
        self._recording = False
        try:
            frames = _env_call(self.env, "stop_clip") or []
        except Exception as e:  # noqa: BLE001
            print(f"car clip: stop failed: {e!r}", flush=True)
            frames = []
        self._last = whole_episodes(frames, self._episode_ends)
        if not frames:
            print("car clip: no overhead frames from Unity (client without ClipRecorder?)", flush=True)

    def save_best(self, step, avg_return, model_location):
        """The last eval saved a new best model: write its recording as
        ``clip_<step>`` off the training thread and stamp the model with it."""
        frames, self._last = self._last, []
        if not frames:
            return
        if duration(frames) < MIN_CLIP_SECONDS:
            print(f"car clip: step {step} recording is {duration(frames):.1f} s "
                  f"(< {MIN_CLIP_SECONDS:.0f} s); not saved", flush=True)
            return
        thread = threading.Thread(target=self._write, args=(frames, step, avg_return),
                                  name=f"car-clip-{step}", daemon=True)
        thread.start()
        self._writers.append(thread)
        out = clip_path(self.job_id, step)
        try:
            self.db.models.update_many({"location": model_location}, {"$set": {
                "clip": str(out), "clip_gif": str(out.with_suffix(".gif"))}})
        except Exception as e:  # noqa: BLE001
            print(f"car clip: stamping model {model_location} failed: {e!r}", flush=True)

    def _write(self, frames, step, avg_return):
        try:
            out = media.write_clip(resample(frames, CLIP_FPS), clip_path(self.job_id, step), CLIP_FPS,
                                   f"step {step}  return {avg_return:.1f}")
            with self._lock:
                self.db.jobs.update_one({"_id": self.job_id},
                                        {"$set": {"videos": media.clips_in(self.dir)}})
            print(f"car clip: wrote {out} ({len(frames)} frames)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"car clip: writing step {step} failed: {e!r}", flush=True)

    def finish(self):
        """Wait for clips still being written, then stitch the progression video."""
        for thread in self._writers:
            thread.join(timeout=300)
        clips = media.clips_in(self.dir)
        if not (self.progress_on and clips):
            return
        picked = media.pick_evenly(clips, self.progress_clips)
        out = self.dir / f"progress_{picked[-1]['iteration']}.mp4"
        try:
            media.stitch([c["path"] for c in picked], out, max_seconds=PROGRESS_CLIP_SECONDS)
            self.db.jobs.update_one({"_id": self.job_id}, {"$set": {
                "videos": clips, "progress_video": str(out), "progress_gif": str(out.with_suffix(".gif"))}})
        except Exception as e:  # noqa: BLE001
            print(f"car clip: progression video failed: {e!r}", flush=True)
