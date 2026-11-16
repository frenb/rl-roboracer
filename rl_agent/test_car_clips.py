import io
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
np = pytest.importorskip("numpy")
pytest.importorskip("imageio.v2")
from PIL import Image  # noqa: E402

import car_clips  # noqa: E402


def jpeg(shade, size=(480, 270)):
    out = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(out, format="JPEG")
    return out.getvalue()


def test_resample_holds_the_newest_frame_at_a_steady_rate():
    frames = [(10.0, jpeg(0)), (10.05, jpeg(100)), (10.35, jpeg(200))]
    out = list(car_clips.resample(frames, 10))
    assert len(out) == 4  # 10.0, 10.1, 10.2, 10.3: the last frame arrived after 10.3
    assert [round(int(f[0, 0, 0]) / 50) for f in out] == [0, 2, 2, 2]  # 100 from 10.05 on
    frames.append((10.4, jpeg(200, (640, 360))))
    assert list(car_clips.resample(frames, 10))[-1].shape == (270, 480, 3)
    assert list(car_clips.resample([], 10)) == []


def test_whole_episodes_run_to_the_minimum_length_plus_a_tail():
    frames = [(t / 10, b"") for t in range(120)]
    assert car_clips.whole_episodes(frames, []) == frames
    assert car_clips.whole_episodes(frames, [1.0, 3.0]) == frames  # none ends 5 s in
    kept = car_clips.whole_episodes(frames, [1.0, 3.0, 6.0, 9.0])
    assert kept[-1][0] == pytest.approx(6.0 + car_clips.EPISODE_TAIL_SECONDS)


class FakeTrajectory:
    def __init__(self, last):
        self.last = last

    def is_last(self):
        return np.array([self.last])


class FakeEnv:
    def __init__(self, frames):
        self.frames = frames
        self.started = None

    def start_clip(self, fps, width, max_s):
        self.started = (fps, width, max_s)

    def stop_clip(self):
        return self.frames


class FakeJobs:
    def __init__(self, job):
        self.job = job

    def find_one(self, query, projection=None):
        return dict(self.job)

    def update_one(self, query, update):
        self.job.update(update["$set"])


class FakeModels:
    def __init__(self):
        self.stamped = {}

    def update_many(self, query, update):
        self.stamped[query["location"]] = update["$set"]


class FakeDb:
    def __init__(self, job):
        self.jobs = FakeJobs(job)
        self.models = FakeModels()


def recorder(tmp_path, monkeypatch, job, frames):
    monkeypatch.setenv("SAVED_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("ROBOT_TYPE", "robotaxi")
    monkeypatch.setattr(car_clips, "_env_call", lambda env, method, *args: getattr(env, method)(*args))
    db = FakeDb(job)
    return car_clips.CarClips(FakeEnv(frames), "job1", db), db


def test_only_new_best_evals_get_a_clip(tmp_path, monkeypatch):
    frames = [(i / 10, jpeg(2 * i)) for i in range(60)]
    clips, db = recorder(tmp_path, monkeypatch, {"_id": "job1"}, frames)
    for step in (100, 250, 400):
        clips.start()
        clips.stop()
        if step != 250:
            clips.save_best(step, 3.5, f"/saved_models/robotaxi/SacAgent/1_step_{step}")
    clips.finish()

    out = tmp_path / "robotaxi" / "clips" / "job1"
    assert clips.env.started == (car_clips.CLIP_FPS, car_clips.CLIP_WIDTH, car_clips.CLIP_SECONDS)
    assert [v["iteration"] for v in db.jobs.job["videos"]] == [100, 400]
    assert not (out / "clip_250.mp4").exists()
    assert db.jobs.job["progress_video"] == str(out / "progress_400.mp4")
    assert (out / "progress_400.gif").exists() and (out / "clip_100.gif").exists()
    assert db.models.stamped["/saved_models/robotaxi/SacAgent/1_step_400"] == {
        "clip": str(out / "clip_400.mp4"), "clip_gif": str(out / "clip_400.gif")}


def test_short_recordings_are_not_saved(tmp_path, monkeypatch):
    clips, db = recorder(tmp_path, monkeypatch, {"_id": "job1"},
                         [(i / 10, jpeg(20 * i)) for i in range(12)])
    clips.start()
    clips.stop()
    clips.save_best(100, 3.5, "/m")
    clips.finish()
    assert "videos" not in db.jobs.job and not db.models.stamped


def test_clip_runs_whole_episodes_past_the_minimum(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(car_clips.time, "monotonic", lambda: now[0])
    frames = [(i / 10, jpeg(2 * i)) for i in range(120)]
    clips, db = recorder(tmp_path, monkeypatch, {"_id": "job1"}, frames)
    clips.observe(FakeTrajectory(True))  # not recording yet: ignored
    clips.start()
    for t, last in ((1.0, False), (2.0, True), (4.0, True), (7.0, True), (9.0, True)):
        now[0] = t
        clips.observe(FakeTrajectory(last))
    clips.stop()
    assert clips._last[-1][0] == pytest.approx(7.0 + car_clips.EPISODE_TAIL_SECONDS)


def test_job_can_turn_clips_off(tmp_path, monkeypatch):
    clips, db = recorder(tmp_path, monkeypatch, {"_id": "job1", "video": False, "progress_video": False},
                         [(0.0, jpeg(0))])
    clips.start()
    clips.stop()
    clips.save_best(100, 1.0, "/m")
    clips.finish()
    assert clips.env.started is None and "videos" not in db.jobs.job and not db.models.stamped


def test_client_without_recorder_writes_nothing(tmp_path, monkeypatch):
    clips, db = recorder(tmp_path, monkeypatch, {"_id": "job1"}, [])
    clips.start()
    clips.stop()
    clips.save_best(100, 1.0, "/m")
    clips.finish()
    assert "videos" not in db.jobs.job and "progress_video" not in db.jobs.job
