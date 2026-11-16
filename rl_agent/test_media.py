import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import media  # noqa: E402

np = pytest.importorskip("numpy")
imageio = pytest.importorskip("imageio.v2")


def frames(count, shade=0):
    return [np.full((270, 480, 3), (shade + 8 * i) % 255, np.uint8) for i in range(count)]


def test_clip_and_progression(tmp_path):
    first = media.write_clip(frames(30), tmp_path / "clip_10.mp4", 10, "step 10")
    second = media.write_clip(frames(20, 100), tmp_path / "clip_25.mp4", 10)
    progress = media.stitch([first, second], tmp_path / "progress_25.mp4")

    assert media.clips_in(tmp_path) == [media.video_entry(10, first), media.video_entry(25, second)]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "clip_10.gif", "clip_10.mp4", "clip_25.gif", "clip_25.mp4", "progress_25.gif", "progress_25.mp4"]
    reader = imageio.get_reader(progress)
    assert reader.get_meta_data()["fps"] == 10
    assert sum(1 for _ in reader) == 50
    gif = imageio.mimread(progress.with_suffix(".gif"))
    assert gif[0].shape[1] == media.GIF_WIDTH

    cut = media.stitch([first, second], tmp_path / "cut" / "progress_25.mp4", max_seconds=1.5)
    assert sum(1 for _ in imageio.get_reader(cut)) == 30  # 15 frames of each


def test_clips_in_ignores_partial_and_other_files(tmp_path):
    (tmp_path / "clip_5.part.mp4").write_bytes(b"")
    (tmp_path / "progress_5.mp4").write_bytes(b"")
    (tmp_path / "clip_7.mp4").write_bytes(b"")
    assert [v["iteration"] for v in media.clips_in(tmp_path)] == [7]
    assert media.clips_in(tmp_path / "missing") == []


def test_pick_evenly_keeps_first_and_last():
    assert media.pick_evenly(list(range(11)), 3) == [0, 5, 10]
    assert media.pick_evenly([1, 2], 10) == [1, 2]
    assert media.pick_evenly([1, 2, 3], 1) == [3]
