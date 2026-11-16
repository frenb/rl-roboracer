"""Training videos and GIFs, the same for every robot.

Each saved checkpoint of a training job gets a short clip, ``clip_<N>.mp4`` plus
a smaller ``clip_<N>.gif`` labelled with its step, and when training finishes a
progression video stitches evenly spaced clips into ``progress_<N>.mp4`` and
``.gif``. The job records them as ``videos`` (``[{iteration, path, gif}]``),
``progress_video`` and ``progress_gif``. Only where the frames come from is
robot-specific: the duck renders its checkpoints in mjlab, the car records its
Unity client's overhead camera during every eval.

imageio, numpy and Pillow are imported where they're used, so code that only
needs the paths and bookkeeping here doesn't need them installed.
"""
import os
import re
from pathlib import Path

GIF_WIDTH = 360
GIF_FRAME_STRIDE = 4
_CLIP_RE = re.compile(r"^clip_(\d+)\.mp4$")


def font(size=28):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size bitmap font
        return ImageFont.load_default()


def label(frame, text, label_font):
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((12, 10), text, font=label_font)
    draw.rectangle((left - 6, top - 4, right + 6, bottom + 4), fill=(0, 0, 0))
    draw.text((12, 10), text, fill=(255, 255, 255), font=label_font)
    return np.asarray(image)


def gif_frame(frame):
    import numpy as np
    from PIL import Image
    scale = GIF_WIDTH / frame.shape[1]
    return np.asarray(Image.fromarray(frame).resize((GIF_WIDTH, int(frame.shape[0] * scale))))


def fit(frame, size):
    """``frame`` scaled to fit ``size`` (width, height), letterboxed in black."""
    import numpy as np
    from PIL import Image
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    image = Image.fromarray(frame)
    scale = min(width / image.width, height / image.height)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return np.asarray(canvas)


class Media:
    """Streams frames into ``<out>.mp4`` and a subsampled ``<out>.gif``.

    Both are written under temporary names and renamed when complete, the MP4
    last, so an existing MP4 always means a finished pair.
    """

    def __init__(self, out_mp4, fps):
        import imageio.v2 as imageio
        self._imageio = imageio
        self.mp4 = Path(out_mp4)
        self.gif = self.mp4.with_suffix(".gif")
        self.fps = fps
        self.mp4.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_mp4 = self.mp4.with_name(self.mp4.stem + ".part.mp4")
        self._writer = self._imageio.get_writer(self._tmp_mp4, fps=fps, macro_block_size=1)
        self._gif_frames = []
        self._count = 0

    def add(self, frame):
        self._writer.append_data(frame)
        if self._count % GIF_FRAME_STRIDE == 0:
            self._gif_frames.append(gif_frame(frame))
        self._count += 1

    def close(self):
        self._writer.close()
        tmp_gif = self.gif.with_name(self.gif.stem + ".part.gif")
        self._imageio.mimsave(tmp_gif, self._gif_frames, duration=1000 * GIF_FRAME_STRIDE / self.fps, loop=0)
        os.replace(tmp_gif, self.gif)
        os.replace(self._tmp_mp4, self.mp4)


def write_clip(frames, out_mp4, fps, text=None):
    """``frames`` (RGB arrays) as ``out_mp4`` and its GIF, each labelled ``text``
    in type sized to the frame width."""
    label_font = None
    media = Media(out_mp4, fps)
    for frame in frames:
        if text and label_font is None:
            label_font = font(max(16, frame.shape[1] // 30))
        media.add(label(frame, text, label_font) if text else frame)
    media.close()
    return media.mp4


def stitch(clips, out_mp4, max_seconds=None):
    """One video of ``clips`` back to back, with its GIF, at the newest clip's
    size (a job's clips can differ, e.g. across a client upgrade); each clip
    cut to its first ``max_seconds`` when given."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(clips[-1])
    meta = reader.get_meta_data()
    reader.close()
    fps, size = meta["fps"], tuple(meta["size"])
    limit = round(max_seconds * fps) if max_seconds else None
    media = Media(out_mp4, fps)
    for clip in clips:
        reader = imageio.get_reader(clip)
        for n, frame in enumerate(reader):
            if limit is not None and n >= limit:
                break
            media.add(fit(frame, size))
        reader.close()
    media.close()
    print(f"Written {media.mp4} and {media.gif} from {len(clips)} clips", flush=True)
    return media.mp4


def pick_evenly(items, count):
    """Up to ``count`` items spread evenly, always keeping the first and last."""
    if count <= 1 or len(items) <= count:
        return list(items) if count > 1 else list(items[-1:])
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def video_entry(iteration, mp4):
    """The ``videos`` entry a job records for one clip."""
    return {"iteration": iteration, "path": str(mp4), "gif": str(Path(mp4).with_suffix(".gif"))}


def clips_in(clip_dir):
    """``videos`` entries for the finished clips in ``clip_dir``, oldest first."""
    clip_dir = Path(clip_dir)
    if not clip_dir.is_dir():
        return []
    found = []
    for path in clip_dir.iterdir():
        match = _CLIP_RE.match(path.name)
        if match:
            found.append(video_entry(int(match.group(1)), path))
    return sorted(found, key=lambda v: v["iteration"])
