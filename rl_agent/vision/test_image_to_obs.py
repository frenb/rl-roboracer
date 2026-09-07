"""Unit tests for vision.image_to_obs. No gym, no ROS, no tensorflow.

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python vision/test_image_to_obs.py"

Exit 0 = all assertions passed.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import traceback

# `python vision/test_image_to_obs.py` puts this file's dir on sys.path,
# not /python_ws/src. Add the trainer root so `vision` imports resolve.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from vision.image_to_obs import image_to_obs, load_camera_calibration


_passed = 0
_failed = 0


def _expect(label, predicate, detail=''):
    global _passed, _failed
    try:
        ok = bool(predicate())
    except Exception:
        ok = False
        detail = (detail + '\n' + traceback.format_exc()).strip()
    if ok:
        _passed += 1
        print('PASS  ' + label, flush=True)
    else:
        _failed += 1
        extra = ('  ' + detail) if detail else ''
        print('FAIL  ' + label + extra, flush=True)


def _rgb_ramp(h, w):
    """Known uint8 RGB: R = x, G = y, B = 128 (clipped to 255)."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.minimum(xs, 255)
    img[:, :, 1] = np.minimum(ys, 255)
    img[:, :, 2] = 128
    return img


def _camera_msg(img_uint8, encoding='rgb8'):
    h, w, c = img_uint8.shape
    return {
        'frame': {
            'header': {'seq': 3, 'stamp': {}, 'frame_id': 'camera_visual'},
            'height': h,
            'width': w,
            'encoding': encoding,
            'is_bigendian': 0,
            'step': w * c,
            'data': base64.b64encode(img_uint8.tobytes()).decode('ascii'),
        }
    }


def main():
    ramp84 = _rgb_ramp(84, 84)

    obs = image_to_obs(ramp84, encoding='rgb8', src_hw=(84, 84))
    _expect('ndarray rgb8 shape', lambda: obs.shape == (84, 84, 3))
    _expect('ndarray rgb8 dtype', lambda: obs.dtype == np.float32)
    _expect('ndarray rgb8 range', lambda: obs.min() >= 0.0 and obs.max() <= 1.0)
    _expect(
        'ndarray rgb8 pixel (0,0)',
        lambda: np.allclose(obs[0, 0], np.array([0, 0, 128], np.float32) / 255.0))
    _expect(
        'ndarray rgb8 pixel (10,20)',
        lambda: np.allclose(obs[10, 20], np.array([20, 10, 128], np.float32) / 255.0))
    _expect(
        'ndarray rgb8 pixel (83,83)',
        lambda: np.allclose(obs[83, 83], np.array([83, 83, 128], np.float32) / 255.0))

    msg = _camera_msg(ramp84, 'rgb8')
    obs_msg = image_to_obs(msg)
    _expect('Camera dict infers encoding/hw', lambda: np.allclose(obs_msg, obs))

    raw = ramp84.tobytes()
    obs_bytes = image_to_obs(raw, encoding='rgb8', src_hw=(84, 84))
    _expect('packed bytes rgb8', lambda: np.allclose(obs_bytes, obs))

    rgba = np.dstack([ramp84, np.full((84, 84), 255, dtype=np.uint8)])
    obs_rgba = image_to_obs(rgba, encoding='rgba8', src_hw=(84, 84))
    _expect('rgba8 drops alpha', lambda: np.allclose(obs_rgba, obs))

    bgr = ramp84[:, :, ::-1].copy()
    obs_bgr = image_to_obs(bgr, encoding='bgr8', src_hw=(84, 84))
    _expect('bgr8 converts to RGB', lambda: np.allclose(obs_bgr, obs))

    k = np.eye(3, dtype=np.float64)
    k[0, 0] = 100.0
    k[1, 1] = 100.0
    k[0, 2] = 41.5
    k[1, 2] = 41.5
    d = np.zeros(5, dtype=np.float64)
    obs_ud = image_to_obs(
        ramp84, encoding='rgb8', src_hw=(84, 84), undistort=True, K=k, D=d)
    _expect(
        'undistort D=0 is near identity',
        lambda: np.mean(np.abs(obs_ud - obs)) < 0.02)

    small = _rgb_ramp(16, 16)
    obs_small = image_to_obs(small, encoding='rgb8', src_hw=(16, 16), dst_hw=(8, 8))
    _expect('resize to dst_hw', lambda: obs_small.shape == (8, 8, 3))
    _expect(
        'resized corner still blue-ish',
        lambda: obs_small[0, 0, 2] > 0.4)

    zeros = np.zeros((84, 84, 3), dtype=np.uint8)
    obs0 = image_to_obs(_camera_msg(zeros))
    _expect('zeros stay zeros', lambda: float(obs0.max()) == 0.0)

    yaml = (
        'image_width: 640\n'
        'camera_matrix:\n'
        '  rows: 3\n'
        '  cols: 3\n'
        '  data: [400.0, 0.0, 320.0, 0.0, 533.0, 240.0, 0.0, 0.0, 1.0]\n'
        'distortion_coefficients:\n'
        '  rows: 1\n'
        '  cols: 5\n'
        '  data: [-0.3, 0.08, 0.0, 0.0, 0.0]\n'
    )
    fd, path = tempfile.mkstemp(suffix='.yaml')
    try:
        os.write(fd, yaml.encode('utf-8'))
        os.close(fd)
        K, D = load_camera_calibration(path)
    finally:
        os.remove(path)
    _expect('calib K shape', lambda: K.shape == (3, 3))
    _expect('calib fx', lambda: abs(K[0, 0] - 400.0) < 1e-6)
    _expect('calib D len', lambda: D.size == 5)
    _expect('calib k1', lambda: abs(D[0] + 0.3) < 1e-6)

    def _bad_undistort():
        try:
            image_to_obs(ramp84, encoding='rgb8', undistort=True)
        except ValueError:
            return True
        return False

    _expect('undistort without K/D raises', _bad_undistort)

    print('', flush=True)
    print('%d passed, %d failed' % (_passed, _failed), flush=True)
    return 0 if _failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
