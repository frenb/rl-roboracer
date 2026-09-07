"""CSI frame -> policy observation. Shared by the gym and the Jetson sidecar.

Keep this file copy-pasteable onto the Nano: numpy + cv2 only. Do not import
tensorflow, rospy, or anything under rl_agent. Unity gym path is pinhole
(undistort=False). The car runs cv2.undistort with K/D from
config/camera_calibration/cam_640x480.yaml, then the same resize.

utility.frame_to_tensor assumes RGBA and must not be used for CSI rgb8.
"""
from __future__ import annotations

import base64
import re

import cv2
import numpy as np

DST_HW = (84, 84)
_CALIB_DATA_RE = re.compile(
    r'(?P<key>camera_matrix|distortion_coefficients)\s*:.*?'
    r'data\s*:\s*\[(?P<data>[^\]]+)\]',
    re.DOTALL,
)


def load_camera_calibration(path):
    """Return (K 3x3, D 5,) from an OpenCV/ROS camera yaml."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    mats = {}
    for m in _CALIB_DATA_RE.finditer(text):
        nums = [float(x.strip()) for x in m.group('data').split(',') if x.strip()]
        mats[m.group('key')] = np.asarray(nums, dtype=np.float64)
    if 'camera_matrix' not in mats or 'distortion_coefficients' not in mats:
        raise ValueError('yaml missing camera_matrix or distortion_coefficients: ' + path)
    k = mats['camera_matrix']
    if k.size != 9:
        raise ValueError('camera_matrix must have 9 values, got %d' % k.size)
    return k.reshape(3, 3), mats['distortion_coefficients']


def image_to_obs(msg_or_uint8, encoding=None, src_hw=None, dst_hw=DST_HW,
                 undistort=False, K=None, D=None):
    """float32 [H, W, 3] in [0, 1], RGB.

    msg_or_uint8: niryo_moveit/Camera dict, sensor_msgs/Image dict,
    packed uint8 bytes, or uint8 ndarray [H, W, C].
    encoding: rgb8 (gym), rgba8 (overhead leftover), or bgr8 (some
    gscam feeds). Inferred from a dict message when omitted.
    src_hw: (H, W) of the packed buffer. Inferred from a dict / ndarray.
    undistort: Nano only. Requires K (3x3) and D (plumb_bob, 4 or 5).
    Apply at source resolution, then downsample with INTER_AREA
    (same intent as tf.image.resize method=AREA).
    """
    rgb = _as_rgb_uint8(msg_or_uint8, encoding, src_hw)
    if undistort:
        if K is None or D is None:
            raise ValueError('undistort=True requires K and D')
        k = np.asarray(K, dtype=np.float64).reshape(3, 3)
        d = np.asarray(D, dtype=np.float64).reshape(-1)
        rgb = cv2.undistort(rgb, k, d)
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    if rgb.shape[0] != dh or rgb.shape[1] != dw:
        rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def _as_rgb_uint8(msg_or_uint8, encoding, src_hw):
    if isinstance(msg_or_uint8, np.ndarray):
        arr = np.ascontiguousarray(msg_or_uint8)
        if arr.dtype != np.uint8 or arr.ndim != 3:
            raise TypeError('ndarray must be uint8 [H, W, C]')
        enc = (encoding or 'rgb8').lower()
        h, w = (src_hw if src_hw is not None else arr.shape[:2])
        if arr.shape[0] != h or arr.shape[1] != w:
            raise ValueError('ndarray shape %s != src_hw %s' % (arr.shape, (h, w)))
        return _drop_to_rgb(arr, enc)

    raw, enc, h, w = _decode_message(msg_or_uint8, encoding, src_hw)
    channels = 4 if enc == 'rgba8' else 3
    expected = h * w * channels
    if raw.size != expected:
        raise ValueError(
            'pixel buffer %d bytes, expected %d (%dx%d %s)' % (
                raw.size, expected, w, h, enc))
    return _drop_to_rgb(raw.reshape(h, w, channels), enc)


def _decode_message(msg_or_uint8, encoding, src_hw):
    if isinstance(msg_or_uint8, dict):
        img = msg_or_uint8['frame'] if 'frame' in msg_or_uint8 else msg_or_uint8
        enc = (encoding or img.get('encoding') or 'rgb8').lower()
        if src_hw is not None:
            h, w = int(src_hw[0]), int(src_hw[1])
        else:
            h = int(img['height'])
            w = int(img['width'])
        data = img.get('data', b'')
        if isinstance(data, str):
            buf = base64.b64decode(data)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            buf = bytes(data)
        else:
            raise TypeError('image data must be base64 str or bytes')
        return np.frombuffer(buf, dtype=np.uint8), enc, h, w

    if isinstance(msg_or_uint8, (bytes, bytearray, memoryview)):
        if encoding is None or src_hw is None:
            raise ValueError('bytes input requires encoding and src_hw')
        enc = encoding.lower()
        h, w = int(src_hw[0]), int(src_hw[1])
        return np.frombuffer(bytes(msg_or_uint8), dtype=np.uint8), enc, h, w

    raise TypeError(
        'msg_or_uint8 must be Camera/Image dict, bytes, or uint8 ndarray')


def _drop_to_rgb(hwc, encoding):
    if encoding == 'rgb8':
        if hwc.shape[2] != 3:
            raise ValueError('rgb8 needs 3 channels, got %d' % hwc.shape[2])
        return np.ascontiguousarray(hwc)
    if encoding == 'rgba8':
        if hwc.shape[2] != 4:
            raise ValueError('rgba8 needs 4 channels, got %d' % hwc.shape[2])
        return np.ascontiguousarray(hwc[:, :, :3])
    if encoding == 'bgr8':
        if hwc.shape[2] != 3:
            raise ValueError('bgr8 needs 3 channels, got %d' % hwc.shape[2])
        return cv2.cvtColor(hwc, cv2.COLOR_BGR2RGB)
    raise ValueError('unsupported encoding %r (want rgb8, rgba8, or bgr8)' % encoding)
