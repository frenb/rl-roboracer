"""Phase 3 assess: subscribe to camera/front over gRPC and print one frame.

Run inside sim-controller (gRPC stubs + Docker DNS to ros-server). Unity
Editor Play must be up so CsiFramePublisher is sending. First frame is
usually cmd_id=0; press Play after this script starts if you already
published that id.

    docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
        sh -c "cd /python_ws/src && python check_front_camera.py"

Optional args: ros-server address (default ros-server:50051), wait seconds
(default 15). Actor 0 is ros-server-0:50051.
"""
import asyncio
import base64
import sys
import time

from api import RobotApi, _front_camera_cmd_id, _empty_front_camera_frame

ADDR = sys.argv[1] if len(sys.argv) > 1 else "ros-server:50051"
WAIT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0


def _summarize(frame):
    img = (frame or {}).get("frame") or {}
    header = img.get("header") or {}
    data = img.get("data") or ""
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        raw = b""
    nonzero = any(raw)
    return {
        "width": img.get("width"),
        "height": img.get("height"),
        "encoding": img.get("encoding"),
        "step": img.get("step"),
        "frame_id": header.get("frame_id"),
        "seq": header.get("seq"),
        "bytes": len(raw),
        "nonzero": nonzero,
    }


async def main():
    api = RobotApi(addr=ADDR)
    await api.Initialize()
    print(f"Subscribed to camera/front on {ADDR}. "
          f"Waiting up to {WAIT_S:.0f}s for a frame (Unity Play / F) ...",
          flush=True)

    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline and api.latest_front_camera_frame is None:
        await asyncio.sleep(0.1)

    live = api.latest_front_camera_frame
    if live is None:
        print("FAIL: no camera/front on gRPC. "
              "Is Unity in Play? Did Phase 2 ros-server accept the topic?",
              flush=True)
        return 1

    info = _summarize(live)
    cmd_id = _front_camera_cmd_id(live)
    print(
        f"first frame {info['width']}x{info['height']} "
        f"encoding={info['encoding']} step={info['step']} "
        f"seq={info['seq']} bytes={info['bytes']} "
        f"frame_id={info['frame_id']} nonzero={info['nonzero']}",
        flush=True)

    cached = await api.GetFrontCameraFrame(cmd_id)
    same = cached is live
    print(f"GetFrontCameraFrame({cmd_id}) cache hit={same}", flush=True)

    ok = (
        info["width"] == 84 and info["height"] == 84
        and info["encoding"] == "rgb8" and info["step"] == 252
        and info["bytes"] == 21168 and info["nonzero"]
        and info["frame_id"] == "camera_visual" and same
    )
    if ok:
        print("OK - gRPC sees 84x84 rgb8 matching Unity cmd_id.", flush=True)
        return 0

    empty = _empty_front_camera_frame(cmd_id or 0)
    if not info["nonzero"]:
        print("FAIL: frame is zeros (timeout fallback or black capture).",
              flush=True)
    else:
        print("FAIL: unexpected size/encoding/frame_id.", flush=True)
    print(f"(zeros fallback would be { _summarize(empty) })", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
