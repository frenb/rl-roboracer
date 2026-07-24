#!/usr/bin/env python3
"""Extract frames from a screen recording of the car driving and correlate
each frame with the trainer's timestamped diagnostic log lines
([steer-diag] / [crash-pos] / [demo-curriculum] / num_trajectories), so a
video can be reviewed frame-by-frame alongside the exact speed/steer/crash
telemetry at that instant.

Why this exists (2026-07-19): a video recording is wall-clock time but
trainer.log only records step/trajectory counts - there was no way to line
up "the car looks like it's not reacting at 0:42 in the video" with "here's
what the policy/controller actually saw and did at that moment" without
manually eyeballing both. robotaxi.py's collect_expert_demos() diagnostic
prints now include `t=HH:MM:SS.mmm` (local time, same clock as this
machine) specifically so this script can match them against a recording's
embedded start time.

Requires ffmpeg + ffprobe. If not on PATH, pass --ffmpeg-dir pointing at the
folder containing ffmpeg.exe/ffprobe.exe (e.g. the winget install location).

Note: `t=HH:MM:SS.mmm` in the log is whatever local time the sim-controller
CONTAINER's clock is set to, which is commonly UTC and NOT the same as the
Windows host's local timezone the video's start time is derived from (e.g.
container UTC vs host UTC-7 - confirmed via `docker compose exec
sim-controller date` vs the host clock on 2026-07-19). This script
auto-detects that offset (see get_container_tz_offset_hours()) and shifts
log timestamps into host-local time before matching, so --log-tz-offset-hours
should rarely need to be set manually - it's only there as an override/
escape hatch if the auto-detected container isn't reachable or the offset
needs to be pinned for a re-run against an older, already-copied log file.

Basic usage (copies /tmp/trainer.log out of the sim-controller container
automatically unless --log or --no-docker-log is given):

    python scripts/analyze_video.py --video "C:\\path\\to\\Recording.mp4"

Re-run against an already-extracted log file (e.g. if the container isn't
up right now):

    python scripts/analyze_video.py --video recording.mp4 --log trainer.log

Output: a folder of extracted JPEG frames plus a report.md pairing each
frame with the trainer log lines that fall within --log-window seconds of
that frame's estimated wall-clock time.
"""
import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Diagnostic lines emitted by collect_expert_demos() (rl_agent/robotaxi.py)
# all share this `t=HH:MM:SS.mmm` marker (see the _ts() helper there).
TIMESTAMP_RE = re.compile(r"\bt=(\d{2}):(\d{2}):(\d{2})\.(\d{3})\b")


def find_ffmpeg_tools(ffmpeg_dir):
    """Resolve ffmpeg/ffprobe: explicit --ffmpeg-dir > PATH > winget default."""
    candidates = []
    if ffmpeg_dir:
        candidates.append(Path(ffmpeg_dir))
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(Path(on_path).parent)
    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_root.is_dir():
        for p in winget_root.glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
            candidates.append(p)
    for c in candidates:
        ffmpeg_exe = c / "ffmpeg.exe"
        ffprobe_exe = c / "ffprobe.exe"
        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            return str(ffmpeg_exe), str(ffprobe_exe)
        # Non-Windows / already-on-PATH names without .exe
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return "ffmpeg", "ffprobe"
    sys.exit(
        "Could not locate ffmpeg/ffprobe. Install with "
        "`winget install --id Gyan.FFmpeg -e` or pass --ffmpeg-dir.")


def probe_duration_seconds(ffprobe_exe, video_path):
    out = subprocess.check_output(
        [ffprobe_exe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        text=True)
    return float(out.strip())


def resolve_video_start_time(video_path, duration_s, override):
    """Video END time is taken from the file's last-write-time (the moment
    the recording tool finished flushing the file to disk), which is far
    more reliable across recording tools than trusting the meaning of the
    embedded creation_time tag (some tools stamp recording START, others
    stamp the moment muxing FINISHED - empirically the latter was true for
    the first video this script was built against, where creation_time
    landed within ~1s of the file's mtime rather than duration-seconds
    before it). Start = end - duration. --video-start-time overrides this
    entirely for whenever this guess is wrong.
    """
    if override:
        return dt.datetime.fromisoformat(override)
    mtime = dt.datetime.fromtimestamp(video_path.stat().st_mtime)
    return mtime - dt.timedelta(seconds=duration_s)


def extract_frames(ffmpeg_exe, video_path, out_dir, interval_s):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    fps = 1.0 / interval_s
    cmd = [ffmpeg_exe, "-y", "-i", str(video_path),
           "-vf", f"fps={fps}", "-q:v", "2", pattern]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def get_container_tz_offset_hours(container, compose_files):
    """Hours to ADD to the container's local clock to get the host's local
    time (i.e. host_local = container_local + offset). Measured by comparing
    a `date` call inside the container against host time taken immediately
    around it - NOT via epoch seconds (those are timezone-independent and
    would show ~0 even when the display timezones differ, which is exactly
    the bug this works around). Returns None if the container isn't
    reachable, in which case callers should fall back to assuming no offset
    (with a loud warning, since a silent wrong-by-N-hours match is far worse
    than skipping correlation entirely).
    """
    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd += ["-f", f]
    cmd += ["exec", "-T", container, "date", "+%H:%M:%S"]
    host_before = dt.datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    host_after = dt.datetime.now()
    if result.returncode != 0 or not result.stdout.strip():
        return None
    h, m, s = (int(x) for x in result.stdout.strip().split(":"))
    host_mid = host_before + (host_after - host_before) / 2
    container_today = dt.datetime.combine(host_mid.date(), dt.time(h, m, s))
    offset_hours = round((host_mid - container_today).total_seconds() / 3600.0)
    return offset_hours


def copy_log_from_docker(container, container_log_path, dest_path,
                          compose_files):
    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd += ["-f", f]
    cmd += ["cp", f"{container}:{container_log_path}", str(dest_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"(warning) could not copy log from docker: {result.stderr.strip()}")
        return None
    return dest_path


def parse_log_lines(log_path, video_date):
    """Returns a list of (datetime, raw_line) for every line carrying a
    t=HH:MM:SS.mmm marker, anchored to video_date (the log has no date, only
    time-of-day, so we assume same calendar day as the video; see the
    midnight-wrap handling below for the one edge case that breaks that
    assumption)."""
    entries = []
    if not log_path or not Path(log_path).exists():
        return entries
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = TIMESTAMP_RE.search(line)
            if not m:
                continue
            h, mi, s, ms = (int(x) for x in m.groups())
            ts = dt.datetime.combine(
                video_date, dt.time(h, mi, s, ms * 1000))
            entries.append((ts, line.rstrip("\n")))
    return entries


def nearest_log_lines(entries, frame_time, window_s, max_lines):
    window = dt.timedelta(seconds=window_s)
    matches = [(abs((ts - frame_time).total_seconds()), ts, line)
               for ts, line in entries if abs(ts - frame_time) <= window]
    matches.sort(key=lambda m: m[0])
    # Prefer crash-pos lines (rare + high-signal) then fall back to nearest.
    crashes = [m for m in matches if "[crash-pos]" in m[2]]
    others = [m for m in matches if "[crash-pos]" not in m[2]]
    picked = (crashes + others)[:max_lines]
    picked.sort(key=lambda m: m[1])
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to the recording (.mp4 etc.)")
    ap.add_argument("--out-dir", default=None,
                     help="Default: <video_dir>/<video_stem>_frames")
    ap.add_argument("--interval", type=float, default=2.0,
                     help="Seconds between extracted frames (default 2.0)")
    ap.add_argument("--ffmpeg-dir", default=None,
                     help="Folder containing ffmpeg.exe/ffprobe.exe if not on PATH")
    ap.add_argument("--video-start-time", default=None,
                     help="ISO datetime override, e.g. 2026-07-19T11:07:07 "
                          "(default: auto-derived from file mtime - duration)")
    ap.add_argument("--log", default=None,
                     help="Local path to a trainer.log copy. If omitted, "
                          "auto-copies from the sim-controller container "
                          "unless --no-docker-log is set.")
    ap.add_argument("--no-docker-log", action="store_true",
                     help="Skip auto-copying the log from docker")
    ap.add_argument("--container", default="sim-controller",
                     help="Compose service name to copy the log from (default sim-controller)")
    ap.add_argument("--container-log-path", default="/tmp/trainer.log")
    ap.add_argument("--compose-file", action="append", default=None,
                     help="docker-compose -f file(s); default docker-compose.yml")
    ap.add_argument("--log-window", type=float, default=1.5,
                     help="Seconds of tolerance when matching a frame to log lines (default 1.5)")
    ap.add_argument("--log-tz-offset-hours", type=float, default=None,
                     help="Hours to ADD to log timestamps to align them with "
                          "host-local time. Default: auto-detected from the "
                          "container's clock (see get_container_tz_offset_hours).")
    ap.add_argument("--max-log-lines", type=int, default=6,
                     help="Max correlated log lines shown per frame (default 6)")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        video_path.parent / f"{video_path.stem}_frames")

    ffmpeg_exe, ffprobe_exe = find_ffmpeg_tools(args.ffmpeg_dir)

    duration_s = probe_duration_seconds(ffprobe_exe, video_path)
    video_start = resolve_video_start_time(video_path, duration_s, args.video_start_time)
    video_end = video_start + dt.timedelta(seconds=duration_s)
    print(f"Video: {video_path.name}")
    print(f"Duration: {duration_s:.1f}s")
    print(f"Estimated wall-clock window: {video_start} -> {video_end}")

    compose_files = args.compose_file or ["docker-compose.yml"]
    log_path = args.log
    if not log_path and not args.no_docker_log:
        tmp_log = out_dir.parent / f"{video_path.stem}_trainer.log"
        copied = copy_log_from_docker(
            args.container, args.container_log_path, tmp_log, compose_files)
        if copied:
            log_path = str(copied)
            print(f"Copied log from docker -> {log_path}")

    tz_offset_hours = args.log_tz_offset_hours
    if tz_offset_hours is None and log_path:
        tz_offset_hours = get_container_tz_offset_hours(args.container, compose_files)
        if tz_offset_hours is None:
            print("(warning) could not auto-detect container/host clock "
                  "offset (container unreachable?) - assuming 0. Pass "
                  "--log-tz-offset-hours explicitly if correlation looks off.")
            tz_offset_hours = 0.0
        elif tz_offset_hours != 0:
            print(f"Detected container clock is {tz_offset_hours:+.1f}h "
                  "off host-local time - shifting log timestamps to match.")

    log_entries = parse_log_lines(log_path, video_start.date())
    if tz_offset_hours:
        shift = dt.timedelta(hours=tz_offset_hours)
        # Re-stamp the displayed line's own t=HH:MM:SS.mmm token to the
        # shifted (host-local) time too, not just the datetime used
        # internally for matching - otherwise a frame timestamped
        # "11:33" in host-local time would show paired log lines still
        # printing the raw container-clock "t=18:33", which reads like a
        # bug/mismatch even though the correlation itself is correct.
        def _restamp(ts, line):
            return TIMESTAMP_RE.sub(f"t={ts.strftime('%H:%M:%S.%f')[:-3]}", line, count=1)
        log_entries = [(ts + shift, _restamp(ts + shift, line)) for ts, line in log_entries]
    if log_path and not log_entries:
        print(f"(warning) no `t=HH:MM:SS.mmm` lines found in {log_path} - "
              "this trainer.log may predate the 2026-07-19 timestamp logging "
              "change, or covers a different time window than this video.")
    elif log_entries:
        print(f"Parsed {len(log_entries)} timestamped log lines "
              f"({log_entries[0][0].time()} -> {log_entries[-1][0].time()})")

    print(f"Extracting frames every {args.interval}s to {out_dir} ...")
    frames = extract_frames(ffmpeg_exe, video_path, out_dir, args.interval)
    print(f"Extracted {len(frames)} frames")

    report_path = out_dir / "report.md"
    with open(report_path, "w") as report:
        report.write(f"# Video analysis: {video_path.name}\n\n")
        report.write(f"- Duration: {duration_s:.1f}s\n")
        report.write(f"- Estimated wall-clock window: {video_start} -> {video_end}\n")
        report.write(f"- Log source: {log_path or '(none)'}\n\n")
        for i, frame in enumerate(frames):
            offset_s = i * args.interval
            frame_time = video_start + dt.timedelta(seconds=offset_s)
            report.write(f"## Frame {i+1} - t+{offset_s:.1f}s ({frame_time.time()})\n\n")
            report.write(f"![frame {i+1}]({frame.name})\n\n")
            matched = nearest_log_lines(
                log_entries, frame_time, args.log_window, args.max_log_lines)
            if matched:
                report.write("```\n")
                for _, ts, line in matched:
                    report.write(line + "\n")
                report.write("```\n\n")
            elif log_entries:
                report.write("_(no log lines within window)_\n\n")
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
