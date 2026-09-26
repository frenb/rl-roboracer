#!/usr/bin/env bash
#
# Launch the trainer with its stdout preserved across restarts.
#
# The documented launch used to be
#
#   python -u robotaxi.py --num-envs 4 2>&1 | tee /tmp/trainer.log
#
# and `tee` opens with O_TRUNC, so every relaunch destroyed the previous
# session's log - which is precisely the thing you want to read after a
# crash or a wedge. This rotates instead: the outgoing log moves into
# /tmp/trainer-logs/ and the trainer starts on a fresh file.
#
# /tmp/trainer.log stays the path the dashboard tails - server.ts spawns
# `tail -F` on it - and -F follows a rotate-and-recreate by name, so the
# live log panel keeps streaming across a restart.
#
# Usage, from the repo root on the host:
#
#   docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller \
#     bash /python_ws/src/run_trainer.sh --num-envs 4
#
# Arguments are forwarded to robotaxi.py untouched.

set -uo pipefail

LOG="${TRAINER_LOG:-/tmp/trainer.log}"
ARCHIVE="${TRAINER_LOG_ARCHIVE:-/tmp/trainer-logs}"
KEEP="${TRAINER_LOG_KEEP:-10}"
SRC_DIR="${TRAINER_SRC_DIR:-/python_ws/src}"

mkdir -p "$ARCHIVE"

# Self-heal the symlink. /tmp/trainer.log spent two months pointing at
# /tmp/trainer_resume6.log, left behind by a debugging session, so
# `ls -la /tmp/trainer.log` reported the LINK's date and the log looked
# abandoned while it was being actively written through. Nothing in the
# repo ever created it. Collapse it back to a real file once and the
# confusion cannot come back.
if [ -L "$LOG" ]; then
  target="$(readlink -f "$LOG" || true)"
  echo "run_trainer: $LOG was a symlink to ${target:-?}; collapsing to a real file."
  rm -f "$LOG"
  if [ -n "$target" ] && [ -s "$target" ]; then
    stamp="$(date -r "$target" +%Y%m%d-%H%M%S 2>/dev/null || date +%Y%m%d-%H%M%S)"
    mv "$target" "$ARCHIVE/trainer-$stamp.log" \
      && echo "run_trainer: kept its contents as $ARCHIVE/trainer-$stamp.log"
  fi
fi

# Rotate the outgoing session out of the way. Named for the log's own
# mtime rather than now, so the filename says when that session ended.
if [ -s "$LOG" ]; then
  stamp="$(date -r "$LOG" +%Y%m%d-%H%M%S 2>/dev/null || date +%Y%m%d-%H%M%S)"
  dest="$ARCHIVE/trainer-$stamp.log"
  # Two restarts within the same second would otherwise collide.
  [ -e "$dest" ] && dest="$ARCHIVE/trainer-$stamp-$$.log"
  mv "$LOG" "$dest" && echo "run_trainer: previous session -> $dest"
fi

# Keep the newest KEEP archives. Unbounded retention is how /tmp
# accumulated 35 MB of one-off trainer logs the first time around.
if [ "$KEEP" -gt 0 ] 2>/dev/null; then
  ls -1t "$ARCHIVE"/trainer-*.log 2>/dev/null | tail -n +$((KEEP + 1)) \
    | while IFS= read -r old; do
        rm -f "$old" && echo "run_trainer: pruned $old"
      done
fi

# Code provenance for the banner. Parsed out of /git_meta by hand
# because the sim-controller image has no git binary - the same reason
# robotaxi.py's _read_git_provenance reads HEAD directly.
git_desc="unknown"
if [ -r /git_meta/HEAD ]; then
  head="$(cat /git_meta/HEAD)"
  case "$head" in
    "ref: "*)
      ref="${head#ref: }"
      branch="${ref##*/}"
      sha=""
      [ -r "/git_meta/$ref" ] && sha="$(cat "/git_meta/$ref")"
      # A freshly-cloned repo keeps branch refs in packed-refs rather
      # than as loose files.
      if [ -z "$sha" ] && [ -r /git_meta/packed-refs ]; then
        sha="$(awk -v r="$ref" '$2 == r {print $1; exit}' /git_meta/packed-refs)"
      fi
      if [ -n "$sha" ]; then git_desc="$branch @ ${sha:0:12}"; else git_desc="$branch"; fi
      ;;
    *)
      git_desc="detached @ ${head:0:12}"
      ;;
  esac
fi

# Session banner. A dashboard client connecting later gets `tail -n 100`,
# so whoever opens the panel mid-run can still see which code and which
# arguments produced the lines under it.
{
  echo "=== trainer session $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
  echo "=== code: $git_desc"
  echo "=== args: robotaxi.py $*"
} > "$LOG"

cd "$SRC_DIR" || exit 1

# -u keeps stdout unbuffered through the pipe. Python switches from line
# to ~8 KB block buffering the moment it sees a pipe, which would hold
# lines back from both the log file and the dashboard's live panel.
python -u robotaxi.py "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
