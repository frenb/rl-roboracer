<#
.SYNOPSIS
    Monitor a TRAIN/DEMO/EVAL job's progress and crash rate in real time.

.DESCRIPTION
    Combines two checks that were previously done as one-off temp scripts
    during interactive debugging sessions:

      1. Job status/progress from MongoDB (status, percent_complete, gym,
         timestamps, notes).
      2. Live trainer telemetry from the sim-controller container's
         /tmp/trainer.log - total crash count, current trajectory/step
         count, the most recent [crash-pos] crash-location log lines, and
         the most recent [steer-diag] steering/speed diagnostic lines.

    Requires the docker compose stack to be up (mongo + sim-controller
    containers running) and a trainer process writing to
    sim-controller:/tmp/trainer.log (see README.md for the trainer launch
    command). Safe to run against an idle/finished job - sections just show
    whatever is available (e.g. "NO_LOG_FILE" if the trainer hasn't been
    (re)started since the container came up).

.PARAMETER JobId
    Mongo ObjectId (as a hex string) of the job to check. If omitted,
    auto-detects the most recently started job with status IN_PROGRESS.

.PARAMETER Watch
    If set, clears the screen and refreshes every -IntervalSeconds instead
    of printing once and exiting. Stop with Ctrl+C.

.PARAMETER IntervalSeconds
    Refresh interval in -Watch mode, in seconds. Default 15.

.PARAMETER CrashSamples
    How many of the most recent [crash-pos] log lines to print. Default 5.

.EXAMPLE
    .\scripts\Monitor-Job.ps1
    One-shot status check of whichever job is currently IN_PROGRESS.

.EXAMPLE
    .\scripts\Monitor-Job.ps1 -JobId 6a5d0f67a1863ba177b7c34d -Watch
    Continuously monitor a specific job every 15s until Ctrl+C.

.EXAMPLE
    .\scripts\Monitor-Job.ps1 -Watch -IntervalSeconds 30 -CrashSamples 10
    Watch the current IN_PROGRESS job every 30s, showing the last 10 crashes.
#>
param(
    [string]$JobId,
    [switch]$Watch,
    [int]$IntervalSeconds = 15,
    [int]$CrashSamples = 5
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Get-JobStatusJson {
    param([string]$Id)

    if ($Id) {
        $js = "printjson(db.jobs.findOne({_id: ObjectId('$Id')}, " +
              "{status:1, job_type:1, gym_name:1, percent_complete:1, " +
              "started_at:1, ended_at:1, notes:1}));"
    } else {
        $js = "var r = db.jobs.find({status:'IN_PROGRESS'}, " +
              "{status:1, job_type:1, gym_name:1, percent_complete:1, " +
              "started_at:1, notes:1}).sort({started_at:-1}).limit(1).toArray(); " +
              "printjson(r.length ? r[0] : 'No IN_PROGRESS job found - pass -JobId to check a specific job.');"
    }

    $tmpFile = Join-Path $RepoRoot "scripts\_monitor_query_tmp.js"
    [System.IO.File]::WriteAllText($tmpFile, $js, [System.Text.UTF8Encoding]::new($false))
    docker compose cp $tmpFile mongo:/tmp/_monitor_query_tmp.js *> $null
    docker compose exec -T mongo mongosh -u root -p example --authenticationDatabase admin robotaxi --quiet /tmp/_monitor_query_tmp.js
    Remove-Item $tmpFile -ErrorAction SilentlyContinue
}

# Single-quoted here-string: no PowerShell interpolation, so every bash `$VAR`
# below is passed through to bash literally. __SAMPLES__ is the only value
# substituted from the PowerShell side (a plain integer, so no quoting risk).
$BashLogStatsTemplate = @'
#!/bin/bash
LOG=/tmp/trainer.log
if [ ! -f "$LOG" ]; then
    echo "NO_LOG_FILE ($LOG does not exist - has the trainer been started in this container?)"
    exit 0
fi
# Read crashes/trajectory count from the trainer's own most recent
# "num_trajectories: X vs Y, crashes: N" line rather than a raw grep -c over
# the whole file - collect_expert_demos() resets both counters to 0 at the
# start of EVERY job, but /tmp/trainer.log is NOT truncated between jobs
# picked up by the same trainer process, so a whole-file grep -c would mix
# crash counts from earlier, already-finished jobs into the current job's
# rate and make it meaningless.
LAST_STATUS_LINE=$(grep -oP 'num_trajectories: \d+ vs [0-9.]+, crashes: \d+' "$LOG" | tail -1)
CURRENT_TRAJ=$(echo "$LAST_STATUS_LINE" | grep -oP 'num_trajectories: \K[0-9]+')
CURRENT_CRASHES=$(echo "$LAST_STATUS_LINE" | grep -oP 'crashes: \K[0-9]+')
echo "current_job_trajectories=${CURRENT_TRAJ:-unknown}"
echo "current_job_crashes=${CURRENT_CRASHES:-unknown}"
if [ -n "$CURRENT_TRAJ" ] && [ -n "$CURRENT_CRASHES" ] && [ "$CURRENT_TRAJ" -gt 0 ]; then
    RATE=$(awk -v c="$CURRENT_CRASHES" -v t="$CURRENT_TRAJ" 'BEGIN{printf "%.3f", (c/t)*1000}')
    echo "current_job_crashes_per_1k_steps=$RATE"
fi
echo "(note: crash-pos lines below may include earlier jobs too, if this trainer process has picked up more than one job since it last (re)started - only current_job_crashes above is scoped to just the current job)"
echo '--- recent [crash-pos] lines (log-wide, may span multiple jobs) ---'
grep 'crash-pos' "$LOG" 2>/dev/null | tail -__SAMPLES__ || echo '(none yet)'
echo '--- recent [steer-diag] lines ---'
grep 'steer-diag' "$LOG" 2>/dev/null | tail -3 || echo '(none yet)'
echo '--- log tail ---'
tail -5 "$LOG"
'@

function Get-TrainerLogStats {
    param([int]$Samples)

    # Force LF-only line endings - Set-Content on Windows defaults to CRLF,
    # which bash chokes on ("syntax error: unexpected end of file") since a
    # trailing \r gets embedded in tokens (e.g. inside a heredoc-free script
    # this can land mid-command). Writing bytes directly avoids that.
    $bashScript = $BashLogStatsTemplate.Replace('__SAMPLES__', [string]$Samples).Replace("`r`n", "`n")
    $tmpFile = Join-Path $RepoRoot "scripts\_monitor_logstats_tmp.sh"
    [System.IO.File]::WriteAllText($tmpFile, $bashScript, [System.Text.UTF8Encoding]::new($false))
    docker compose cp $tmpFile sim-controller:/tmp/_monitor_logstats_tmp.sh *> $null
    docker compose exec -T sim-controller bash /tmp/_monitor_logstats_tmp.sh
    Remove-Item $tmpFile -ErrorAction SilentlyContinue
}

function Show-Snapshot {
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ("Job monitor - {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan

    Write-Host "`n--- Job status (MongoDB) ---" -ForegroundColor Yellow
    Get-JobStatusJson -Id $JobId

    Write-Host "`n--- Trainer log stats (sim-controller:/tmp/trainer.log) ---" -ForegroundColor Yellow
    Get-TrainerLogStats -Samples $CrashSamples
}

Push-Location $RepoRoot
try {
    if ($Watch) {
        Write-Host "Watching every $IntervalSeconds s. Press Ctrl+C to stop." -ForegroundColor Green
        while ($true) {
            Clear-Host
            Show-Snapshot
            Start-Sleep -Seconds $IntervalSeconds
        }
    } else {
        Show-Snapshot
    }
} finally {
    Pop-Location
}
