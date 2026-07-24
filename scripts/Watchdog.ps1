<#
.SYNOPSIS
    Auto-recover the trainer from the recurring gym<->ROS connection wedge.

.DESCRIPTION
    Runs in its own terminal alongside the trainer. Polls the trainer's
    /tmp/trainer.log for the wedge signature (one actor producing a flood of
    "timed out" messages and/or collect_sec stuck high, or the log going
    silent), and on detection performs the SAME graceful recovery we do by
    hand:

      1. PAUSE_REQUESTED the running job -> wait (<=60s) for a clean PAUSED
         checkpoint (falls through if the trainer is fully hung; the Learner
         auto-checkpoints every 100 steps so crash-recovery still resumes).
      2. Force-kill the trainer process in the container.
      3. .\scripts\Restart-Stack.ps1  (recreate containers + fresh Unity gyms).
      4. Set the job back to NOT_STARTED (resume signal).
      5. Relaunch the trainer (viz-off) in a new window.

    Every detection appends a signature line to scripts/watchdog.log so the
    accumulated data can later drive a real root-cause fix.

    Guardrails: a cooldown between restarts and a per-hour circuit breaker
    that STOPS the watchdog (rather than loop forever) if the wedge recurs
    too often - that means something deeper than a transient gym is wrong.

    SEPARATELY (added 2026-07-22, after a fork+CUDA worker-subprocess
    segfault killed a 70%-complete TRAIN job and left it sitting in FAILED
    with nobody noticing for hours), every poll also scans for TRAIN jobs
    with status=FAILED that still have an on-disk Learner checkpoint under
    /tmp/active/<job_id>/learner/train/checkpoints/. If found, the watchdog
    auto-resumes that job by:

      1. Finding the LATEST checkpoint's step number from the checkpoint
         filenames (ckpt-<step>.index).
      2. Setting the job doc's paused_at_step to that step (CRITICAL: this
         is the exact field _restore_paused_active_dir() in robotaxi.py
         checks to decide "resume, don't archive /tmp/active away" - a
         manual recovery done the same day this feature was added
         forgot to set it and the trainer archived a 177K-step checkpoint
         out from under itself, treating the job as fresh. Do not repeat
         that mistake here) and status back to NOT_STARTED.
      3. Relaunching the trainer only if it isn't already running (a
         worker-subprocess segfault or an uncaught do_job exception both
         leave the main trainer process alive and polling - most FAILED
         jobs need only the Mongo update, not a process restart).

    This path is capped per-job by MaxAutoResumesPerJob (tracked via the
    job doc's watchdog_resume_count field, which persists across watchdog
    restarts) so a job that's failing for a REAL, non-transient reason
    (e.g. a genuine NaN/inf loss divergence) doesn't get auto-resumed
    forever - once the cap is hit it's left FAILED for a human to
    investigate, with an ALERT line in the log.

.PARAMETER PollSeconds
    Seconds between health checks. Default 30.

.PARAMETER CooldownSeconds
    Minimum seconds between recoveries (also covers fresh-trainer startup so
    we don't trigger on a still-warming-up trainer). Default 600.

.PARAMETER MaxRestartsPerHour
    Circuit breaker. If exceeded, the watchdog logs a loud ALERT and exits.
    Default 4.

.PARAMETER TimeoutThreshold
    Per-actor "timed out" line count (over the last TailLines) that signals a
    wedged actor. Default 20. Lowered from 40 because symmetric two-actor wedges
    produce ~15-20 per actor (not 40+) in 600 lines. Healthy is ~0.

.PARAMETER SlowCollectThreshold
    Number of recent TRAIN-end lines with collect_sec >= 5s that signals a
    stall. Default 2. Lowered from 3 as belt-and-suspenders for wedges where
    per-actor timeouts are spread symmetrically and each stays below the
    TimeoutThreshold. Healthy collect_sec is ~0.1-0.3s.

.PARAMETER SilentHangPolls
    Consecutive polls with byte-identical log tail (no new output at all)
    that signals a silent hang. Default 6 (=> ~3 min at the default poll).

.PARAMETER NumEnvs
    --num-envs for the relaunched trainer. Default 2.

.PARAMETER MaxAutoResumesPerJob
    Per-job cap on FAILED-with-checkpoint auto-resumes (see DESCRIPTION).
    Default 2 - i.e. a job gets one automatic second chance; if it fails a
    second time it's left FAILED for manual investigation rather than
    looping forever on a job that's failing for a real reason.

.PARAMETER DryRun
    Detect and log only; do NOT actually recover. Useful for tuning
    thresholds against a live trainer before arming it.

.EXAMPLE
    .\scripts\Watchdog.ps1
.EXAMPLE
    .\scripts\Watchdog.ps1 -DryRun        # observe detections without acting
#>
[CmdletBinding()]
param(
    [int]$PollSeconds = 30,
    [int]$CooldownSeconds = 600,
    [int]$MaxRestartsPerHour = 4,
    [int]$TimeoutThreshold = 20,
    [int]$SlowCollectThreshold = 2,
    [int]$SilentHangPolls = 6,
    [int]$TailLines = 600,
    [int]$NumEnvs = 2,
    [int]$MaxAutoResumesPerJob = 2,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$ComposeArgs = @('compose', '-f', 'docker-compose.yml', '-f', 'compose/scale.yml')
$LogFile = Join-Path $PSScriptRoot 'watchdog.log'

function Write-WdLog([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Invoke-Mongo([string]$evalJs) {
    # -T disables TTY so output is clean; returns stdout lines.
    docker compose exec -T mongo mongosh --quiet -u root -p example `
        --authenticationDatabase admin robotaxi --eval $evalJs 2>$null
}

function Get-RunningJobId {
    (Invoke-Mongo "var j=db.jobs.findOne({status:'IN_PROGRESS'}); print(j?String(j._id):'')").Trim()
}

function Get-LatestCheckpointStep([string]$jobId) {
    # Prints the highest ckpt-<N>.index step number under this job's
    # /tmp/active/ Learner checkpoint dir, or nothing if none exists (job
    # never got far enough to checkpoint, or its /tmp/active was already
    # archived away - either way, not safely auto-resumable).
    $out = (& docker @ComposeArgs exec -T sim-controller bash -c `
        "ls /tmp/active/$jobId/learner/train/checkpoints/*.index 2>/dev/null | sed -E 's/.*ckpt-([0-9]+)\.index/\1/' | sort -n | tail -1") 2>$null
    $out = ($out | Out-String).Trim()
    if ($out -match '^\d+$') { return [int]$out }
    return $null
}

function Test-TrainerProcessAlive {
    $out = (& docker @ComposeArgs exec -T sim-controller bash -c `
        "pgrep -f 'python -u robotaxi.py' >/dev/null 2>&1 && echo yes || echo no") 2>$null
    return (($out | Out-String).Trim() -eq 'yes')
}

function Invoke-FailedJobAutoResume {
    # Separate from the wedge path above: this handles TRAIN jobs that
    # already reached a terminal FAILED status (e.g. the do_job uncaught-
    # exception path from a worker-subprocess segfault) rather than a
    # currently-IN_PROGRESS wedge. See DESCRIPTION for the full rationale
    # and the paused_at_step gotcha this must get right.
    $raw = Invoke-Mongo @'
db.jobs.find({status:'FAILED', job_type:'TRAIN'}, {_id:1, watchdog_resume_count:1})
  .forEach(j => print(String(j._id) + '|' + (j.watchdog_resume_count || 0)));
'@
    foreach ($line in ($raw -split "`n")) {
        $line = $line.Trim()
        if (-not $line -or $line -notmatch '\|') { continue }
        $parts = $line -split '\|'
        $jobId = $parts[0].Trim()
        $resumeCount = 0
        [int]::TryParse($parts[1].Trim(), [ref]$resumeCount) | Out-Null

        if ($resumeCount -ge $MaxAutoResumesPerJob) { continue }  # already alerted; leave for a human

        $step = Get-LatestCheckpointStep $jobId
        if ($null -eq $step) {
            Write-WdLog "FAILED job $jobId has no on-disk checkpoint; not auto-resumable. Marking so we stop re-checking it."
            Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId')},{`$set:{watchdog_resume_count:$MaxAutoResumesPerJob}})" | Out-Null
            continue
        }

        $newCount = $resumeCount + 1
        if ($DryRun) {
            Write-WdLog "DRYRUN would auto-resume FAILED job $jobId from checkpoint step=$step (attempt $newCount/$MaxAutoResumesPerJob)"
            continue
        }

        Write-WdLog "AUTO-RESUME: FAILED job $jobId has checkpoint at step=$step (attempt $newCount/$MaxAutoResumesPerJob)"
        # paused_at_step MUST be set together with status in the SAME update -
        # see the DESCRIPTION note on why a status-only update archives the
        # checkpoint away instead of resuming from it.
        Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId')},{`$set:{status:'NOT_STARTED',paused_at_step:$step,watchdog_resume_count:$newCount},`$unset:{eval_error:''}})" | Out-Null

        if (Test-TrainerProcessAlive) {
            Write-WdLog "  trainer process already running; it will pick up the resumed job on its next poll."
        } else {
            Write-WdLog "  trainer process not running; relaunching."
            $trainerCmd = "docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c 'cd /python_ws/src && python -u robotaxi.py --num-envs $NumEnvs 2>&1 | tee /tmp/trainer.log'"
            Start-Process powershell -WorkingDirectory $RepoRoot -ArgumentList '-NoExit', '-Command', $trainerCmd
        }

        if ($newCount -ge $MaxAutoResumesPerJob) {
            Write-WdLog "NOTE: job $jobId has now used its auto-resume budget ($newCount/$MaxAutoResumesPerJob). If it fails again it will NOT be auto-resumed - investigate manually."
        }
    }
}

function Get-Tail {
    # Reads the SAME canonical log the trainer now writes via
    # `| tee /tmp/trainer.log` (see the relaunch commands above and
    # Start-Stack.ps1). Previously /python_ws/src/robotaxi.out, which
    # drifted out of sync with the dashboard/Monitor-Job path and left
    # wedge detection scanning a stale file after a manual restart.
    (& docker @ComposeArgs exec -T sim-controller bash -c "tail -n $TailLines /tmp/trainer.log" 2>$null) -join "`n"
}

function Test-Wedged([string]$tail) {
    $a0 = ([regex]::Matches($tail, '\[actor-0\][^\n]*timed out')).Count
    $a1 = ([regex]::Matches($tail, '\[actor-1\][^\n]*timed out')).Count
    # collect_sec >= 5s : single digit 5-9, or two+ digits, before the decimal
    $slow = ([regex]::Matches($tail, 'collect_sec=(?:[5-9]|\d\d+)\.')).Count
    $wedged = ($a0 -ge $TimeoutThreshold) -or ($a1 -ge $TimeoutThreshold) -or ($slow -ge $SlowCollectThreshold)
    return [pscustomobject]@{ Wedged = $wedged; A0 = $a0; A1 = $a1; Slow = $slow }
}

function Invoke-Recovery([string]$jobId, [string]$sig) {
    # --- circuit breaker ---
    $cutoff = (Get-Date).AddHours(-1)
    $script:RestartTimes = @($script:RestartTimes | Where-Object { $_ -gt $cutoff })
    if ($script:RestartTimes.Count -ge $MaxRestartsPerHour) {
        Write-WdLog "ALERT: $MaxRestartsPerHour recoveries in the last hour (sig=$sig). Something deeper than a transient gym wedge is wrong. STOPPING watchdog."
        exit 2
    }

    Write-WdLog "WEDGE detected: job=$jobId $sig -> starting recovery"

    # 1. graceful pause (best-effort checkpoint)
    Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId')},{`$set:{status:'PAUSE_REQUESTED'}})" | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Seconds 5
        $st = (Invoke-Mongo "print(db.jobs.findOne({_id:ObjectId('$jobId')}).status)").Trim()
    } while ($st -notmatch 'PAUSED' -and (Get-Date) -lt $deadline)
    if ($st -match 'PAUSED') { Write-WdLog "  job PAUSED (clean checkpoint)." }
    else { Write-WdLog "  pause not confirmed in 60s (trainer hung?); proceeding - crash-recovery will resume from the last auto-checkpoint." }

    # 2. force-kill the trainer in the container
    & docker @ComposeArgs exec -T sim-controller bash -c "pkill -9 -f 'robotaxi[.]py'" 2>$null
    Start-Sleep -Seconds 3

    # 3. full stack restart (recreate containers + fresh Unity gyms)
    Write-WdLog "  running Restart-Stack.ps1 ..."
    & (Join-Path $PSScriptRoot 'Restart-Stack.ps1') -N $NumEnvs

    # 4. resume the job
    Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId')},{`$set:{status:'NOT_STARTED'}})" | Out-Null

    # 5. relaunch the trainer (viz-off) in a new window
    $trainerCmd = "docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c 'cd /python_ws/src && python -u robotaxi.py --num-envs $NumEnvs 2>&1 | tee /tmp/trainer.log'"
    Start-Process powershell -WorkingDirectory $RepoRoot -ArgumentList '-NoExit', '-Command', $trainerCmd
    Write-WdLog "  trainer relaunched; job set NOT_STARTED to resume."

    $script:RestartTimes += (Get-Date)
    $script:LastRestart = Get-Date
}

# --- main loop ----------------------------------------------------------
$script:RestartTimes = @()
$script:LastRestart = [DateTime]::MinValue
$prevTail = ''
$silentCount = 0

Write-WdLog "Watchdog started (poll=${PollSeconds}s cooldown=${CooldownSeconds}s max/hr=$MaxRestartsPerHour timeoutThresh=$TimeoutThreshold maxAutoResumes=$MaxAutoResumesPerJob dryRun=$DryRun). Logging to $LogFile"

while ($true) {
    Start-Sleep -Seconds $PollSeconds

    # FAILED-with-checkpoint auto-resume runs every poll regardless of the
    # wedge cooldown below - it's a much lighter-weight action (a Mongo
    # update, not a full stack restart) and a FAILED job sitting idle for
    # the remainder of a CooldownSeconds window is exactly the "nobody
    # noticed for hours" failure mode this was added to close.
    Invoke-FailedJobAutoResume

    if (((Get-Date) - $script:LastRestart).TotalSeconds -lt $CooldownSeconds) { continue }

    $jobId = Get-RunningJobId
    if (-not $jobId) { $silentCount = 0; $prevTail = ''; continue }   # nothing training; idle

    $tail = Get-Tail
    if (-not $tail) { continue }

    # --- silent-hang detection (log content frozen across polls) ---
    if ($tail -eq $prevTail) { $silentCount++ } else { $silentCount = 0 }
    $prevTail = $tail
    $silentHang = ($silentCount -ge $SilentHangPolls)

    # --- wedge detection ---
    $r = Test-Wedged $tail

    if ($r.Wedged -or $silentHang) {
        $sig = "a0_timeouts=$($r.A0) a1_timeouts=$($r.A1) slow_collect=$($r.Slow) silent_hang=$silentHang"
        if ($DryRun) {
            Write-WdLog "DRYRUN would recover: job=$jobId $sig"
        } else {
            Invoke-Recovery $jobId $sig
            $silentCount = 0; $prevTail = ''
        }
    }
}
