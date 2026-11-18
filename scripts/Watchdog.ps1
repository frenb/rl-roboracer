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
      3. Restart the ros-server containers and the Unity clients (Mongo, the
         dashboard and duck training are left running).
      4. Set the job back to NOT_STARTED (resume signal).
      5. Relaunch the trainer in the background (docker compose exec -d).

    Every poll also probes each ROS host port (10000 + actor index). A port
    whose WSL relay has stopped accepting connections triggers the same
    recovery, logged as ros_ports_wedged=<ports>.

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
    [int]$RosBasePort = 10000,
    [int]$PortProbeAttempts = 6,
    [int]$PortProbeTimeoutMs = 1500,
    [int]$PortFailPolls = 3,
    [switch]$DryRun
)

$mutex = New-Object System.Threading.Mutex($false, 'Global\rl-roboracer-watchdog')
if (-not $mutex.WaitOne(0)) {
    Write-Host 'Another Watchdog.ps1 is already running; exiting.'
    exit 0
}

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
    # -T disables TTY so output is clean; returns stdout lines. The password
    # is the mongo container's own, so it follows MONGO_ROOT_PASSWORD in .env.
    if (-not $script:MongoPw) {
        $script:MongoPw = (docker compose exec -T mongo printenv MONGODB_ROOT_PASSWORD | Out-String).Trim()
    }
    docker compose exec -T mongo mongosh --quiet -u root -p $script:MongoPw `
        --authenticationDatabase admin robotaxi --eval $evalJs 2>$null
}

# Car jobs only (robot roboracer, or missing on older jobs). This watchdog
# reads the car trainer's log, which goes quiet while a microduck job holds
# the shared queue; matching a duck job made it pause that job and restart
# the stack out from under the duck worker.
$CarJobFilter = "robot:{`$in:[null,'roboracer']}"

function Get-RunningJobId {
    (Invoke-Mongo "var j=db.jobs.findOne({status:'IN_PROGRESS', $CarJobFilter}); print(j?String(j._id):'')").Trim()
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
    $raw = Invoke-Mongo @"
db.jobs.find({status:'FAILED', job_type:'TRAIN', $CarJobFilter}, {_id:1, watchdog_resume_count:1})
  .forEach(j => print(String(j._id) + '|' + (j.watchdog_resume_count || 0)));
"@
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

# Unity opens a new TCP connection to its ros-server for every message, and
# after hours of that the WSL (mirrored networking) relay for a host port can
# leak connections until its accept queue fills: connects to that one port
# time out while the others are fine, and its actor times out on every step.
# Returns the host ports that fail most of a few quick connection attempts.
function Get-WedgedRosPorts {
    $bad = @()
    for ($i = 0; $i -lt $NumEnvs; $i++) {
        $port = $RosBasePort + $i
        $ok = 0
        for ($k = 0; $k -lt $PortProbeAttempts; $k++) {
            $c = New-Object Net.Sockets.TcpClient
            try {
                $t = $c.ConnectAsync('127.0.0.1', $port)
                if ($t.Wait($PortProbeTimeoutMs) -and $c.Connected) { $ok++ }
            } catch { } finally { $c.Close() }
        }
        if ($ok -lt $PortProbeAttempts - 1) { $bad += "$port"; $script:PortProbeDetail += "$port=$ok/$PortProbeAttempts " }
    }
    return $bad
}

# A real relay wedge lasts for hours, but a single probe round can fail
# for a few seconds while both relays are healthy (seen 2026-11-14, a
# false positive that restarted a fine job), so act only on a streak.
function Get-PersistentlyWedgedRosPorts {
    $script:PortProbeDetail = ''
    $bad = @(Get-WedgedRosPorts)
    if (-not $bad.Count) { $script:PortFailStreak = 0; return @() }
    $script:PortFailStreak++
    Write-WdLog "ROS port probe failed ($script:PortFailStreak/$PortFailPolls): $script:PortProbeDetail"
    if ($script:PortFailStreak -lt $PortFailPolls) { return @() }
    $script:PortFailStreak = 0
    return $bad
}

function Get-RosServerContainers {
    @(docker ps --format '{{.Names}}' 2>$null | Where-Object { $_ -match 'ros-server' })
}

# Restarting a ros-server container drops its host-port relay along with
# every connection the relay was holding, and cuts the trainer's gRPC
# subscriptions to it, so the trainer and the Unity clients restart too.
function Restart-RosAndClients {
    & docker @ComposeArgs exec -T sim-controller bash -c "pkill -9 -f 'robotaxi[.]py'" 2>$null
    Start-Sleep -Seconds 3
    $containers = Get-RosServerContainers
    Write-WdLog "  restarting $($containers -join ', ')"
    if ($containers) { docker restart @containers | Out-Null }
    Start-Sleep -Seconds 8
    & powershell -NoProfile -File (Join-Path $PSScriptRoot 'Stop-Clients.ps1') | Out-Null
    & powershell -NoProfile -File (Join-Path $PSScriptRoot 'Start-Clients.ps1') -N $NumEnvs -StaggerSeconds 15 | Out-Null
}

function Start-Trainer {
    & docker @ComposeArgs exec -d sim-controller bash -c "cd /python_ws/src && python -u robotaxi.py --num-envs $NumEnvs 2>&1 | tee /tmp/trainer.log" | Out-Null
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
    Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId'),status:'IN_PROGRESS'},{`$set:{status:'PAUSE_REQUESTED'}})" | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Seconds 5
        $st = (Invoke-Mongo "print(db.jobs.findOne({_id:ObjectId('$jobId')}).status)").Trim()
    } while ($st -notmatch 'PAUSED|DONE|CANCELLED|FAILED' -and (Get-Date) -lt $deadline)
    # A quiet log at the end of a job (final eval, progress video) reads as a
    # silent hang; on 2026-11-17 the job finished mid-recovery, the trainer
    # had already started the next job, and the reset below re-ran the
    # finished one from scratch.
    if ($st -match 'DONE|CANCELLED|FAILED') {
        Write-WdLog "  job is $st (it ended on its own); skipping recovery."
        $script:LastRestart = Get-Date
        return
    }
    if ($st -match 'PAUSED') { Write-WdLog "  job PAUSED (clean checkpoint)." }
    else { Write-WdLog "  pause not confirmed in 60s (trainer hung?); proceeding - crash-recovery will resume from the last auto-checkpoint." }

    # 2-3. stop the trainer, restart the ros-servers and Unity clients. Not
    # Restart-Stack.ps1: its full compose down also took Mongo, the dashboard
    # and duck training down, and the watchdog never logged past it.
    Restart-RosAndClients

    # 4. resume the job, unless it reached a terminal status meanwhile
    Invoke-Mongo "db.jobs.updateOne({_id:ObjectId('$jobId'),status:{`$in:['IN_PROGRESS','PAUSE_REQUESTED','PAUSED']}},{`$set:{status:'NOT_STARTED'}})" | Out-Null

    # 5. relaunch the trainer in the background
    Start-Trainer
    $still = Get-WedgedRosPorts
    Write-WdLog "  trainer relaunched; job set NOT_STARTED to resume. Ports still failing after recovery: $(if ($still) { $still -join ',' } else { 'none' })"

    $script:RestartTimes += (Get-Date)
    $script:LastRestart = Get-Date
}

# --- main loop ----------------------------------------------------------
$script:RestartTimes = @()
$script:LastRestart = [DateTime]::MinValue
$script:PortFailStreak = 0
$script:PortProbeDetail = ''
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

    $badPorts = @(Get-PersistentlyWedgedRosPorts)
    $jobId = Get-RunningJobId
    if (-not $jobId) {
        $silentCount = 0; $prevTail = ''
        if ($badPorts.Count) {
            Write-WdLog "ROS port(s) $($badPorts -join ',') not accepting connections with no job running$(if ($DryRun) { '; DRYRUN, not restarting' } else { '; restarting ros-servers and clients' })"
            if (-not $DryRun) { Restart-RosAndClients; $script:LastRestart = Get-Date }
        }
        continue
    }
    if ($badPorts.Count) {
        $sig = "ros_ports_wedged=$($badPorts -join ',')"
        if ($DryRun) { Write-WdLog "DRYRUN would recover: job=$jobId $sig" }
        else { Invoke-Recovery $jobId $sig; $silentCount = 0; $prevTail = '' }
        continue
    }

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
