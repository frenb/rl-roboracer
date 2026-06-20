<#
.SYNOPSIS
    Auto-recover the trainer from the recurring gym<->ROS connection wedge.

.DESCRIPTION
    Runs in its own terminal alongside the trainer. Polls the trainer's
    robotaxi.out for the wedge signature (one actor producing a flood of
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
    wedged actor. Default 40. (Healthy is ~0; a real wedge produces hundreds.)

.PARAMETER SlowCollectThreshold
    Number of recent TRAIN-end lines with collect_sec >= 5s that signals a
    stall. Default 3. (Healthy collect_sec is ~0.1-0.3s.)

.PARAMETER SilentHangPolls
    Consecutive polls with byte-identical log tail (no new output at all)
    that signals a silent hang. Default 6 (=> ~3 min at the default poll).

.PARAMETER NumEnvs
    --num-envs for the relaunched trainer. Default 2.

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
    [int]$TimeoutThreshold = 40,
    [int]$SlowCollectThreshold = 3,
    [int]$SilentHangPolls = 6,
    [int]$TailLines = 600,
    [int]$NumEnvs = 2,
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

function Get-Tail {
    (& docker @ComposeArgs exec -T sim-controller bash -c "tail -n $TailLines /python_ws/src/robotaxi.out" 2>$null) -join "`n"
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
    $trainerCmd = "docker compose -f docker-compose.yml -f compose/scale.yml exec sim-controller bash -c 'cd /python_ws/src && python -u robotaxi.py --num-envs $NumEnvs 2>&1 | tee robotaxi.out'"
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

Write-WdLog "Watchdog started (poll=${PollSeconds}s cooldown=${CooldownSeconds}s max/hr=$MaxRestartsPerHour timeoutThresh=$TimeoutThreshold dryRun=$DryRun). Logging to $LogFile"

while ($true) {
    Start-Sleep -Seconds $PollSeconds

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
