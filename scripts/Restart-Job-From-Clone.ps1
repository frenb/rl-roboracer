<#
.SYNOPSIS
    Restart training from a fresh clone of an existing job: kill the running
    trainer, clone the job as NOT_STARTED (optionally attaching a reward design
    and retiring the source), then relaunch the trainer so it picks up the
    clone.

.DESCRIPTION
    Higher-level wrapper around scripts/Clone-Job.ps1. Automates the manual
    "restart a stuck job from a clone" dance:

        1. pkill the running `robotaxi.py` trainer in the sim-controller
           container (idempotent; safe if nothing is running).
        2. Clone-Job.ps1 -SourceJobId ... [-RewardDesignId ...] [-SetSourceDone]
           inserts a clean NOT_STARTED clone (step 0, no checkpoint resume).
        3. Relaunch `python -u robotaxi.py --num-envs N | tee /tmp/trainer.log`
           (detached) with the scale overlay, so the trainer's poll loop grabs
           the NOT_STARTED clone.

    Does NOT restart the docker stack or Unity clients - it assumes the stack
    is already up and both Unity gyms are connected (this is the lightweight
    trainer-only restart, not Restart-Stack.ps1). Cloning is delegated entirely
    to Clone-Job.ps1 so the doc-shaping logic lives in exactly one place.

.PARAMETER SourceJobId
    24-char hex ObjectId of the job to clone + restart from.

.PARAMETER RewardDesignId
    Optional reward_designs ObjectId (string) to attach to the clone. See
    Clone-Job.ps1.

.PARAMETER RewardDesignName
    Optional cosmetic reward-design name stored on the clone.

.PARAMETER Notes
    Optional replacement for the clone's notes field.

.PARAMETER SetSourceDone
    Mark the source job status='DONE' before cloning (typical when retiring a
    stuck run). Forwarded to Clone-Job.ps1.

.PARAMETER NumEnvs
    Number of parallel Unity envs for the relaunched trainer. Default 2 (needs
    the scale overlay's ros-server-0..N-1 + that many connected Unity clients).

.PARAMETER SkipRelaunch
    Clone (and kill) but do NOT relaunch the trainer. Use when you want to
    inspect state or relaunch manually.

.PARAMETER SkipKill
    Do NOT kill the existing trainer first. Use only if you know none is
    running (otherwise the singleton lock will reject the relaunch).

.EXAMPLE
    # Retire the stuck job, clone it with the v4 reward design, restart 2-gym training.
    .\scripts\Restart-Job-From-Clone.ps1 -SourceJobId 6a6290d6c496c2efed601a04 `
        -RewardDesignId 6a3350e52ddfadd149b0db2f `
        -RewardDesignName 'Goal-count speed (v4, TIME_COST 0.0073)' `
        -SetSourceDone

.EXAMPLE
    # Clone + relaunch with no reward-design change, single env.
    .\scripts\Restart-Job-From-Clone.ps1 -SourceJobId <id> -NumEnvs 1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceJobId,
    [string]$RewardDesignId,
    [string]$RewardDesignName,
    [string]$Notes,
    [switch]$SetSourceDone,
    [int]$NumEnvs = 2,
    [switch]$SkipRelaunch,
    [switch]$SkipKill
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$cloneScript = Join-Path $PSScriptRoot 'Clone-Job.ps1'
if (-not (Test-Path -LiteralPath $cloneScript)) {
    throw "Clone-Job.ps1 not found at $cloneScript"
}
$composeArgs = @('-f', 'docker-compose.yml', '-f', 'compose/scale.yml')

Push-Location $repoRoot
try {
    # ---- 1. Kill the running trainer ------------------------------------
    if (-not $SkipKill) {
        Write-Host "=== Killing running trainer (if any) ==="
        # pkill's own cmdline contains the pattern, so it may SIGKILL its own
        # shell (exit 137/143) - that's expected and the trainer still dies.
        # Ignore the exit code and verify separately.
        docker compose @composeArgs exec -T sim-controller bash -c "pkill -9 -f 'python -u robotaxi.py'; sleep 2; true" 2>$null | Out-Null
        $remaining = (docker compose @composeArgs exec -T sim-controller bash -c "pgrep -af 'robotaxi.py' | grep -v pgrep | grep -v 'bash -c' || true" 2>$null) -join "`n"
        if ($remaining.Trim()) {
            Write-Warning "Trainer processes may still be alive:`n$remaining"
        } else {
            Write-Host "  Trainer stopped."
        }
    }

    # ---- 2. Clone the job (delegated to Clone-Job.ps1) ------------------
    Write-Host "=== Cloning job $SourceJobId ==="
    $cloneArgs = @{ SourceJobId = $SourceJobId }
    if ($RewardDesignId)   { $cloneArgs.RewardDesignId   = $RewardDesignId }
    if ($RewardDesignName) { $cloneArgs.RewardDesignName = $RewardDesignName }
    if ($Notes)            { $cloneArgs.Notes            = $Notes }
    if ($SetSourceDone)    { $cloneArgs.SetSourceDone    = $true }

    $cloneOutput = & $cloneScript @cloneArgs | Out-String
    Write-Host $cloneOutput

    $newId = $null
    if ($cloneOutput -match 'Clone inserted _id:\s*([0-9a-fA-F]{24})') {
        $newId = $Matches[1]
        Write-Host "  New clone job id: $newId"
    } else {
        throw "Could not parse new clone id from Clone-Job.ps1 output; aborting relaunch."
    }

    # ---- 3. Relaunch the trainer ---------------------------------------
    if ($SkipRelaunch) {
        Write-Host "=== Skipping relaunch (-SkipRelaunch). Clone $newId is NOT_STARTED and waiting. ==="
        return
    }

    Write-Host "=== Relaunching trainer (--num-envs $NumEnvs) ==="
    Write-Host "  (Ensure the stack is up and $NumEnvs Unity client(s) are connected.)"
    $trainerCmd = "cd /python_ws/src && python -u robotaxi.py --num-envs $NumEnvs 2>&1 | tee /tmp/trainer.log"
    docker compose @composeArgs exec -d sim-controller bash -c $trainerCmd
    if ($LASTEXITCODE -ne 0) { throw "trainer relaunch failed ($LASTEXITCODE)" }

    Write-Host ""
    Write-Host "Trainer relaunched. It should pick up NOT_STARTED clone $newId."
    Write-Host "Watch it with:"
    Write-Host "  docker compose $composeArgs exec sim-controller bash -c 'tail -f /tmp/trainer.log'"
} finally {
    Pop-Location
}
