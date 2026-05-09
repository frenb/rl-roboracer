<#
.SYNOPSIS
    Bring the entire robotaxi stack up: docker compose containers
    (with the scale overlay) followed by N Unity clients, each in
    its own supervised PowerShell window.

.DESCRIPTION
    Two phases:

    1. ``docker compose -f docker-compose.yml -f compose/scale.yml up -d``
       starts the base + scale-overlay services (ros-server,
       ros-server-{1..3}, sim-controller, mongo, mongo-express,
       dashboard). Run this from the host so the per-actor host port
       forwards (10000-10003 for ROS-TCP, 50051-50054 for gRPC) get
       bound by Docker Desktop's vpnkit/com.docker.backend.

    2. ``.\scripts\RunNClients.ps1 -N <N>`` spawns N supervised Unity
       clients on the host. Each one gets a unique --ros-port (10000+i)
       and --unity-port (5005+i) so the bidirectional ROS-TCP-Connector
       protocol routes correctly per actor.

    A small wait between the two phases gives the ros-server containers
    time to finish their internal start.sh (rosmaster + ROS-TCP
    endpoint listener) so the first Unity handshake doesn't see a
    not-yet-listening port. Default 8s is conservative; tune via
    -WaitForRosServersSeconds if your machine is faster.

.PARAMETER N
    Number of Unity clients to spawn. Default 4. Forwarded to
    RunNClients.ps1.

.PARAMETER StaggerSeconds
    Seconds between spawning each Unity client. Default 5. Multi-actor
    GPU contention during simultaneous D3D / Mono / scene init causes
    silent exits without staggering, see RunNClients.ps1 docs.

.PARAMETER Popup
    Use Unity's -popupwindow flag (small borderless tiles, useful when
    several actors share one screen). Forwarded to RunNClients.ps1.

.PARAMETER WaitForRosServersSeconds
    Seconds to wait after ``docker compose up -d`` before launching
    Unity clients. Default 8. Increase if your first Unity handshake
    consistently logs "Connection refused" before connecting.

.PARAMETER SkipUnity
    Bring up only the docker stack, skip the Unity launch. Useful when
    you want to control Unity timing manually (e.g. to debug a single
    instance).

.EXAMPLE
    .\scripts\Start-Stack.ps1

.EXAMPLE
    .\scripts\Start-Stack.ps1 -N 2 -Popup

.EXAMPLE
    .\scripts\Start-Stack.ps1 -SkipUnity
#>
[CmdletBinding()]
param(
    [int]$N = 4,
    [double]$StaggerSeconds = 5,
    [switch]$Popup,
    [int]$WaitForRosServersSeconds = 8,
    [switch]$SkipUnity
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path $PSScriptRoot -Parent

Write-Host "Starting docker compose stack (with scale overlay)..."
Push-Location $repoRoot
try {
    docker compose -f docker-compose.yml -f compose/scale.yml up -d
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

if ($SkipUnity) {
    Write-Host ""
    Write-Host "Skipping Unity launch (-SkipUnity). Stack docker side is up."
    return
}

Write-Host "Waiting ${WaitForRosServersSeconds}s for ros-server containers to start listening..."
Start-Sleep -Seconds $WaitForRosServersSeconds

Write-Host "Launching $N Unity clients via Start-Clients.ps1..."
# Delegate to Start-Clients (rather than calling RunNClients
# directly) so we inherit its idempotent cleanup-first behavior:
# if a previous Start-Stack left supervisors or Unity instances
# running, they get killed before we spawn the new set rather than
# piling up additional wt tabs and Unity processes.
$startClients = Join-Path $PSScriptRoot 'Start-Clients.ps1'
if (-not (Test-Path -LiteralPath $startClients)) {
    throw "Start-Clients.ps1 not found at $startClients"
}

$invokeArgs = @{
    N              = $N
    StaggerSeconds = $StaggerSeconds
}
if ($Popup) { $invokeArgs.Popup = $true }
& $startClients @invokeArgs

Write-Host ""
Write-Host "Stack up. Multi-env training can now be started with:"
$composeArgs = "docker compose -f docker-compose.yml -f compose/scale.yml"
# Two important pieces of the printed command:
#
# `python -u`: forces unbuffered stdout. When Python detects stdout is
# a pipe (which it is via `| tee`), it switches from line-buffered to
# 8 KB block-buffered, so [actor-N] lines get held back for several
# seconds at a time. -u disables that so the dashboard log view and
# the user's terminal both see lines as they're emitted.
#
# `| tee robotaxi.out`: the dashboard log panel
# (dashboard/src/server.ts) tails /python_ws/src/robotaxi.out over a
# WebSocket; without `tee`, the file stays stale and the panel shows
# old data because the scale-overlay sim-controller no longer writes
# to it from its default command (the auto-run trainer is disabled in
# overlay mode - see compose/scale.yml).
Write-Host "  $composeArgs exec sim-controller bash -c 'cd /python_ws/src && python -u robotaxi.py --num-envs $N 2>&1 | tee robotaxi.out'"
