<#
.SYNOPSIS
    Bring the entire robotaxi stack down: Unity clients, their
    supervisor PowerShell windows, and all docker compose containers
    (including the scale-overlay ros-server-{1..3} services).

.DESCRIPTION
    Order matters: kill the supervisors first so they stop respawning
    Unity, then kill the Unity client processes themselves, then tear
    down docker. If we kill Unity before the supervisor, the supervisor
    would race to Start-Client a fresh instance and we'd be playing
    whack-a-mole.

    Every step is best-effort and never aborts the script - if a Unity
    process is already gone, that's fine; if no supervisors are running,
    that's fine; if ``docker compose down`` fails, the function still
    returns and prints what happened.

    Sim-controller's running ``python robotaxi.py`` (whether the default
    one from the container's ``command:`` or one started via
    ``docker compose ... exec``) is terminated automatically when the
    container is stopped, so there's nothing extra to do for "running
    jobs" - they die with their host container.

.PARAMETER KeepDocker
    Skip the ``docker compose down`` step. Useful when iterating on the
    Unity-side launch without churning the containers (rebuilds reverb
    state, MongoDB connections, etc.).

.EXAMPLE
    .\scripts\Stop-Stack.ps1

.EXAMPLE
    .\scripts\Stop-Stack.ps1 -KeepDocker
#>
[CmdletBinding()]
param(
    [switch]$KeepDocker
)

$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path $PSScriptRoot -Parent

# 1. Supervisor PowerShell windows. RunNClients.ps1 spawns each
# RunClientWrapper.ps1 instance under its own powershell.exe with
# -NoExit so the console stays open for log inspection. Detect them by
# CommandLine substring rather than by PID, since we don't want to
# rely on a sidecar pid file (gets out of sync if a window was closed
# manually). Win32_Process exposes CommandLine for processes the
# current user owns without elevation.
Write-Host "Stopping supervisor PowerShell processes..."
$supervisors = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*RunClientWrapper.ps1*' })
if ($supervisors.Count -eq 0) {
    Write-Host "  (none found)"
} else {
    foreach ($s in $supervisors) {
        Write-Host "  killing supervisor PID=$($s.ProcessId)"
        Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

# 2. Unity client processes themselves. Multi-instance runs all share
# the same ProcessName ("robotaxi gym level 1"), so a name-based kill
# catches all of them in one go.
Write-Host "Stopping Unity clients..."
$unities = @(Get-Process -Name 'robotaxi gym level 1' -ErrorAction SilentlyContinue)
if ($unities.Count -eq 0) {
    Write-Host "  (none found)"
} else {
    foreach ($u in $unities) {
        Write-Host "  killing Unity PID=$($u.Id) -> $($u.Path)"
        Stop-Process -Id $u.Id -Force -ErrorAction SilentlyContinue
    }
}

# 3. Docker compose stack. Pass BOTH compose files so the scale-overlay
# ros-server-{1..3} services come down too - plain ``docker compose
# down`` only knows about docker-compose.yml's services and would leave
# the overlay containers running, blocking the network from being
# removed (we hit this footgun earlier this session).
if (-not $KeepDocker) {
    Write-Host "Stopping docker compose stack (with scale overlay)..."
    Push-Location $repoRoot
    try {
        docker compose -f docker-compose.yml -f compose/scale.yml down
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "docker compose down exited with code $LASTEXITCODE; check output above."
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Skipping docker compose down (-KeepDocker)."
}

Write-Host ""
Write-Host "Stack down."
