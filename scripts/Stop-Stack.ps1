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

.PARAMETER CloseWtWindow
    Whether to close the Windows Terminal window that hosted the
    supervisor tabs. Default $true. Without this, the wt window stays
    around with each tab showing ``[process exited]`` because wt's
    default ``closeOnExit`` policy is ``graceful`` (only auto-closes
    tabs that exited with code 0; hard-killed tabs persist as
    placeholders). Pass -CloseWtWindow:$false to leave wt open (e.g.
    if you want to scroll back through a crashed supervisor's last
    log lines before closing manually).

.EXAMPLE
    .\scripts\Stop-Stack.ps1

.EXAMPLE
    .\scripts\Stop-Stack.ps1 -KeepDocker

.EXAMPLE
    .\scripts\Stop-Stack.ps1 -CloseWtWindow:$false
#>
[CmdletBinding()]
param(
    [switch]$KeepDocker,
    [bool]$CloseWtWindow = $true
)

$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path $PSScriptRoot -Parent

# Shared stack-state helpers. The supervisors and the wt host self-
# register on startup, so we can find their PIDs from disk instead
# of asking WMI (which we used to do via Get-CimInstance
# Win32_Process and which freezes the whole script when the WMI
# service is wedged - common on long-running Windows sessions).
. (Join-Path $PSScriptRoot '_StackState.ps1')

# Aggregate the PIDs we want to kill in this object first, then
# do the actual Stop-Process pass in one place. Lets the discovery
# phase be fully best-effort without bleeding into the kill order.
$supervisorPids = @()
$wtPidsToClose  = @()

# 0a. Primary discovery: read the on-disk registry. Each entry has
# already been validated for liveness + process-image kind by
# Get-RegisteredSupervisors (it filters out stale PIDs and PIDs the
# OS has recycled to non-powershell processes).
$registered = @(Get-RegisteredSupervisors)
foreach ($r in $registered) {
    $supervisorPids += [int]$r.ProcessId
    $idxLabel = if ($r.Index -ge 0) { " (actor-$($r.Index))" } else { '' }
    Write-Host "  found registered supervisor PID=$($r.ProcessId)$idxLabel"
}

# 0b. wt host PID from the same registry.
if ($CloseWtWindow) {
    $wtPid = Get-WtHostPid
    if ($wtPid -gt 0) {
        $wtPidsToClose += $wtPid
        Write-Host "  found registered wt host PID=$wtPid"
    }
}

# 0c. Best-effort WMI fallback for stragglers - supervisors started
# under the old code that didn't self-register, or any leftover
# powershell.exe running RunClientWrapper.ps1 that for whatever
# reason isn't in the registry. Wrapped in a hard timeout so a
# wedged WMI service can't freeze the script: the foreground
# returns within $TimeoutSec with a warning, the stuck WMI call is
# orphaned in the background.
$wmiFallback = Invoke-WithTimeout -TimeoutSec 6 -Description 'WMI supervisor scan' -ScriptBlock {
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*RunClientWrapper.ps1*' } |
        Select-Object ProcessId, ParentProcessId
}
if ($wmiFallback) {
    foreach ($w in $wmiFallback) {
        if ($supervisorPids -notcontains [int]$w.ProcessId) {
            $supervisorPids += [int]$w.ProcessId
            Write-Host "  found legacy (non-registered) supervisor PID=$($w.ProcessId) via WMI fallback"
        }
    }
}

# 1. Kill supervisor PowerShell windows. Both the directly-registered
# set and any WMI-fallback stragglers go through the same loop.
Write-Host "Stopping supervisor PowerShell processes..."
$supervisorPids = @($supervisorPids | Sort-Object -Unique)
if ($supervisorPids.Count -eq 0) {
    Write-Host "  (none found)"
} else {
    foreach ($pidToKill in $supervisorPids) {
        Write-Host "  killing supervisor PID=$pidToKill"
        Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
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

# 4. Close the Windows Terminal window(s) that hosted the now-dead
# supervisors. Using Stop-Process (rather than WM_CLOSE) so we don't
# trigger wt's "confirm closing all tabs" prompt - the supervisors are
# already gone, the tabs are just stale [process exited] placeholders
# at this point.
$wtPidsToClose = @($wtPidsToClose | Sort-Object -Unique)
if ($CloseWtWindow -and $wtPidsToClose.Count -gt 0) {
    Write-Host "Closing Windows Terminal host(s)..."
    foreach ($wtPid in $wtPidsToClose) {
        $p = Get-Process -Id $wtPid -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "  closing wt PID=$wtPid"
            Stop-Process -Id $wtPid -Force -ErrorAction SilentlyContinue
        }
    }
}

# 5. Reset the stack-state dir now that everything tracked there is
# gone. The supervisors' own Unregister-Supervisor finally blocks
# don't run when we Stop-Process -Force them, so we have to clean
# up the leftovers ourselves. Without this, the next Start-Clients
# would race the consistency check on resurrected-but-stale PIDs
# (harmless but noisy).
Clear-StackState

Write-Host ""
Write-Host "Stack down."
