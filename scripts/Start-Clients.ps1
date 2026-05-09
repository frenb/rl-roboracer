<#
.SYNOPSIS
    Launch N Unity clients and their PowerShell supervisor tabs.
    Assumes the Docker stack (ros-servers, sim-controller, etc.) is
    already up.

.DESCRIPTION
    Thin wrapper around RunNClients.ps1 that forwards the
    most-commonly-used parameters under a discoverable Verb-Noun name.
    For lower-level control or extra knobs, call RunNClients.ps1
    directly - the parameter set is identical.

    Use when:
      - You've made changes to RunClientWrapper.ps1 (PollSeconds,
        UnresponsiveStrikes, grid placement, etc.) and want to
        cycle the Unity side
      - You rebuilt the Unity binary and promoted it
      - The Docker stack is already healthy and you don't want to
        churn it (loses sim-controller's warm reverb buffer + state)

    For a full stack restart, use Restart-Stack.ps1 instead.

.PARAMETER N
    Number of Unity clients to launch. Default 4.

.PARAMETER StaggerSeconds
    Seconds between supervisor spawns. Default 5.

.PARAMETER Popup
    Use Unity's -popupwindow flag for compact tile mode.

.PARAMETER GridCols
.PARAMETER GridRows
    Manual override of the auto-computed grid layout. Defaults to a
    square-ish layout (e.g. 2x2 for N=4). Pass GridCols=0 to disable
    grid placement entirely.

.PARAMETER Minimized
    Start supervisor consoles minimized to taskbar. Default $true.

.PARAMETER UseWindowsTerminal
    Group supervisors as tabs in one Windows Terminal window.
    Default $true; auto-falls back to separate consoles if wt
    isn't on PATH.

.PARAMETER Append
    Skip the implicit Stop-Clients cleanup at the start of this
    script. Default $false (we DO clean up first), which makes
    Start-Clients idempotent: calling it twice in a row results in
    exactly N clients, not 2N. Pass -Append to keep any existing
    supervisors + Unity instances and just stack new ones on top -
    almost never what you want, but useful in narrow debugging
    scenarios (e.g. attaching one extra actor without recycling
    in-flight runs).

.EXAMPLE
    .\scripts\Start-Clients.ps1
        # 4 Unity clients in a 2x2 grid, supervisors as wt tabs minimized.
        # Cleans up any pre-existing supervisors / Unity first.

.EXAMPLE
    .\scripts\Start-Clients.ps1 -N 2 -GridCols 2 -GridRows 1
        # 2 clients side-by-side

.EXAMPLE
    .\scripts\Start-Clients.ps1 -Append
        # Spawn 4 more on top of whatever is already running. Power-user
        # mode; almost never what you want.
#>
[CmdletBinding()]
param(
    [int]$N = 4,
    [double]$StaggerSeconds = 5,
    [switch]$Popup,
    [int]$GridCols = -1,
    [int]$GridRows = -1,
    [bool]$Minimized = $true,
    [bool]$UseWindowsTerminal = $true,
    [switch]$Append
)

$ErrorActionPreference = 'Stop'

# Idempotency: by default, kill any pre-existing supervisor tabs and
# Unity instances before launching the new set. Without this, calling
# Start-Clients twice piles new wt tabs (and new Unity processes) on
# top of the previous batch, which is almost certainly not what the
# user wants. The -Append switch opts out for the rare case it is.
if (-not $Append) {
    $stopClients = Join-Path $PSScriptRoot 'Stop-Clients.ps1'
    if (Test-Path -LiteralPath $stopClients) {
        Write-Host "Cleaning up any existing supervisors / Unity clients first..."
        & $stopClients
        Write-Host ""
    }
}

$runNClients = Join-Path $PSScriptRoot 'RunNClients.ps1'
if (-not (Test-Path -LiteralPath $runNClients)) {
    throw "RunNClients.ps1 not found at $runNClients"
}

$invokeArgs = @{
    N                  = $N
    StaggerSeconds     = $StaggerSeconds
    Minimized          = $Minimized
    UseWindowsTerminal = $UseWindowsTerminal
}
if ($Popup)          { $invokeArgs.Popup    = $true }
if ($GridCols -ge 0) { $invokeArgs.GridCols = $GridCols }
if ($GridRows -ge 0) { $invokeArgs.GridRows = $GridRows }

& $runNClients @invokeArgs
