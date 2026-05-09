<#
.SYNOPSIS
    Stop and re-launch the Unity clients without touching Docker.

.DESCRIPTION
    ``Stop-Clients.ps1`` followed by ``Start-Clients.ps1`` with the
    same parameters. Use after editing RunClientWrapper.ps1, swapping
    in a freshly-built Unity binary, or wanting to re-grid the
    windows after a layout change - none of which need to disturb
    the running Docker containers.

    For a full stack restart (containers + Unity), use
    Restart-Stack.ps1 instead.

.PARAMETER N
    Number of Unity clients to launch on the way back up. Default 4.

.PARAMETER StaggerSeconds
    Seconds between supervisor spawns. Default 5.

.PARAMETER Popup
    Use Unity's -popupwindow flag for compact tile mode.

.PARAMETER GridCols
.PARAMETER GridRows
    Manual override of the auto-computed grid layout.

.PARAMETER Minimized
    Start supervisor consoles minimized. Default $true.

.PARAMETER UseWindowsTerminal
    Group supervisors as tabs in one wt window. Default $true.
#>
[CmdletBinding()]
param(
    [int]$N = 4,
    [double]$StaggerSeconds = 5,
    [switch]$Popup,
    [int]$GridCols = -1,
    [int]$GridRows = -1,
    [bool]$Minimized = $true,
    [bool]$UseWindowsTerminal = $true
)

$ErrorActionPreference = 'Stop'

$stop  = Join-Path $PSScriptRoot 'Stop-Clients.ps1'
$start = Join-Path $PSScriptRoot 'Start-Clients.ps1'

if (-not (Test-Path -LiteralPath $stop))  { throw "Stop-Clients.ps1 not found at $stop" }
if (-not (Test-Path -LiteralPath $start)) { throw "Start-Clients.ps1 not found at $start" }

Write-Host "=== Phase 1/2: Stop-Clients ==="
& $stop

Write-Host ""
Write-Host "=== Phase 2/2: Start-Clients ==="

$startArgs = @{
    N                  = $N
    StaggerSeconds     = $StaggerSeconds
    Minimized          = $Minimized
    UseWindowsTerminal = $UseWindowsTerminal
}
if ($Popup)          { $startArgs.Popup    = $true }
if ($GridCols -ge 0) { $startArgs.GridCols = $GridCols }
if ($GridRows -ge 0) { $startArgs.GridRows = $GridRows }

& $start @startArgs
