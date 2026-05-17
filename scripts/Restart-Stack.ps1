<#
.SYNOPSIS
    Bring the robotaxi stack fully down and then back up, in one
    command. Equivalent to ``Stop-Stack.ps1`` followed by
    ``Start-Stack.ps1`` with the same parameters.

.DESCRIPTION
    Useful when you've made a code change that needs a clean container
    state to take effect (e.g. modifying docker-compose.yml's command:
    line, changing a service's network alias, updating a baked-in
    Dockerfile dependency). For pure rl_agent/ Python edits or
    rl_agent/ replay.py edits the bind mount picks them up live and
    you don't need a restart - re-run ``python robotaxi.py --num-envs
    N`` instead.

    All Unity-launch parameters are forwarded to Start-Stack.ps1.
    Stop-Stack.ps1 always performs a full ``docker compose down`` here
    (no -KeepDocker pass-through, since the whole point of restart is
    to recycle the containers).

.PARAMETER N
    Number of Unity clients to spawn on the way back up. Default 2.

.PARAMETER StaggerSeconds
    Seconds between spawning each Unity client. Default 15. See
    RunNClients.ps1 for the multi-actor GPU-init rationale.

.PARAMETER Popup
    Use Unity's -popupwindow flag for the relaunched clients.

.PARAMETER WaitForRosServersSeconds
    Seconds to wait after ``docker compose up -d`` before launching
    Unity. Default 8.

.PARAMETER SkipUnity
    Skip the Unity launch on the way back up. Same effect as
    ``Start-Stack.ps1 -SkipUnity``.

.EXAMPLE
    .\scripts\Restart-Stack.ps1

.EXAMPLE
    .\scripts\Restart-Stack.ps1 -N 2 -Popup
#>
[CmdletBinding()]
param(
    [int]$N = 2,
    [double]$StaggerSeconds = 15,
    [switch]$Popup,
    [int]$WaitForRosServersSeconds = 8,
    [switch]$SkipUnity
)

$ErrorActionPreference = 'Stop'

$stop  = Join-Path $PSScriptRoot 'Stop-Stack.ps1'
$start = Join-Path $PSScriptRoot 'Start-Stack.ps1'

if (-not (Test-Path -LiteralPath $stop))  { throw "Stop-Stack.ps1 not found at $stop" }
if (-not (Test-Path -LiteralPath $start)) { throw "Start-Stack.ps1 not found at $start" }

Write-Host "=== Phase 1/2: Stop-Stack ==="
& $stop

Write-Host ""
Write-Host "=== Phase 2/2: Start-Stack ==="
$startArgs = @{
    N                        = $N
    StaggerSeconds           = $StaggerSeconds
    WaitForRosServersSeconds = $WaitForRosServersSeconds
}
if ($Popup)     { $startArgs.Popup     = $true }
if ($SkipUnity) { $startArgs.SkipUnity = $true }
& $start @startArgs
