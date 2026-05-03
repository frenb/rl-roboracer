<#
.SYNOPSIS
    Spawn N supervised Unity clients (RunClientWrapper.ps1 instances), each
    in its own PowerShell window with a unique -Index.

.DESCRIPTION
    Pairs with compose/scale.yml: each client connects to the matching
    ros-server-N container by passing --ros-port (10000 + i).

    Each child wrapper opens its own console so its log output stays
    legible. Close any console window to stop supervising that client.

    Typical workflow:
      1. docker compose -f docker-compose.yml -f compose/scale.yml up -d
      2. .\scripts\RunNClients.ps1 -N 4

.PARAMETER N
    Number of clients to spawn. Default 4. Each gets Index 0..N-1.

.PARAMETER WindowWidth
.PARAMETER WindowHeight
.PARAMETER WindowQuality
    Forwarded to each RunClientWrapper.ps1 invocation. Defaults match
    the wrapper's defaults (320x240 Fastest popup).
#>
[CmdletBinding()]
param(
    [int]$N = 4,
    [int]$WindowWidth = 320,
    [int]$WindowHeight = 240,
    [string]$WindowQuality = 'Fastest'
)

$ErrorActionPreference = 'Stop'

$wrapperPath = Join-Path $PSScriptRoot 'RunClientWrapper.ps1'
if (-not (Test-Path -LiteralPath $wrapperPath)) {
    throw "Wrapper not found: $wrapperPath"
}

if ($N -lt 1) { throw "N must be >= 1 (got $N)." }

Write-Host "Launching $N supervised Unity clients..."
$pids = @()
for ($i = 0; $i -lt $N; $i++) {
    $childArgs = @(
        '-NoExit',
        '-File', $wrapperPath,
        '-Index', $i,
        '-WindowWidth', $WindowWidth,
        '-WindowHeight', $WindowHeight,
        '-WindowQuality', $WindowQuality
    )
    $proc = Start-Process -FilePath 'powershell' -ArgumentList $childArgs -PassThru
    $pids += $proc.Id
    Write-Host "  [$i] supervisor PID=$($proc.Id)"
}

Write-Host ""
Write-Host "All $N clients launched. Each has its own PowerShell window."
Write-Host "Supervisor PIDs: $($pids -join ', ')"
Write-Host "Close a window to stop supervising that client."
