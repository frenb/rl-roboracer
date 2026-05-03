<#
.SYNOPSIS
    Launch and supervise one Unity gym client. Restarts it if the process
    becomes unresponsive or exits.

.DESCRIPTION
    By default, looks for a single .exe at the top of unity/Builds/latest/
    (the destination of scripts/PromoteLatestBuild.ps1).

    Multi-actor support: pass -Index 0..N-1 to derive a unique ROS port
    (10000 + Index) and have RosBootstrap.cs route the client to the
    matching ros-server-N container brought up by compose/scale.yml.

.PARAMETER Index
    Actor index (0 = the base ros-server, 1.. = ros-server-1, etc.).
    Determines the default --ros-port and is also used as a label in
    log output. Default 0 (single-client behavior).

.PARAMETER RosIp
    --ros-ip value passed to the Unity exe. Default host.docker.internal,
    which is how Docker Desktop exposes the host network.

.PARAMETER RosPort
    --ros-port value. Defaults to 10000 + Index.

.PARAMETER Path
    Optional explicit path to the Unity .exe. If omitted, the script globs
    unity/Builds/latest/*.exe and expects exactly one match.

.PARAMETER PollSeconds
    How often (seconds) to check whether the process is still responding.
    Default 5.

.PARAMETER WindowWidth
.PARAMETER WindowHeight
.PARAMETER WindowQuality
    Small-window flags forwarded to Unity. Defaults render to a 320x240
    popup at "Fastest" quality, which keeps GPU/VRAM cost low when
    several actors share one card. Pass -WindowWidth 1280 etc. to bring
    up a normal-sized client (e.g. for the dashboard's video stream).
#>
[CmdletBinding()]
param(
    [int]$Index = 0,
    [string]$RosIp = 'host.docker.internal',
    [int]$RosPort = 0,
    [string]$Path,
    [int]$PollSeconds = 5,
    [int]$WindowWidth = 320,
    [int]$WindowHeight = 240,
    [string]$WindowQuality = 'Fastest'
)

$ErrorActionPreference = 'Stop'

if ($RosPort -le 0) { $RosPort = 10000 + $Index }

if (-not $Path) {
    $latestDir = Join-Path $PSScriptRoot '..\unity\Builds\latest'
    if (-not (Test-Path -LiteralPath $latestDir)) {
        throw "No build at $latestDir. Build from Unity into a subfolder of unity/Builds/, then run scripts/PromoteLatestBuild.ps1."
    }
    $exes = @(Get-ChildItem -LiteralPath $latestDir -Filter '*.exe' -File)
    if ($exes.Count -eq 0) {
        throw "No .exe found at top of $latestDir."
    }
    if ($exes.Count -gt 1) {
        $names = ($exes | ForEach-Object { $_.Name }) -join ', '
        throw "Expected exactly one .exe in $latestDir, found $($exes.Count): $names. Pass -Path to disambiguate."
    }
    $Path = $exes[0].FullName
}

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Binary not found: $Path"
}

$exeArgs = @(
    '--ros-ip', $RosIp,
    '--ros-port', $RosPort,
    '-screen-width', $WindowWidth,
    '-screen-height', $WindowHeight,
    '-screen-quality', $WindowQuality,
    '-popupwindow'
)

function Start-Client {
    return Start-Process -FilePath $Path -ArgumentList $exeArgs -PassThru
}

$proc = Start-Client
Write-Host ("[{0}] started PID={1}, ROS endpoint {2}:{3}, window {4}x{5} {6}" -f `
    $Index, $proc.Id, $RosIp, $RosPort, $WindowWidth, $WindowHeight, $WindowQuality)

while ($true) {
    Start-Sleep -Seconds $PollSeconds

    # Re-fetch so .Responding reads live state.
    $current = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue

    if (-not $current) {
        Write-Host "[$Index] Status = Exited: Restart..."
        $proc = Start-Client
        Write-Host "[$Index] respawned PID=$($proc.Id)"
        continue
    }

    if (-not $current.Responding) {
        Write-Host "[$Index] Status = Not Responding (PID=$($current.Id)): Kill & Restart..."
        try { $current.Kill() } catch { Write-Host "[$Index] kill failed: $_" }
        $proc = Start-Client
        Write-Host "[$Index] respawned PID=$($proc.Id)"
    } else {
        Write-Host "[$Index] working fine (PID=$($current.Id))"
    }
}

# Source: https://community.idera.com/database-tools/powershell/ask_the_experts/f/powershell_for_windows-12/7002/how-to-detect-process-not-responding
