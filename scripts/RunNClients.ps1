<#
.SYNOPSIS
    Spawn N supervised Unity clients (RunClientWrapper.ps1 instances), each
    in its own PowerShell window with a unique -Index.

.DESCRIPTION
    Pairs with compose/scale.yml: each client connects to the matching
    ros-server-N container by passing --ros-port (10000 + i).

    Each child wrapper opens its own console so its log output stays
    legible. Close any console window to stop supervising that client.

    Per-instance build copies: Unity's "Force Single Instance" Player
    Setting is baked into the build and locks on the .exe path. Without
    a workaround, only one instance of unity/Builds/latest/<game>.exe can
    run at a time. This script mirrors unity/Builds/latest/ into
    unity/Builds/instances/0..N-1/ (incremental robocopy, only changed
    files are copied) and points each supervisor at its own .exe path.
    Disk cost is one full build per index; rebuild + re-run picks up
    changes automatically.

    Typical workflow:
      1. docker compose -f docker-compose.yml -f compose/scale.yml up -d
      2. .\scripts\RunNClients.ps1 -N 4

    Make sure no other Unity client is running first (it will hold the
    instances/0 mutex, blocking actor 0 from launching).

.PARAMETER N
    Number of clients to spawn. Default 4. Each gets Index 0..N-1.

.PARAMETER RosIp
    Forwarded to each RunClientWrapper.ps1. Default 127.0.0.1, since
    Unity runs on the Windows host and reaches the ros-server containers
    via their published ports.

.PARAMETER WindowWidth
.PARAMETER WindowHeight
.PARAMETER WindowQuality
.PARAMETER Popup
    Forwarded to each RunClientWrapper.ps1. Defaults match the wrapper:
    960x540 windowed, "Fastest" quality. Pass -Popup for the historical
    320x240 borderless tile (good for stacking many actors on one
    screen during long training runs).

.PARAMETER StaggerSeconds
    Seconds to wait between supervisor spawns. Without a stagger, all N
    Unity instances race their D3D / Mono / Subsystems initialization
    simultaneously and lose under contention - some processes silently
    exit before any logger initializes. 3s gives the previous instance
    time to finish GfxDevice creation before the next one starts. Set
    to 0 to disable.
#>
[CmdletBinding()]
param(
    [int]$N = 4,
    [string]$RosIp = '127.0.0.1',
    [int]$WindowWidth = 0,
    [int]$WindowHeight = 0,
    [string]$WindowQuality = 'Fastest',
    [switch]$Popup,
    [double]$StaggerSeconds = 3.0
)

$ErrorActionPreference = 'Stop'

$wrapperPath = Join-Path $PSScriptRoot 'RunClientWrapper.ps1'
if (-not (Test-Path -LiteralPath $wrapperPath)) {
    throw "Wrapper not found: $wrapperPath"
}

if ($N -lt 1) { throw "N must be >= 1 (got $N)." }

# Resolve build paths.
$repoRoot      = Split-Path $PSScriptRoot -Parent
$latestDir     = Join-Path $repoRoot 'unity\Builds\latest'
$instancesRoot = Join-Path $repoRoot 'unity\Builds\instances'

if (-not (Test-Path -LiteralPath $latestDir)) {
    throw "No build at $latestDir. Build from Unity into unity/Builds/, then run scripts/PromoteLatestBuild.ps1."
}

# Find the game .exe at the top of latest/, ignoring Unity's crash handler.
$gameExes = @(
    Get-ChildItem -LiteralPath $latestDir -Filter '*.exe' -File |
        Where-Object { $_.Name -notlike 'UnityCrashHandler*.exe' }
)
if ($gameExes.Count -eq 0) {
    throw "No game .exe found at top of $latestDir."
}
if ($gameExes.Count -gt 1) {
    $names = ($gameExes | ForEach-Object { $_.Name }) -join ', '
    throw "Expected exactly one game .exe in $latestDir, found $($gameExes.Count): $names."
}
$exeName = $gameExes[0].Name

# Sync per-index copies via robocopy /MIR. /MIR mirrors changes only,
# so a steady-state re-run is fast. Quiet flags suppress robocopy's
# normal stat dump.
Write-Host "Syncing $N per-index build copies under unity\Builds\instances\..."
$indexExePaths = @()
for ($i = 0; $i -lt $N; $i++) {
    $instanceDir = Join-Path $instancesRoot $i
    if (-not (Test-Path -LiteralPath $instanceDir)) {
        New-Item -ItemType Directory -Force -Path $instanceDir | Out-Null
    }

    & robocopy $latestDir $instanceDir /MIR /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    # robocopy exit codes: 0..7 are success-with-info, >=8 is error.
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for index $i (exit code $LASTEXITCODE). Check $instanceDir."
    }
    # robocopy sets $LASTEXITCODE; reset so a benign code doesn't trip a later check.
    $global:LASTEXITCODE = 0

    $exePath = Join-Path $instanceDir $exeName
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Synced instance dir is missing the game exe: $exePath"
    }
    $indexExePaths += $exePath
}

Write-Host "Launching $N supervised Unity clients (stagger ${StaggerSeconds}s)..."
$pids = @()
# We build the spawned-powershell argument list as a single explicitly-
# quoted string. PowerShell 5.1's Start-Process -ArgumentList @(...) joins
# array elements with spaces but does NOT auto-quote, so a path arg
# containing "robot-tycoon Refactor" would be split mid-value when the
# child shell parses it - the supervisor would crash in param binding
# before ever launching Unity (and you'd see no Player.log).
for ($i = 0; $i -lt $N; $i++) {
    if ($i -gt 0 -and $StaggerSeconds -gt 0) {
        Start-Sleep -Seconds $StaggerSeconds
    }

    $exePath = $indexExePaths[$i]
    $cmdLine = @(
        '-NoExit'
        '-File "{0}"'           -f $wrapperPath
        '-Index {0}'            -f $i
        '-RosIp "{0}"'          -f $RosIp
        '-Path "{0}"'           -f $exePath
        '-WindowWidth {0}'      -f $WindowWidth
        '-WindowHeight {0}'     -f $WindowHeight
        '-WindowQuality "{0}"'  -f $WindowQuality
    ) -join ' '
    if ($Popup) { $cmdLine += ' -Popup' }

    $proc = Start-Process -FilePath 'powershell' -ArgumentList $cmdLine -PassThru
    $pids += $proc.Id
    Write-Host "  [$i] supervisor PID=$($proc.Id) -> $exePath"
}

Write-Host ""
Write-Host "All $N clients launched. Each has its own PowerShell window."
Write-Host "Supervisor PIDs: $($pids -join ', ')"
Write-Host "Close a window to stop supervising that client."
Write-Host ""
Write-Host "If the Unity windows don't appear, check the supervisor consoles for"
Write-Host "'Status = Exited: Restart...' loops - that means Unity is dying right"
Write-Host "after launch. Most common causes: another instance of the same .exe"
Write-Host "path is already running, or compose/scale.yml isn't up so port"
Write-Host "10001/10002/... has no listener."
