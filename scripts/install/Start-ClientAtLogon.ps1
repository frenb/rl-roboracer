<#
.SYNOPSIS
    Starts the supervised Unity clients (RunClientWrapper.ps1) at sign-in.

.DESCRIPTION
    Install.ps1 puts a shortcut to this script in the account's Startup
    folder. Docker brings the containers back by itself after a sign-in or
    reboot; the Unity clients are Windows programs and do not come back
    unless something starts them.

    Waits for Docker Desktop (starting it if it is not running) and for
    ros-server on port 10000.

    -N 1 (the default) runs a single wrapper in this window, and exits
    without doing anything if a wrapper for this repo is already running.

    -N 2..4 also starts ros-server-1..N-1 from compose/scale.yml (each
    client needs its own bridge, on ports 10001..10003), then hands off to
    Start-Clients.ps1, which stops any running clients and launches N
    supervisors in Windows Terminal tabs. The trainer has to be started
    with --num-envs N to use them. Fly courses need N = 1: the fly-brain
    service holds a single brain.

.PARAMETER N
    Number of Unity clients, 1 to 4. Default 1.

.PARAMETER GymSource
    The gym build a single client starts from (passed through as
    -GymSource). With N > 1 the clients start from unity\Builds\latest.

.PARAMETER WaitMinutes
    How long to wait for Docker and ros-server before giving up.

.EXAMPLE
    .\scripts\install\Start-ClientAtLogon.ps1 -N 2
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 4)][int]$N = 1,
    [string]$GymSource = '',
    [int]$WaitMinutes = 15
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$wrapper = Join-Path $repo 'scripts\RunClientWrapper.ps1'
$host.UI.RawUI.WindowTitle = "rl-roboracer Unity clients x$N ($repo)"

if ($N -eq 1) {
    $running = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and ($_.CommandLine -like "*$wrapper*" -or $_.CommandLine -like "*$PSCommandPath*") })
    if ($running) { Write-Host "A Unity client wrapper for $repo is already running (pid $($running[0].ProcessId))."; exit 0 }
}

function Test-TcpPort([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.ConnectAsync('127.0.0.1', $Port).Wait(2000) -and $c.Connected } catch { $false } finally { $c.Dispose() }
}

function Test-DockerEngine {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & docker info *> $null; $LASTEXITCODE -eq 0 } catch { $false } finally { $ErrorActionPreference = $old }
}

function Stop-Launcher([string]$Message) {
    Write-Host "$Message; not starting the Unity clients."
    Start-Sleep -Seconds 60
    exit 1
}

$deadline = (Get-Date).AddMinutes($WaitMinutes)
if (-not (Test-DockerEngine)) {
    $dockerExe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue) -and (Test-Path $dockerExe)) {
        Write-Host 'Starting Docker Desktop ...'
        Start-Process -FilePath $dockerExe | Out-Null
    }
    Write-Host 'Waiting for the Docker engine ...'
    while (-not (Test-DockerEngine)) {
        if ((Get-Date) -gt $deadline) { Stop-Launcher "Docker did not start within $WaitMinutes minutes" }
        Start-Sleep -Seconds 10
    }
}

if ($N -gt 1) {
    $extra = @(1..($N - 1) | ForEach-Object { "ros-server-$_" })
    Write-Host "Starting $($extra -join ', ') ..."
    Push-Location $repo
    try {
        & docker compose -f docker-compose.yml -f compose/scale.yml up -d --no-build @extra
        if ($LASTEXITCODE -ne 0) { Stop-Launcher "docker compose up $($extra -join ' ') failed (exit $LASTEXITCODE)" }
    } finally { Pop-Location }
}

$ports = @(0..($N - 1) | ForEach-Object { 10000 + $_ })
Write-Host "Waiting for ros-server on port(s) $($ports -join ', ') ..."
while (@($ports | Where-Object { -not (Test-TcpPort $_) }).Count) {
    if ((Get-Date) -gt $deadline) { Stop-Launcher "ros-server did not come up within $WaitMinutes minutes" }
    Start-Sleep -Seconds 10
}

if ($N -eq 1) {
    $wrapperArgs = @{ Index = 0; GymPollSeconds = 10 }
    if ($GymSource) { $wrapperArgs.GymSource = $GymSource }
    & $wrapper @wrapperArgs
} else {
    & (Join-Path $repo 'scripts\Start-Clients.ps1') -N $N
    Write-Host "Started $N clients. Run the trainer with --num-envs $N to use all of them."
}
