<#
.SYNOPSIS
    Starts the Unity client supervisor (RunClientWrapper.ps1) at sign-in.

.DESCRIPTION
    Install.ps1 puts a shortcut to this script in the account's Startup
    folder. Docker brings the containers back by itself after a sign-in or
    reboot; the Unity client is a Windows program and does not come back
    unless something starts it.

    Waits for Docker Desktop (starting it if it is not running) and for
    ros-server on port 10000, then runs the wrapper in this window. Exits
    without doing anything if a wrapper for this repo is already running.

.PARAMETER GymSource
    The gym build the wrapper starts from (passed through as -GymSource).

.PARAMETER WaitMinutes
    How long to wait for Docker and ros-server before giving up.
#>
[CmdletBinding()]
param(
    [string]$GymSource = '',
    [int]$WaitMinutes = 15
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$wrapper = Join-Path $repo 'scripts\RunClientWrapper.ps1'
$host.UI.RawUI.WindowTitle = "rl-roboracer Unity client 0 ($repo)"

$running = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and ($_.CommandLine -like "*$wrapper*" -or $_.CommandLine -like "*$PSCommandPath*") })
if ($running) { Write-Host "A Unity client wrapper for $repo is already running (pid $($running[0].ProcessId))."; exit 0 }

function Test-TcpPort([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.ConnectAsync('127.0.0.1', $Port).Wait(2000) -and $c.Connected } catch { $false } finally { $c.Dispose() }
}

function Test-DockerEngine {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & docker info *> $null; $LASTEXITCODE -eq 0 } catch { $false } finally { $ErrorActionPreference = $old }
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
        if ((Get-Date) -gt $deadline) { Write-Host "Docker did not start within $WaitMinutes minutes; not starting the Unity client."; Start-Sleep 60; exit 1 }
        Start-Sleep -Seconds 10
    }
}

Write-Host 'Waiting for ros-server on port 10000 ...'
while (-not (Test-TcpPort 10000)) {
    if ((Get-Date) -gt $deadline) { Write-Host "ros-server did not come up within $WaitMinutes minutes; not starting the Unity client."; Start-Sleep 60; exit 1 }
    Start-Sleep -Seconds 10
}

$wrapperArgs = @{ Index = 0; GymPollSeconds = 10 }
if ($GymSource) { $wrapperArgs.GymSource = $GymSource }
& $wrapper @wrapperArgs
