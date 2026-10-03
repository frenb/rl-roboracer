<#
.SYNOPSIS
    Install rl-roboracer on a Windows PC with an NVIDIA GPU, from nothing to a
    first TRAIN job producing training steps.

.DESCRIPTION
    Runs nine phases in order. Each phase first checks whether its work is
    already done, so the script can be re-run at any time and resumes where
    it stopped (including after the reboot WSL needs).

      0 preflight   Windows build, RAM, disk, virtualization, NVIDIA driver,
                    winget, network
      1 wsl         WSL 2 (wsl --install --no-distribution; needs a reboot)
      2 git         Git for Windows (winget)
      3 docker      Docker Desktop (winget), engine running, GPU visible
      4 workspace   Repo clone, sibling data folders, .env
      5 images      ros-server + sim-controller images built and verified
      6 unity       Unity client unpacked into unity\Builds\latest\
      7 start       Stack up, one Unity client connected to ros-server
      8 firstjob    A short TRAIN job queued and producing training steps

    Every phase records what it found and what it did in
    install-report.json, with outcome installed / already-present /
    skipped / passed / failed. -ExpectClean turns "already-present" into a
    failure, which is how a test on a supposedly clean machine catches an
    install step that silently did nothing.

    Run from a downloaded copy:
      powershell -ExecutionPolicy Bypass -File .\Install.ps1

    Exit codes: 0 done, 1 failed, 2 reboot needed (resumes at next sign-in),
    3 sign out and back in needed (re-run afterwards).

.PARAMETER InstallDir
    Parent folder. The repo is cloned into <InstallDir>\rl-roboracer and the
    data folders (saved_models, mongodb, tfrecords) sit beside it.

.PARAMETER RepoUrl
    Git URL or local path to clone from.

.PARAMETER Branch
    Branch to check out.

.PARAMETER UnityZip
    Unity client zip: a URL or a local path. Defaults to the GitHub Release
    asset for -UnityReleaseTag. A "<zip>.sha256" next to it is verified when
    present.

.PARAMETER ProjectName
    Compose project name written to .env (COMPOSE_PROJECT_NAME). Only needed
    to keep a test install apart from an existing one.

.PARAMETER ImagePrefix
    Prefix for locally built image names, written to .env (IMAGE_PREFIX).
    Same purpose as -ProjectName.

.PARAMETER FirstJobIterations
    num_iterations of the first TRAIN job.

.PARAMETER FirstJobWaitMinutes
    How long to wait for the first job to report training steps.

.PARAMETER ExpectClean
    Fail any phase that finds its work already done. For testing on a clean
    machine (combine with -SkipPrereqs for a clean folder on a machine that
    already has WSL, Git and Docker).

.PARAMETER DryRun
    Only detect. Prints what each phase would do; changes nothing.

.PARAMETER SkipPrereqs
    Skip phases 1-3 (WSL, Git, Docker Desktop must already work).

.PARAMETER NoFirstJob
    Stop after phase 7.

.PARAMETER AcceptDockerLicense
    Accept the Docker Desktop Subscription Service Agreement without asking.

.PARAMETER Rebuild
    Rebuild the images even if they exist and pass verification.

.PARAMETER BuildRetries
    Attempts per image build. Docker Desktop's network intermittently
    truncates downloads; finished build steps are cached, so a retry resumes.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'rl-roboracer'),
    [string]$RepoUrl = 'https://github.com/frenb/rl-roboracer.git',
    [string]$Branch = 'main',
    [string]$UnityZip = '',
    [string]$UnityReleaseTag = 'v0.1',
    [string]$ProjectName = '',
    [string]$ImagePrefix = '',
    [int]$FirstJobIterations = 2000,
    [int]$FirstJobWaitMinutes = 30,
    [switch]$ExpectClean,
    [switch]$DryRun,
    [switch]$SkipPrereqs,
    [switch]$NoFirstJob,
    [switch]$AcceptDockerLicense,
    [switch]$Rebuild,
    [int]$BuildRetries = 6
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Windows PowerShell 5.1 still offers TLS 1.0 by default; GitHub refuses it.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$ScriptParams = $PSBoundParameters

$GithubRepo     = 'frenb/rl-roboracer'
$UnityAssetName = 'roboracer-unity-windows.zip'
$MinBuild       = 19045
$MinDriverMajor = 495
$CudaTestImage  = 'nvidia/cuda:11.0.3-base-ubuntu20.04'
$BuildServices  = @('ros-server', 'sim-controller')
$StartServices  = @('ros-server', 'mongo', 'mongo-express', 'sim-controller', 'dashboard')
$DockerExe      = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'

$StateDir   = Join-Path $env:LOCALAPPDATA 'rl-roboracer-install'
$ReportPath = Join-Path $StateDir 'install-report.json'
$RepoDir    = Join-Path $InstallDir 'rl-roboracer'
$RunOnceKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
$RunOnceName = 'rl-roboracer-install'

New-Item -ItemType Directory -Force $StateDir | Out-Null
$LogPath = Join-Path $StateDir ("install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $LogPath -Append | Out-Null

$script:Report = New-Object System.Collections.ArrayList

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Phase([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Info([string]$Text)  { Write-Host "    $Text" }
function Write-Warn([string]$Text)  { Write-Host "    WARNING: $Text" -ForegroundColor Yellow }

class PhaseFailure : System.Exception {
    PhaseFailure([string]$m) : base($m) {}
}

function Stop-Phase([string]$Message) { throw [PhaseFailure]::new($Message) }

# Native commands write progress to stderr; under $ErrorActionPreference =
# 'Stop' Windows PowerShell 5.1 turns that into a terminating error. Run them
# with 'Continue' and judge success by the exit code alone. Arguments must not
# contain double quotes: 5.1 passes those to native programs incorrectly.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments = @(), [switch]$Quiet)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } catch {
        $lines = @("$_")
        $code = 9009
    } finally {
        $ErrorActionPreference = $old
    }
    if (-not $Quiet -and $code -ne 0 -and $lines) { $lines | Select-Object -Last 5 | ForEach-Object { Write-Info "| $_" } }
    [pscustomobject]@{ Code = $code; Out = (@($lines) -join "`n") }
}

function Test-Native([string]$Exe, [string[]]$Arguments = @()) {
    (Invoke-Native $Exe $Arguments -Quiet).Code -eq 0
}

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
    $dockerBin = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin'
    if ((Test-Path $dockerBin) -and ($env:Path -notlike "*$dockerBin*")) { $env:Path += ";$dockerBin" }
}

function Invoke-Elevated([string]$Exe, [string]$ArgumentLine) {
    Write-Info "Asking for administrator rights to run: $Exe $ArgumentLine"
    $p = Start-Process -FilePath $Exe -ArgumentList $ArgumentLine -Verb RunAs -Wait -PassThru
    $p.ExitCode
}

function Wait-Until([scriptblock]$Condition, [int]$TimeoutSec, [int]$IntervalSec = 5, [string]$What = '') {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $announced = $false
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) { return $true }
        if ($What -and -not $announced) { Write-Info "waiting for $What (up to $TimeoutSec s) ..."; $announced = $true }
        Start-Sleep -Seconds $IntervalSec
    }
    return [bool](& $Condition)
}

function Test-TcpPort([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.ConnectAsync('127.0.0.1', $Port).Wait(2000) -and $c.Connected } catch { $false } finally { $c.Dispose() }
}

function Save-Report {
    $script:Report | ConvertTo-Json -Depth 4 | Set-Content -Path $ReportPath -Encoding UTF8
}

function Add-Result([string]$Phase, [string]$Outcome, [string]$Evidence, [double]$Seconds) {
    [void]$script:Report.Add([pscustomobject]@{
        phase = $Phase; outcome = $Outcome; evidence = $Evidence
        seconds = [math]::Round($Seconds, 1); at = (Get-Date).ToString('s')
    })
    Save-Report
    $color = switch ($Outcome) { 'failed' { 'Red' } 'installed' { 'Green' } 'passed' { 'Green' } default { 'Gray' } }
    Write-Host ("    [{0}] {1}" -f $Outcome, $Evidence) -ForegroundColor $color
}

# A phase is a detect block (returns $null when work is needed, or evidence
# text when it is already done) and an install block (does the work, returns
# evidence). Run-Phase applies -DryRun / -ExpectClean uniformly.
function Invoke-Phase {
    param([string]$Name, [string]$Title, [scriptblock]$Detect, [scriptblock]$Install, [switch]$NoExpectClean)
    Write-Phase "[$Name] $Title"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $found = & $Detect
        if ($found) {
            if ($ExpectClean -and -not $NoExpectClean) {
                Add-Result $Name 'failed' "expected a clean machine, but found: $found" $sw.Elapsed.TotalSeconds
                throw [PhaseFailure]::new("$Name was already present (-ExpectClean)")
            }
            Add-Result $Name 'already-present' $found $sw.Elapsed.TotalSeconds
            return
        }
        if ($DryRun) {
            Add-Result $Name 'would-install' 'not present' $sw.Elapsed.TotalSeconds
            return
        }
        $evidence = & $Install
        Add-Result $Name 'installed' $evidence $sw.Elapsed.TotalSeconds
    } catch [PhaseFailure] {
        if (-not ($script:Report | Where-Object { $_.phase -eq $Name })) {
            Add-Result $Name 'failed' $_.Exception.Message $sw.Elapsed.TotalSeconds
        }
        throw
    } catch {
        Add-Result $Name 'failed' "$($_.Exception.Message)" $sw.Elapsed.TotalSeconds
        throw [PhaseFailure]::new("$Name failed: $($_.Exception.Message)")
    }
}

function Register-Resume {
    $self = Join-Path $StateDir 'Install.ps1'
    if ($PSCommandPath -and ($PSCommandPath -ne $self)) { Copy-Item -LiteralPath $PSCommandPath -Destination $self -Force }
    if (-not (Test-Path $self)) {
        Write-Warn 'Cannot resume automatically (the script was not run from a file). Run it again after the restart.'
        return
    }
    $argList = @()
    foreach ($kv in $ScriptParams.GetEnumerator()) {
        if ($kv.Value -is [switch]) { if ($kv.Value) { $argList += "-$($kv.Key)" } }
        else { $argList += "-$($kv.Key) `"$($kv.Value)`"" }
    }
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File `"$self`" $($argList -join ' ')"
    New-Item -Path $RunOnceKey -Force | Out-Null
    Set-ItemProperty -Path $RunOnceKey -Name $RunOnceName -Value $cmd
}

function Invoke-Compose([string[]]$Arguments, [switch]$Quiet) {
    Push-Location $RepoDir
    try { Invoke-Native 'docker' (@('compose') + $Arguments) -Quiet:$Quiet } finally { Pop-Location }
}

function Get-ComposeImages {
    $r = Invoke-Compose @('config', '--format', 'json') -Quiet
    if ($r.Code -ne 0) { Stop-Phase "docker compose config failed: $($r.Out)" }
    $cfg = $r.Out | ConvertFrom-Json
    $map = @{}
    foreach ($s in $BuildServices) { $map[$s] = $cfg.services.$s.image }
    $map
}

function Get-EnvValue([string]$Key) {
    $envFile = Join-Path $RepoDir '.env'
    if (-not (Test-Path $envFile)) { return $null }
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -Last 1
    if ($line) { return ($line -split '=', 2)[1].Trim() }
    return $null
}

# Runs a shell script inside an image. Passed base64-encoded so no quoting
# survives the trip through PowerShell 5.1's native argument handling.
function Invoke-InImage([string]$Image, [string]$Script) {
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Script -replace "`r", '')))
    Invoke-Native 'docker' @('run', '--rm', '--entrypoint', 'bash', $Image, '-c', "echo $b64 | base64 -d | bash") -Quiet
}

function Invoke-Mongo([string]$Js) {
    $pw = Get-EnvValue 'MONGO_ROOT_PASSWORD'
    if (-not $pw) { $pw = 'example' }
    Invoke-Compose @('exec', '-T', 'mongo', 'mongosh', '--quiet', '-u', 'root', '-p', $pw,
                     '--authenticationDatabase', 'admin', 'robotaxi', '--eval', $Js) -Quiet
}

function Get-UnityClientProcess {
    $buildsRoot = Join-Path $RepoDir 'unity\Builds'
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path.StartsWith($buildsRoot, [StringComparison]::OrdinalIgnoreCase) -and
        $_.Path -notmatch 'UnityCrashHandler'
    }
}

# ---------------------------------------------------------------------------
# Phase 0 - preflight
# ---------------------------------------------------------------------------

function Invoke-Preflight {
    Write-Phase '[preflight] Checking this PC'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $problems = @(); $notes = @()

    $build = [Environment]::OSVersion.Version.Build
    if (-not [Environment]::Is64BitOperatingSystem) { $problems += 'Windows must be 64-bit.' }
    if ($build -lt $MinBuild) { $problems += "Windows build $build is too old; need Windows 10 22H2 ($MinBuild) or Windows 11." }
    else { $notes += "Windows build $build" }

    $cs = Get-CimInstance Win32_ComputerSystem
    $ramGb = [math]::Round($cs.TotalPhysicalMemory / 1GB)
    if ($ramGb -lt 15) { $problems += "$ramGb GB RAM; at least 16 GB is needed." }
    elseif ($ramGb -lt 31) { Write-Warn "$ramGb GB RAM works for one Unity client; 32 GB is recommended." }
    $notes += "$ramGb GB RAM"

    foreach ($drive in @($env:SystemDrive.TrimEnd(':'), (Split-Path -Qualifier $InstallDir).TrimEnd(':')) | Select-Object -Unique) {
        $freeGb = [math]::Round((Get-PSDrive $drive).Free / 1GB)
        if ($freeGb -lt 100) { $problems += "Drive ${drive}: has $freeGb GB free; at least 100 GB is needed (150 GB recommended)." }
        elseif ($freeGb -lt 150) { Write-Warn "Drive ${drive}: has $freeGb GB free; 150 GB is recommended." }
        $notes += "${drive}: $freeGb GB free"
    }

    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    if (-not ($cs.HypervisorPresent -or $cpu.VirtualizationFirmwareEnabled)) {
        $problems += 'CPU virtualization is off. Enable Intel VT-x / AMD SVM in the BIOS/UEFI settings, then re-run.'
    }

    $smi = Invoke-Native 'nvidia-smi' @('--query-gpu=name,driver_version', '--format=csv,noheader') -Quiet
    if ($smi.Code -ne 0) {
        $problems += 'No NVIDIA driver found (nvidia-smi failed). Install the latest Game Ready or Studio driver from https://www.nvidia.com/Download/index.aspx and re-run.'
    } else {
        $gpu = ($smi.Out -split "`n")[0]
        $driver = ($gpu -split ',')[-1].Trim()
        if ([int]($driver -split '\.')[0] -lt $MinDriverMajor) {
            $problems += "NVIDIA driver $driver is too old for WSL 2 GPU support; need R$MinDriverMajor or newer from https://www.nvidia.com/Download/index.aspx."
        }
        $notes += "GPU $gpu"
    }

    if (-not $SkipPrereqs -and -not (Get-Command winget -ErrorAction SilentlyContinue)) {
        $problems += 'winget is missing. Install "App Installer" from the Microsoft Store, then re-run.'
    }

    foreach ($url in @('https://github.com', 'https://pypi.org/simple/pip/', 'https://registry-1.docker.io/v2/')) {
        try { Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null }
        catch {
            if (-not ($_.Exception.PSObject.Properties['Response'] -and $_.Exception.Response)) { $problems += "Cannot reach $url ($($_.Exception.Message))." }
        }
    }

    if ($problems) {
        $problems | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
        Add-Result 'preflight' 'failed' ($problems -join ' ') $sw.Elapsed.TotalSeconds
        throw [PhaseFailure]::new('preflight checks failed')
    }
    Add-Result 'preflight' 'passed' ($notes -join '; ') $sw.Elapsed.TotalSeconds
}

# ---------------------------------------------------------------------------
# Phases 1-3 - prerequisites
# ---------------------------------------------------------------------------

function Test-WslReady {
    (Test-Native 'wsl.exe' @('--version')) -and (Test-Native 'wsl.exe' @('--status'))
}

function Invoke-WslPhase {
    Invoke-Phase 'wsl' 'WSL 2' -Detect {
        if (Test-WslReady) {
            $v = ((Invoke-Native 'wsl.exe' @('--version') -Quiet).Out -replace "`0", '' -split "`n")[0]
            return "wsl --version and --status succeed ($($v.Trim()))"
        }
    } -Install {
        $code = Invoke-Elevated 'wsl.exe' '--install --no-distribution'
        if ($code -ne 0) { Stop-Phase "wsl --install exited with $code" }
        $script:RebootNeeded = $true
        "wsl --install --no-distribution exited 0; reboot required"
    }
}

function Invoke-GitPhase {
    Invoke-Phase 'git' 'Git' -Detect {
        Update-SessionPath
        $r = Invoke-Native 'git' @('--version') -Quiet
        if ($r.Code -eq 0) { return $r.Out.Trim() }
    } -Install {
        $r = Invoke-Native 'winget' @('install', '--id', 'Git.Git', '--exact', '--silent', '--source', 'winget',
                                      '--accept-package-agreements', '--accept-source-agreements')
        Update-SessionPath
        $v = Invoke-Native 'git' @('--version') -Quiet
        if ($v.Code -ne 0) { Stop-Phase "Git install did not produce a working git (winget exit $($r.Code))" }
        "winget exit $($r.Code); $($v.Out.Trim())"
    }
}

function Start-DockerEngine {
    if (Test-Native 'docker' @('info')) { return 0 }
    if (-not (Test-Path $DockerExe)) { Stop-Phase "Docker Desktop is not installed at $DockerExe" }
    Write-Info 'Starting Docker Desktop (the first start can take a few minutes) ...'
    Start-Process -FilePath $DockerExe | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $ok = Wait-Until { Test-Native 'docker' @('info') } -TimeoutSec 300 -IntervalSec 10
    if (-not $ok) {
        $groups = (Invoke-Native 'whoami' @('/groups') -Quiet).Out
        $groupExists = Test-Native 'net' @('localgroup', 'docker-users')
        if ($groupExists -and $groups -notmatch 'docker-users') {
            $script:SignOutNeeded = $true
            Stop-Phase 'Docker Desktop added your account to the docker-users group. Sign out of Windows and back in, then run this script again.'
        }
        Stop-Phase 'Docker Desktop did not start within 5 minutes. Open it from the Start menu, finish any prompts, then re-run.'
    }
    [int]$sw.Elapsed.TotalSeconds
}

function Test-DockerGpu {
    $r = Invoke-Native 'docker' @('run', '--rm', '--gpus', 'all', $CudaTestImage, 'nvidia-smi', '-L') -Quiet
    if ($r.Code -ne 0 -or $r.Out -notmatch 'GPU 0') {
        Stop-Phase "Docker cannot see the GPU (docker run --gpus all ... nvidia-smi failed). Update the NVIDIA driver and run 'wsl --update'. Output: $($r.Out)"
    }
    ($r.Out -split "`n" | Where-Object { $_ -match 'GPU 0' } | Select-Object -First 1).Trim()
}

function Invoke-DockerPhase {
    Invoke-Phase 'docker' 'Docker Desktop' -Detect {
        if (-not (Test-Path $DockerExe)) { return $null }
        if ($DryRun) { return "Docker Desktop installed at $DockerExe" }
        Update-SessionPath
        $waited = Start-DockerEngine
        $gpu = Test-DockerGpu
        $ver = (Invoke-Native 'docker' @('version', '--format', '{{.Server.Version}}') -Quiet).Out.Trim()
        "Docker Desktop installed; engine $ver up (waited ${waited}s); container sees $gpu"
    } -Install {
        if (-not $AcceptDockerLicense) {
            Write-Host ''
            Write-Host '    Docker Desktop is free for personal use, education, non-commercial open source and'
            Write-Host '    small businesses (under 250 employees AND under USD 10M revenue). Larger organisations'
            Write-Host '    need a paid subscription. Terms: https://www.docker.com/legal/docker-subscription-service-agreement/'
            $answer = Read-Host '    Accept the Docker Subscription Service Agreement and install Docker Desktop? [y/N]'
            if ($answer -notmatch '^(y|yes)$') { Stop-Phase 'Docker Desktop licence not accepted.' }
        }
        $r = Invoke-Native 'winget' @('install', '--id', 'Docker.DockerDesktop', '--exact', '--source', 'winget',
                                      '--accept-package-agreements', '--accept-source-agreements',
                                      '--override', 'install --quiet --accept-license --backend=wsl-2')
        if (-not (Test-Path $DockerExe)) { Stop-Phase "Docker Desktop install failed (winget exit $($r.Code))" }
        Update-SessionPath
        $waited = Start-DockerEngine
        $gpu = Test-DockerGpu
        "winget exit $($r.Code); engine up after ${waited}s; container sees $gpu"
    }
}

# ---------------------------------------------------------------------------
# Phase 4 - workspace
# ---------------------------------------------------------------------------

function New-RandomSecret([int]$Length = 32) {
    $chars = [char[]]'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
    $bytes = New-Object byte[] $Length
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
}

function Invoke-WorkspacePhase {
    Invoke-Phase 'workspace' "Repo and data folders in $InstallDir" -Detect {
        $haveRepo = Test-Path (Join-Path $RepoDir '.git')
        $haveEnv  = Test-Path (Join-Path $RepoDir '.env')
        $haveData = @('saved_models', 'mongodb', 'tfrecords' | Where-Object { -not (Test-Path (Join-Path $InstallDir $_)) }).Count -eq 0
        if ($haveRepo -and $haveEnv -and $haveData) {
            $sha = (Invoke-Native 'git' @('-C', $RepoDir, 'rev-parse', '--short', 'HEAD') -Quiet).Out.Trim()
            $br  = (Invoke-Native 'git' @('-C', $RepoDir, 'rev-parse', '--abbrev-ref', 'HEAD') -Quiet).Out.Trim()
            return "repo at $RepoDir ($br@$sha), .env and data folders exist"
        }
    } -Install {
        $did = @()
        if (-not (Test-Path (Join-Path $RepoDir '.git'))) {
            New-Item -ItemType Directory -Force $InstallDir | Out-Null
            $r = Invoke-Native 'git' @('clone', '--branch', $Branch, $RepoUrl, $RepoDir)
            if ($r.Code -ne 0) { Stop-Phase "git clone $RepoUrl failed" }
            $sha = (Invoke-Native 'git' @('-C', $RepoDir, 'rev-parse', '--short', 'HEAD') -Quiet).Out.Trim()
            $did += "cloned $Branch@$sha"
        }
        foreach ($d in 'saved_models', 'mongodb', 'tfrecords') {
            $p = Join-Path $InstallDir $d
            if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force $p | Out-Null; $did += "created $d" }
        }
        $envFile = Join-Path $RepoDir '.env'
        if (-not (Test-Path $envFile)) {
            $lines = @(Get-Content (Join-Path $RepoDir '.env.example'))
            $lines += ''
            $lines += '# ---- Written by scripts/install/Install.ps1 ----'
            $lines += "MONGO_ROOT_PASSWORD=$(New-RandomSecret)"
            if ($ProjectName) { $lines += "COMPOSE_PROJECT_NAME=$ProjectName" }
            if ($ImagePrefix) { $lines += "IMAGE_PREFIX=$ImagePrefix" }
            [IO.File]::WriteAllLines($envFile, $lines)
            $did += 'wrote .env with a random Mongo password'
        }
        $did -join '; '
    }
}

# ---------------------------------------------------------------------------
# Phase 5 - images
# ---------------------------------------------------------------------------

$SimCheck = @'
set -eo pipefail
kb=$(du -s /usr/local/lib/python3.8/dist-packages/tensorflow | cut -f1)
[ "$kb" -gt 900000 ] || { echo "tensorflow folder is only ${kb} KB - corrupted layer"; exit 1; }
python3 -c 'import tensorflow, tf_agents, reverb, pymongo; from tensorboard_plugin_profile import profile_plugin; print("tf", tensorflow.__version__)' 2>&1 | tail -1
source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash
python3 -c 'from roboracer.msg import CarSceneData, SimCommand; print("roboracer msgs ok")'
'@

$RosCheck = @'
set -eo pipefail
source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash
python3 -c 'from roboracer.msg import CarSceneData, SimCommand, SimStatus, Camera; from ros_tcp_endpoint import TcpServer; from virtual_endpoint import VirtualNode; print("unity_node and remote_node imports ok")'
'@

function Test-Images {
    $images = Get-ComposeImages
    $evidence = @()
    foreach ($svc in $BuildServices) {
        $img = $images[$svc]
        if (-not (Test-Native 'docker' @('image', 'inspect', $img))) { return @{ ok = $false; why = "$img missing" } }
        $check = if ($svc -eq 'sim-controller') { $SimCheck } else { $RosCheck }
        $r = Invoke-InImage $img $check
        if ($r.Code -ne 0) { return @{ ok = $false; why = "$img failed verification: $(($r.Out -split "`n" | Select-Object -Last 3) -join ' | ')" } }
        $evidence += "$img ok ($((($r.Out -split "`n") | Where-Object { $_ } | Select-Object -Last 2) -join ', '))"
    }
    @{ ok = $true; why = ($evidence -join '; ') }
}

function Invoke-ImagesPhase {
    Invoke-Phase 'images' 'Docker images (ros-server, sim-controller)' -Detect {
        if ($Rebuild) { return $null }
        $t = Test-Images
        if ($t.ok) { return $t.why }
        Write-Info $t.why
    } -Install {
        $images = Get-ComposeImages
        $buildLog = Join-Path $StateDir 'build.log'
        foreach ($svc in $BuildServices) {
            $built = $false
            for ($i = 1; $i -le $BuildRetries; $i++) {
                Write-Info "Building $svc ($($images[$svc])), attempt $i of $BuildRetries. This takes 20-60 minutes the first time; log: $buildLog"
                $buildArgs = @('build', '--progress=plain')
                if ($Rebuild -and $i -eq 1) { $buildArgs += '--no-cache' }
                $r = Invoke-Compose ($buildArgs + @($svc)) -Quiet
                Add-Content -Path $buildLog -Value $r.Out
                if ($r.Code -eq 0) { $built = $true; break }
                $err = ($r.Out -split "`n" | Where-Object { $_ -match 'ERROR' } | Select-Object -Last 2) -join ' | '
                Write-Warn "attempt $i failed: $err"
            }
            if (-not $built) { Stop-Phase "building $svc failed $BuildRetries times; see $buildLog" }
            $script:BuildAttempts[$svc] = $i
        }
        $t = Test-Images
        if (-not $t.ok) { Stop-Phase $t.why }
        $attempts = ($script:BuildAttempts.GetEnumerator() | ForEach-Object { "$($_.Key) in $($_.Value) attempt(s)" }) -join ', '
        "built $attempts; $($t.why)"
    }
}

# ---------------------------------------------------------------------------
# Phase 6 - Unity client
# ---------------------------------------------------------------------------

function Invoke-UnityPhase {
    $latest = Join-Path $RepoDir 'unity\Builds\latest'
    $marker = Join-Path $latest '.roboracer-client.json'
    Invoke-Phase 'unity' 'Unity client' -Detect {
        if ((Test-Path $marker) -and (Get-ChildItem $latest -Filter '*.exe' -File | Where-Object { $_.Name -notmatch 'UnityCrashHandler' })) {
            $m = Get-Content $marker -Raw | ConvertFrom-Json
            return "client from $($m.source) (sha256 $($m.sha256.Substring(0,12))...) in unity\Builds\latest"
        }
    } -Install {
        $source = $UnityZip
        if (-not $source) { $source = "https://github.com/$GithubRepo/releases/download/$UnityReleaseTag/$UnityAssetName" }
        $zip = $source; $expected = $null
        if ($source -match '^https?://') {
            $zip = Join-Path $StateDir $UnityAssetName
            Write-Info "Downloading $source"
            Invoke-WebRequest -Uri $source -OutFile $zip -UseBasicParsing
            try { $expected = ((Invoke-WebRequest -Uri "$source.sha256" -UseBasicParsing).Content -split '\s+')[0] } catch { $expected = $null }
        } elseif (Test-Path "$source.sha256") {
            $expected = ((Get-Content "$source.sha256" -Raw) -split '\s+')[0]
        }
        if (-not (Test-Path $zip)) { Stop-Phase "Unity client zip not found: $zip" }
        $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($expected) {
            if ($actual -ne $expected.ToLower()) { Stop-Phase "checksum mismatch for $zip (expected $expected, got $actual)" }
            $check = 'checksum verified'
        } else {
            Write-Warn 'No .sha256 found next to the zip; skipping checksum verification.'
            $check = 'no checksum available'
        }
        $tmp = Join-Path $StateDir 'unity-unpack'
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $tmp
        $root = $tmp
        $top = @(Get-ChildItem $tmp)
        if ($top.Count -eq 1 -and $top[0].PSIsContainer) { $root = $top[0].FullName }
        $exes = @(Get-ChildItem $root -Filter '*.exe' -File | Where-Object { $_.Name -notmatch 'UnityCrashHandler' })
        if ($exes.Count -ne 1) { Stop-Phase "expected one game .exe in the zip, found $($exes.Count)" }
        if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
        New-Item -ItemType Directory -Force (Split-Path $latest) | Out-Null
        Move-Item -LiteralPath $root -Destination $latest
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        [pscustomobject]@{ source = $source; sha256 = $actual; installedAt = (Get-Date).ToString('s') } |
            ConvertTo-Json | Set-Content -Path $marker -Encoding UTF8
        "unpacked $($exes[0].Name) from $source ($check)"
    }
}

# ---------------------------------------------------------------------------
# Phase 7 - start
# ---------------------------------------------------------------------------

function Get-UnityRosConnection {
    foreach ($p in @(Get-UnityClientProcess)) {
        $c = Get-NetTCPConnection -OwningProcess $p.Id -RemotePort 10000 -State Established -ErrorAction SilentlyContinue
        if ($c) { return "Unity pid $($p.Id) connected to 127.0.0.1:10000" }
    }
    $null
}

function Invoke-StartPhase {
    Invoke-Phase 'start' 'Start the stack and one Unity client' -NoExpectClean -Detect {
        if (-not (Test-Path $RepoDir)) { return $null }
        $running = @((Invoke-Compose @('ps', '--status', 'running', '--services') -Quiet).Out -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        $missing = @($StartServices | Where-Object { $running -notcontains $_ })
        $conn = Get-UnityRosConnection
        if ($missing.Count -eq 0 -and $conn) { return "services running: $($StartServices -join ', '); $conn" }
    } -Install {
        $r = Invoke-Compose (@('up', '-d', '--no-build') + $StartServices)
        if ($r.Code -ne 0) { Stop-Phase 'docker compose up failed' }
        if (-not (Wait-Until { Test-TcpPort 10000 } -TimeoutSec 120 -What 'ros-server on port 10000')) { Stop-Phase 'ros-server did not open port 10000 within 2 minutes' }
        if (-not (Wait-Until { Test-TcpPort 6006 } -TimeoutSec 180 -What 'TensorBoard on port 6006')) { Stop-Phase 'TensorBoard did not open port 6006 within 3 minutes' }
        Start-Sleep -Seconds 8
        if (-not (Get-UnityClientProcess)) {
            $wrapper = Join-Path $RepoDir 'scripts\RunClientWrapper.ps1'
            Start-Process -FilePath 'powershell.exe' -WindowStyle Minimized -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$wrapper`"", '-Index', '0', '-GymPollSeconds', '0')
        }
        if (-not (Wait-Until { [bool](Get-UnityRosConnection) } -TimeoutSec 180 -What 'the Unity client to connect to ros-server')) {
            Stop-Phase 'the Unity client did not connect to ros-server within 3 minutes; check unity\Builds\latest\Player.log'
        }
        $dash = if (Wait-Until { Test-TcpPort 80 } -TimeoutSec 300 -IntervalSec 10 -What 'the dashboard (first start runs npm install)') { 'dashboard up' } else { 'dashboard still starting' }
        "services started; $(Get-UnityRosConnection); $dash"
    }
}

# ---------------------------------------------------------------------------
# Phase 8 - first TRAIN job
# ---------------------------------------------------------------------------

function Get-FirstJobStatus {
    $r = Invoke-Mongo "const j = db.jobs.findOne({installer_first_job: true}, {status: 1}); print(j ? j.status : 'NONE')"
    if ($r.Code -ne 0) { return 'ERROR' }
    ($r.Out -split "`n" | Where-Object { $_ } | Select-Object -Last 1).Trim()
}

function Get-TrainLines {
    $out = Join-Path $RepoDir 'rl_agent\robotaxi.out'
    if (-not (Test-Path $out)) { return @() }
    @(Select-String -Path $out -Pattern 'TRAIN end:\s+iter=' | ForEach-Object { $_.Line.Trim() })
}

function Invoke-FirstJobPhase {
    Invoke-Phase 'firstjob' "First TRAIN job ($FirstJobIterations iterations)" -Detect {
        if (-not (Test-Path $RepoDir)) { return $null }
        $s = Get-FirstJobStatus
        if ($s -in 'IN_PROGRESS', 'DONE') { return "installer's first job already exists (status $s)" }
    } -Install {
        if ((Get-FirstJobStatus) -eq 'NONE') {
            $js = "db.jobs.insertOne({job_type: 'TRAIN', model_type: 'SacAgent', robot_type: 'robotaxi', " +
                  "num_iterations: $FirstJobIterations, status: 'NOT_STARTED', create_date: new Date(), " +
                  "demo_job_id: '', pass_through_actions: false, nn_size_x: '', nn_size_y: '', seed: 0, " +
                  "percent_complete: 0, installer_first_job: true}).insertedId.toString()"
            $r = Invoke-Mongo $js
            if ($r.Code -ne 0) { Stop-Phase "could not insert the job: $($r.Out)" }
            Write-Info "Queued job $(($r.Out -split "`n" | Select-Object -Last 1).Trim())"
        }
        if (-not (Wait-Until { (Get-FirstJobStatus) -in 'IN_PROGRESS', 'DONE' } -TimeoutSec 600 -IntervalSec 10 -What 'the trainer to pick up the job')) {
            Stop-Phase "the trainer did not start the job within 10 minutes (status $(Get-FirstJobStatus)); see rl_agent\robotaxi.out"
        }
        $ok = Wait-Until { (Get-TrainLines).Count -ge 2 } -TimeoutSec ($FirstJobWaitMinutes * 60) -IntervalSec 20 -What 'training steps'
        if (-not $ok) { Stop-Phase "no training steps after $FirstJobWaitMinutes minutes; see rl_agent\robotaxi.out" }
        $lines = Get-TrainLines
        "job $(Get-FirstJobStatus); $($lines.Count) training iterations so far; last: $($lines[-1])"
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

$script:RebootNeeded = $false
$script:SignOutNeeded = $false
$script:BuildAttempts = @{}
$exitCode = 0

Write-Host "rl-roboracer installer  (log: $LogPath)" -ForegroundColor White
Write-Host "Install folder: $InstallDir"
if ($DryRun) { Write-Host 'Dry run: nothing will be changed.' -ForegroundColor Yellow }
Remove-ItemProperty -Path $RunOnceKey -Name $RunOnceName -ErrorAction SilentlyContinue

try {
    Invoke-Preflight
    if (-not $SkipPrereqs) {
        Invoke-WslPhase
        if ($script:RebootNeeded) {
            Register-Resume
            Write-Host ''
            Write-Host 'WSL is installed but Windows must restart before Docker Desktop can use it.' -ForegroundColor Yellow
            Write-Host 'This installer will continue automatically when you sign in again.'
            $exitCode = 2
            $answer = Read-Host 'Restart now? [Y/n]'
            if ($answer -notmatch '^(n|no)$') { Stop-Transcript | Out-Null; Restart-Computer -Force }
            throw [PhaseFailure]::new('restart Windows to continue')
        }
        Invoke-GitPhase
        Invoke-DockerPhase
    } else {
        Write-Phase '[prereqs] Skipped (-SkipPrereqs)'
        Add-Result 'prereqs' 'skipped' '-SkipPrereqs' 0
        Update-SessionPath
        if (-not $DryRun) { [void](Start-DockerEngine) }
    }
    Invoke-WorkspacePhase
    if (-not $DryRun -or (Test-Path $RepoDir)) {
        Invoke-ImagesPhase
        Invoke-UnityPhase
        if (-not $DryRun) {
            Invoke-StartPhase
            if (-not $NoFirstJob) { Invoke-FirstJobPhase }
        }
    }
} catch [PhaseFailure] {
    Write-Host ''
    if ($exitCode -eq 2) {
        Write-Host "Install paused: $($_.Exception.Message)" -ForegroundColor Yellow
    } else {
        Write-Host "Install stopped: $($_.Exception.Message)" -ForegroundColor Red
        $exitCode = if ($script:SignOutNeeded) { 3 } else { 1 }
    }
} catch {
    Write-Host ''
    Write-Host "Unexpected error: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    Add-Result 'installer' 'failed' "$_" 0
    $exitCode = 1
} finally {
    Write-Host ''
    Write-Host 'Summary' -ForegroundColor White
    $script:Report | Format-Table phase, outcome, seconds -AutoSize | Out-String | Write-Host
    Write-Host "Report: $ReportPath"
    if ($exitCode -eq 0 -and -not $DryRun) {
        Write-Host ''
        Write-Host 'Done. Training is running:' -ForegroundColor Green
        Write-Host '  Dashboard      http://localhost'
        Write-Host '  TensorBoard    http://localhost:6006'
        Write-Host '  Mongo Express  http://127.0.0.1:8081'
        Write-Host "  Trainer log    $RepoDir\rl_agent\robotaxi.out"
    }
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
exit $exitCode
