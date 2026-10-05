<#
.SYNOPSIS
    Install rl-roboracer on a Windows PC with an NVIDIA GPU, from nothing to a
    first TRAIN job producing training steps.

.DESCRIPTION
    Runs ten phases in order. Each phase first checks whether its work is
    already done, so the script can be re-run at any time and resumes where
    it stopped (including after the reboot WSL needs).

      0 preflight   Windows build, RAM, disk, virtualization, NVIDIA driver,
                    winget, network
      1 wsl         WSL 2 (wsl --install --no-distribution; needs a reboot)
      2 git         Git for Windows (winget)
      3 docker      Docker Desktop (winget), engine running, GPU visible
      4 workspace   Repo clone, sibling data folders, .env
      5 images      ros-server + sim-controller images built and verified
      6 unity       The starter gyms unpacked into <InstallDir>\UnityBinary\,
                    the default one (v13) also into unity\Builds\latest\
        demos       Expert-demo recording the donut TRAIN jobs start from
      7 start       Stack up, one Unity client connected to ros-server
        seed        Starter gyms, reward design and experiment designs in
                    MongoDB (seed\seed.json; existing documents are kept)
      8 firstjob    A short TRAIN job on the default gym, straight into SAC
                    training (no BC pretrain, no first eval), producing
                    training steps and TensorBoard scalars

    Every phase records what it found and what it did in
    install-report.json, with outcome installed / already-present /
    skipped / passed / failed. -ExpectClean turns "already-present" into a
    failure, which is how a test on a supposedly clean machine catches an
    install step that silently did nothing.

    When a step fails, its output is matched against a catalog of known
    failures (see "Failure catalog" below). A match prints what went wrong
    and what to do, may apply a fix (retry, restart Docker Desktop, clean
    rebuild, wsl --update), and is recorded under "diagnoses" in the report.

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

.PARAMETER AssetDir
    Local folder holding the release assets (the gym zips and
    roboracer-demos-donut.zip, each optionally with a "<zip>.sha256").
    Without it the assets are downloaded from the GitHub Release
    -UnityReleaseTag.

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

.PARAMETER AutoFix
    Apply catalogued fixes that would otherwise ask first (restart Docker
    Desktop, clear the build cache, run wsl --update). Fixes that affect
    other software, such as stopping another program on a needed port, still
    ask.

.PARAMETER SelfTest
    Check the failure catalog against its own sample log lines, then exit.
    Changes nothing.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'rl-roboracer'),
    [string]$RepoUrl = 'https://github.com/frenb/rl-roboracer.git',
    [string]$Branch = 'main',
    [string]$AssetDir = '',
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
    [int]$BuildRetries = 6,
    [switch]$AutoFix,
    [switch]$SelfTest
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Windows PowerShell 5.1 still offers TLS 1.0 by default; GitHub refuses it.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$ScriptParams = $PSBoundParameters

$GithubRepo     = 'frenb/rl-roboracer'
$DemoAssetName  = 'roboracer-demos-donut.zip'
# Ids and names must match the gyms in seed\seed.json. Each ships as
# roboracer-gym-<Folder>.zip; the first one is the default client.
$Gyms = @(
    @{ Id = '6aafa685c2736b990584403f'; Name = 'wCourseJetRacer2026.09.20-v13'; Folder = 'wCourseJetRacer2026.09.20-v13' },
    @{ Id = '6abc6906a249d535d4d976e3'; Name = 'FlyBrain-wCourseJetRacer2026.09.28-v49'; Folder = 'wCourseJetRacer2026.09.28-v49' }
)
$GymExeName     = 'robotaxi gym level 1.exe'
# Must match COURSE_DEFAULT_DEMO_JOB_IDS['donut'] in rl_agent/robotaxi.py.
$DemoJobId      = '64168c1b58d4d8ccdb76e721'
$MinBuild       = 19045
$MinDriverMajor = 495
$CudaTestImage  = 'nvidia/cuda:11.0.3-base-ubuntu20.04'
$BuildServices  = @('ros-server', 'sim-controller', 'fly-brain')
$StartServices  = @('ros-server', 'mongo', 'mongo-express', 'sim-controller', 'dashboard', 'fly-brain')
$DockerExe      = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'

$StateDir   = Join-Path $env:LOCALAPPDATA 'rl-roboracer-install'
$RunStamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$ReportPath = Join-Path $StateDir 'install-report.json'
$RunReportPath = Join-Path $StateDir "install-report-$RunStamp.json"
$RepoDir    = Join-Path $InstallDir 'rl-roboracer'
$UnityBinaryDir = Join-Path $InstallDir 'UnityBinary'
$DefaultGymExe  = Join-Path $UnityBinaryDir "$($Gyms[0].Folder)\$GymExeName"
$RunOnceKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
$RunOnceName = 'rl-roboracer-install'

$script:Report = New-Object System.Collections.ArrayList
$script:Diagnoses = New-Object System.Collections.ArrayList

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
    $json = [pscustomobject]@{ results = @($script:Report); diagnoses = @($script:Diagnoses) } | ConvertTo-Json -Depth 5
    Set-Content -Path $ReportPath -Value $json -Encoding UTF8
    Set-Content -Path $RunReportPath -Value $json -Encoding UTF8
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

function Invoke-Mongo([string]$Js, [string]$File = '') {
    $pw = Get-EnvValue 'MONGO_ROOT_PASSWORD'
    if (-not $pw) { $pw = 'example' }
    $script = if ($File) { @('--file', $File) } else { @('--eval', $Js) }
    Invoke-Compose (@('exec', '-T', 'mongo', 'mongosh', '--quiet', '-u', 'root', '-p', $pw,
                      '--authenticationDatabase', 'admin', 'robotaxi') + $script) -Quiet
}

function Get-UnityClientProcess {
    $buildsRoot = Join-Path $RepoDir 'unity\Builds'
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path.StartsWith($buildsRoot, [StringComparison]::OrdinalIgnoreCase) -and
        $_.Path -notmatch 'UnityCrashHandler'
    }
}

# The wrapper relaunches a client it sees exit, so it has to go first.
function Stop-UnityClients {
    $wrappers = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$RepoDir\scripts\RunClientWrapper.ps1*" })
    $wrappers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $clients = @(Get-UnityClientProcess)
    $clients | Stop-Process -Force -ErrorAction SilentlyContinue
    if ($clients) { Start-Sleep -Seconds 3 }
    $wrappers.Count + $clients.Count
}

# ---------------------------------------------------------------------------
# Failure catalog
#
# Known failures: where they show up (Phases), how to recognise them
# (Pattern), what they mean, what the user should do, and which fix the
# installer may apply. Samples are real log lines; -SelfTest checks every
# sample is recognised as its own entry. Order matters: the first match wins.
#
# Fix kinds: retry (automatic), restart-docker / rebuild-clean / wsl-update
# (ask first, or -AutoFix), port-owner (always asks), signout, none.
# ---------------------------------------------------------------------------

$FailureCatalog = @(
    @{ Id = 'download-truncated'; Phases = @('build'); Fix = 'retry'
       Pattern = 'DO NOT MATCH THE HASHES|BadZipFile: Bad CRC-32'
       Diagnosis = 'A download was cut short or corrupted. Docker Desktop''s network intermittently truncates large downloads.'
       Advice = 'Retrying; finished build steps are cached, so the retry resumes where it failed.'
       Samples = @('#20 5.798 ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the package versions, please update the hashes.',
                   "#26 2.736 zipfile.BadZipFile: Bad CRC-32 for file 'numpy/core/tests/data/umath-validation-set-log1p.csv'") }

    @{ Id = 'docker-vm-crash'; Phases = @('build'); Fix = 'restart-docker'
       Pattern = 'failed to receive status: rpc error|error reading from server: EOF|session healthcheck failed|desc = connection error'
       Diagnosis = 'Docker Desktop''s engine crashed or restarted during the build, usually while unpacking a large image.'
       Advice = 'Restart Docker Desktop, then run the installer again.'
       Samples = @('ERROR: failed to build: failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF',
                   'session healthcheck failed fatally: Unavailable: connection error: desc = "transport: Error while dialing: only one connection allowed"') }

    @{ Id = 'disk-full'; Phases = @('*'); Fix = 'none'
       Pattern = 'no space left on device|not enough space on the disk'
       Diagnosis = 'The disk Docker uses is full.'
       Advice = 'Free space, then re-run. "docker system df" shows what Docker uses; "docker builder prune -af" and "docker image prune -a" reclaim build cache and unused images.'
       Samples = @('failed to copy files: write /var/lib/docker/tmp/buildkit-mount/x.whl: no space left on device') }

    @{ Id = 'network-flaky'; Phases = @('build', 'git', 'download'); Fix = 'retry'
       Pattern = 'Temporary failure in name resolution|Could not resolve host|Read timed out|ReadTimeoutError|Connection reset by peer|TLS handshake timeout|i/o timeout|Connection timed out|unable to access ''https'
       Diagnosis = 'A network request failed or timed out.'
       Advice = 'Retrying. If it keeps failing, check the internet connection, VPN or proxy.'
       Samples = @('pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host=''files.pythonhosted.org'', port=443): Read timed out.',
                   'fatal: unable to access ''https://github.com/frenb/rl-roboracer.git/'': Could not resolve host: github.com') }

    @{ Id = 'illegal-instruction'; Phases = @('verify', 'trainer'); Fix = 'none'
       Pattern = 'Illegal instruction'
       Diagnosis = 'This CPU, or the emulator running the container, lacks instructions (AVX) that TensorFlow''s prebuilt packages need.'
       Advice = 'The installer cannot fix this; this machine cannot run the trainer image as built.'
       Samples = @('bash: line 3:    12 Illegal instruction     (core dumped) python3 -c ''import tensorflow''') }

    @{ Id = 'corrupted-layers'; Phases = @('verify'); Fix = 'rebuild-clean'
       Pattern = 'cannot import name|ImportError|SyntaxError|No module named|is only \d+ KB - corrupted layer'
       Diagnosis = 'The image built, but files inside it are damaged or missing. A crashed earlier build most likely left half-written layers in Docker''s build cache, which this build reused.'
       Advice = 'Clear Docker''s build cache and rebuild this image without cache.'
       Samples = @('ImportError: cannot import name ''LazyLoader'' from ''tensorflow.python.util.lazy_loader'' (/usr/local/lib/python3.8/dist-packages/tensorflow/python/util/lazy_loader.py)',
                   'tensorflow folder is only 122900 KB - corrupted layer') }

    @{ Id = 'port-in-use'; Phases = @('compose-up', 'unity-client'); Fix = 'port-owner'
       Pattern = 'Ports are not available|port is already allocated|bind: (address already in use|Only one usage of each socket address|An attempt was made to access a socket)|SocketException[^\n]*(Address already in use|Only one usage)'
       Diagnosis = 'Another program is already using a port this stack needs.'
       Advice = 'Stop the program using the port, then re-run.'
       Samples = @('Error response from daemon: Ports are not available: exposing port TCP 0.0.0.0:80 -> 0.0.0.0:0: listen tcp 0.0.0.0:80: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.',
                   'Error response from daemon: driver failed programming external connectivity: Bind for 0.0.0.0:6006 failed: port is already allocated',
                   'System.Net.Sockets.SocketException (0x80004005): Only one usage of each socket address (protocol/network address/port) is normally permitted.') }

    @{ Id = 'docker-access-denied'; Phases = @('docker-start'); Fix = 'signout'
       Pattern = 'dockerDesktopLinuxEngine: Access is denied|permission denied while trying to connect to the Docker daemon'
       Diagnosis = 'Your account cannot use Docker yet. Docker Desktop added it to the docker-users group, which only takes effect at the next sign-in.'
       Advice = 'Sign out of Windows and back in, then run the installer again.'
       Samples = @('error during connect: in the default daemon configuration on Windows, the docker client must be run with elevated privileges to connect: open //./pipe/dockerDesktopLinuxEngine: Access is denied.') }

    @{ Id = 'wsl-outdated'; Phases = @('docker-start'); Fix = 'wsl-update'
       Pattern = 'WSL (needs|requires) (updating|an update)|WSL kernel version too low|wsl --update'
       Diagnosis = 'WSL needs an update before Docker Desktop can start.'
       Advice = 'Run "wsl --update" as administrator, then restart Docker Desktop.'
       Samples = @('WSL needs updating. Your version of WSL is too old. Run ''wsl --update'' to update it.') }

    @{ Id = 'docker-not-running'; Phases = @('docker-start'); Fix = 'none'
       Pattern = 'dockerDesktopLinuxEngine: The system cannot find the file specified|Is the docker daemon running|Cannot connect to the Docker daemon'
       Diagnosis = 'Docker Desktop''s engine is not running and did not start.'
       Advice = 'Open Docker Desktop from the Start menu, answer any prompts it shows, wait until it says "Engine running", then re-run.'
       Samples = @('error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.47/info": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.') }

    @{ Id = 'gpu-not-in-docker'; Phases = @('docker-gpu'); Fix = 'wsl-update'
       Pattern = 'could not select device driver[^\n]*gpu|nvidia-container-cli|libnvidia-ml\.so|Failed to initialize NVML|no CUDA-capable device'
       Diagnosis = 'Docker cannot reach the NVIDIA GPU.'
       Advice = 'Run "wsl --update" and restart Docker Desktop. If that does not help, install the latest NVIDIA driver from https://www.nvidia.com/Download/index.aspx.'
       Samples = @('docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].',
                   'nvidia-container-cli: initialization error: WSL environment detected but no adapters were found: unknown.') }

    @{ Id = 'clone-target-exists'; Phases = @('git'); Fix = 'none'
       Pattern = 'already exists and is not an empty directory'
       Diagnosis = 'The install folder already contains an rl-roboracer folder that is not a git checkout.'
       Advice = 'Move or delete that folder, or pass a different -InstallDir.'
       Samples = @('fatal: destination path ''C:\Users\me\rl-roboracer\rl-roboracer'' already exists and is not an empty directory.') }

    @{ Id = 'repo-files-missing'; Phases = @('container'); Fix = 'recreate'
       Pattern = 'ENOENT[^\n]*package\.json|can''t open file ''[^'']*\.py''|No such file or directory[^\n]*\.py'''
       Diagnosis = 'A container cannot see the repo files it runs from. Usually the install folder was deleted or moved while its containers still existed; Docker restarted them and recreated the missing folders empty.'
       Advice = 'Recreate the containers from the current install folder: docker compose up -d --force-recreate.'
       Samples = @('npm error enoent Could not read package.json: Error: ENOENT: no such file or directory, open ''/dashboard/package.json''',
                   'python: can''t open file ''robotaxi.py'': [Errno 2] No such file or directory') }

    @{ Id = 'mongo-data-permission'; Phases = @('container'); Fix = 'mongo-owner'
       Pattern = '/bitnami/mongodb[^\n]*Permission denied|Permission denied[^\n]*/bitnami/mongodb'
       Diagnosis = 'MongoDB (which runs as user 1001) cannot write to its data folder. Folders that Docker creates itself on a Windows drive come out owned by root and read-only to everyone else.'
       Advice = 'Give the folder to user 1001: docker compose run --rm --no-deps --user root --entrypoint chown mongo -R 1001:1001 /bitnami/mongodb'
       Samples = @('mongo-1  | mkdir: cannot create directory ''/bitnami/mongodb/data'': Permission denied') }

    @{ Id = 'mongo-auth'; Phases = @('trainer', 'container'); Fix = 'none'
       Pattern = 'Authentication failed|AuthenticationFailed|bad auth'
       Diagnosis = 'The trainer cannot log in to MongoDB. The password in .env differs from the one the database was created with, typically because the mongodb folder was reused from an earlier install.'
       Advice = 'Put the original password back in .env as MONGO_ROOT_PASSWORD. Or, to start with an empty database: stop the stack, empty the mongodb folder, and start it again.'
       Samples = @('pymongo.errors.OperationFailure: Authentication failed., full error: {''ok'': 0.0, ''errmsg'': ''Authentication failed.'', ''code'': 18, ''codeName'': ''AuthenticationFailed''}') }

    @{ Id = 'mongo-unreachable'; Phases = @('trainer', 'container'); Fix = 'none'
       Pattern = 'ServerSelectionTimeoutError'
       Diagnosis = 'The trainer cannot reach MongoDB.'
       Advice = 'Check the mongo container with "docker compose ps mongo" and "docker compose logs mongo".'
       Samples = @('pymongo.errors.ServerSelectionTimeoutError: mongo:27017: [Errno -3] Temporary failure in name resolution') }

    @{ Id = 'gpu-oom'; Phases = @('trainer'); Fix = 'none'
       Pattern = 'ResourceExhaustedError|CUDA_ERROR_OUT_OF_MEMORY|OOM when allocating'
       Diagnosis = 'The GPU ran out of memory.'
       Advice = 'Close other programs using the GPU (games, other ML jobs, a second training stack), then re-run.'
       Samples = @('tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[512,512] and type float') }

    @{ Id = 'cuda-init'; Phases = @('trainer'); Fix = 'none'
       Pattern = 'failed call to cuInit|CUDA_ERROR_NO_DEVICE|Could not load dynamic library .libcuda'
       Diagnosis = 'TensorFlow inside the trainer cannot use the GPU.'
       Advice = 'Update the NVIDIA driver, run "wsl --update", restart Docker Desktop, then re-run.'
       Samples = @('E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected') }

    @{ Id = 'demo-data-missing'; Phases = @('trainer'); Fix = 'none'
       Pattern = 'No such file or directory: ''/tfrecords/job_[0-9a-f]+'
       Diagnosis = 'The training job needs expert demonstrations recorded by an earlier DEMO job (a /tfrecords/job_<id> folder), and this install does not have that recording.'
       Advice = 'Put that job''s recording in the tfrecords folder next to the repo, or give the job demo_job_ids of a DEMO job recorded on this install.'
       Samples = @('FileNotFoundError: [Errno 2] No such file or directory: ''/tfrecords/job_64168c1b58d4d8ccdb76e721''') }

    @{ Id = 'python-traceback'; Phases = @('trainer'); Fix = 'none'
       Pattern = 'Traceback \(most recent call last\)'
       Seen = '(?m)^[A-Za-z_][\w.]*(Error|Exception)\b[^\n]*'
       Diagnosis = 'The trainer stopped with a Python error.'
       Advice = 'The error is shown above; the full log is rl_agent\robotaxi.out. Include it if you report the problem.'
       Samples = @("Traceback (most recent call last):`n  File ""robotaxi.py"", line 1, in <module>`nKeyError: 'x'") }
)

function Find-Failure([string]$Phase, [string]$Text) {
    if (-not $Text) { return $null }
    foreach ($e in $FailureCatalog) {
        if (($e.Phases -notcontains $Phase) -and ($e.Phases -notcontains '*')) { continue }
        $m = [regex]::Match($Text, $e.Pattern, 'IgnoreCase')
        if (-not $m.Success) { continue }
        $start = $Text.LastIndexOf("`n", [math]::Max(0, $m.Index - 1)) + 1
        if ($m.Index -eq 0) { $start = 0 }
        $end = $Text.IndexOf("`n", $m.Index)
        if ($end -lt 0) { $end = $Text.Length }
        $line = $Text.Substring($start, $end - $start).Trim()
        if ($e.ContainsKey('Seen')) {
            $seen = [regex]::Matches($Text.Substring($m.Index), $e.Seen)
            if ($seen.Count) { $line = $seen[$seen.Count - 1].Value.Trim() }
        }
        $port = 0
        $pm = [regex]::Match($line, '(?:0\.0\.0\.0|127\.0\.0\.1|\[::\]|localhost):(\d+)')
        if ($pm.Success) { $port = [int]$pm.Groups[1].Value }
        return [pscustomobject]@{ Id = $e.Id; Entry = $e; Line = $line; Port = $port }
    }
    $null
}

function Show-Diagnosis($Hit) {
    $shown = if ($Hit.Line.Length -gt 220) { $Hit.Line.Substring(0, 220) + '...' } else { $Hit.Line }
    Write-Host "    Diagnosis [$($Hit.Id)]: $($Hit.Entry.Diagnosis)" -ForegroundColor Yellow
    Write-Host "      seen: $shown" -ForegroundColor DarkGray
    Write-Host "      what to do: $($Hit.Entry.Advice)" -ForegroundColor Yellow
}

function Add-Diagnosis([string]$Phase, $Hit, [string]$Action) {
    [void]$script:Diagnoses.Add([pscustomobject]@{
        phase = $Phase; id = $Hit.Id; seen = $Hit.Line; action = $Action; at = (Get-Date).ToString('s')
    })
    Save-Report
}

# Diagnoses $Text, records it, and stops the phase with the catalog's advice
# (or the plain message when nothing matches).
function Stop-WithDiagnosis([string]$Phase, [string]$Text, [string]$Message) {
    $hit = Find-Failure $Phase $Text
    if ($hit) {
        Show-Diagnosis $hit
        Add-Diagnosis $Phase $hit 'advised'
        Stop-Phase "$Message [$($hit.Id)] $($hit.Entry.Advice)"
    }
    $tail = (($Text -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 4) -join ' | '
    Stop-Phase "$Message. Last output: $tail"
}

function Confirm-Fix([string]$Question, [switch]$AffectsOtherSoftware) {
    if ($DryRun) { return $false }
    if ($AutoFix -and -not $AffectsOtherSoftware) { Write-Info "$Question -> yes (-AutoFix)"; return $true }
    $answer = Read-Host "    $Question [y/N]"
    $answer -match '^(y|yes)$'
}

function Restart-DockerDesktop {
    Write-Info 'Restarting Docker Desktop ...'
    if (Test-Native 'docker' @('desktop', 'version')) {
        [void](Invoke-Native 'docker' @('desktop', 'restart') -Quiet)
    } else {
        Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 10
        Start-Process -FilePath $DockerExe | Out-Null
    }
    if (-not (Wait-Until { Test-Native 'docker' @('info') } -TimeoutSec 300 -IntervalSec 10 -What 'Docker Desktop to come back')) {
        Stop-Phase 'Docker Desktop did not come back within 5 minutes after the restart'
    }
}

function Invoke-WslUpdate {
    $code = Invoke-Elevated 'wsl.exe' '--update'
    if ($code -ne 0) { Write-Warn "wsl --update exited with $code" }
    Restart-DockerDesktop
}

function Get-PortOwner([int]$Port) {
    $rows = @((Invoke-Native 'docker' @('ps', '--filter', "publish=$Port", '--format', '{{.Names}}|{{.Labels}}') -Quiet).Out -split "`n" | Where-Object { $_.Trim() })
    if ($rows) {
        $name, $labels = $rows[0] -split '\|', 2
        $project = ([regex]::Match($labels, 'com\.docker\.compose\.project=([^,]+)')).Groups[1].Value
        return [pscustomobject]@{ Kind = 'docker'; Name = $name; Project = $project; Text = "Docker container '$name'" + $(if ($project) { " (compose project '$project')" } else { '' }) }
    }
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        if ($conn.OwningProcess -eq 4) {
            return [pscustomobject]@{ Kind = 'system'; Name = 'System'; Project = ''; Text = 'the Windows HTTP service (http.sys), often IIS or "World Wide Web Publishing Service"; "netsh http show servicestate" lists who registered it' }
        }
        $p = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        $pname = if ($p) { $p.ProcessName } else { 'unknown' }
        return [pscustomobject]@{ Kind = 'process'; Name = $pname; Project = ''; Text = "program '$pname' (pid $($conn.OwningProcess))" }
    }
    $null
}

# Returns $true when the port was freed and the caller should retry.
function Resolve-PortConflict([int]$Port) {
    if ($Port -le 0) { return $false }
    $owner = Get-PortOwner $Port
    if (-not $owner) { Write-Info "Port $Port is no longer in use."; return $true }
    Write-Host "      port $Port is used by $($owner.Text)" -ForegroundColor Yellow
    if ($owner.Kind -eq 'docker' -and $owner.Project) {
        if (Confirm-Fix "Stop compose project '$($owner.Project)' so this install can use port ${Port}? (docker compose -p $($owner.Project) stop)" -AffectsOtherSoftware) {
            [void](Invoke-Native 'docker' @('compose', '-p', $owner.Project, 'stop'))
            return $true
        }
    }
    $false
}

function Invoke-SelfTest {
    $failures = 0; $checked = 0
    foreach ($e in $FailureCatalog) {
        foreach ($s in $e.Samples) {
            foreach ($ph in $e.Phases) {
                $phase = if ($ph -eq '*') { 'build' } else { $ph }
                $checked++
                $hit = Find-Failure $phase $s
                $got = if ($hit) { $hit.Id } else { '(none)' }
                if ($got -ne $e.Id) { $failures++; Write-Host "FAIL  $($e.Id) sample in phase '$phase' matched $got" -ForegroundColor Red }
            }
        }
    }
    foreach ($clean in @('#44 naming to docker.io/library/sim_controller:latest done', 'TRAIN end:   iter=3/2000 train_step=3 loss=1.2')) {
        foreach ($ph in 'build', 'verify', 'compose-up', 'docker-start', 'docker-gpu', 'git', 'trainer', 'unity-client', 'container') {
            $checked++
            $hit = Find-Failure $ph $clean
            if ($hit) { $failures++; Write-Host "FAIL  healthy line matched $($hit.Id) in phase '$ph': $clean" -ForegroundColor Red }
        }
    }
    $portSample = ($FailureCatalog | Where-Object { $_.Id -eq 'port-in-use' }).Samples[0]
    $port = (Find-Failure 'compose-up' $portSample).Port
    $checked++
    if ($port -ne 80) { $failures++; Write-Host "FAIL  port extraction gave $port, expected 80" -ForegroundColor Red }
    Write-Host "Failure catalog self-test: $($FailureCatalog.Count) entries, $checked checks, $failures failures."
    $failures
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

    # Any HTTP answer, including Docker Hub's 401, proves the host is reachable.
    # Plain WebRequest keeps those expected errors out of the transcript.
    foreach ($url in @('https://github.com', 'https://pypi.org/simple/pip/', 'https://registry-1.docker.io/v2/')) {
        $req = [Net.WebRequest]::Create($url)
        $req.Method = 'HEAD'; $req.Timeout = 20000
        try { $req.GetResponse().Close() }
        catch {
            $ex = $_.Exception
            if ($ex.InnerException) { $ex = $ex.InnerException }
            if (($ex -is [Net.WebException]) -and $ex.Response) { $ex.Response.Close() }
            else { $problems += "Cannot reach $url ($($ex.Message))." }
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
    # Docker Desktop runs for one signed-in user at a time; a copy left
    # running by another account (e.g. after "Switch user") blocks ours.
    $mySession = [Diagnostics.Process]::GetCurrentProcess().SessionId
    $other = @(Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -ne $mySession })
    if ($other) {
        $owner = $null
        try {
            $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $($other[0].Id)"
            $o = Invoke-CimMethod -InputObject $cim -MethodName GetOwner
            if ($o.ReturnValue -eq 0) { $owner = $o.User }
        } catch { }
        $who = if ($owner) { "the Windows account '$owner'" } else { "another signed-in Windows account (session $($other[0].SessionId))" }
        Stop-Phase "Docker Desktop is already running for $who, and it only runs for one account at a time. Sign in to that account, quit Docker Desktop (tray whale icon > Quit Docker Desktop) and sign out, then run this installer again."
    }
    Write-Info 'Starting Docker Desktop (the first start can take a few minutes) ...'
    Start-Process -FilePath $DockerExe | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $ok = Wait-Until { Test-Native 'docker' @('info') } -TimeoutSec 300 -IntervalSec 10
    if (-not $ok) {
        $info = (Invoke-Native 'docker' @('info') -Quiet).Out
        $groups = (Invoke-Native 'whoami' @('/groups') -Quiet).Out
        $groupExists = Test-Native 'net' @('localgroup', 'docker-users')
        if ($groupExists -and $groups -notmatch 'docker-users') {
            $info += "`nopen //./pipe/dockerDesktopLinuxEngine: Access is denied."
        }
        $hit = Find-Failure 'docker-start' $info
        if ($hit -and $hit.Entry.Fix -eq 'signout') { $script:SignOutNeeded = $true }
        if ($hit -and $hit.Entry.Fix -eq 'wsl-update') {
            Show-Diagnosis $hit
            if (Confirm-Fix 'Run "wsl --update" now (needs administrator rights) and restart Docker Desktop?') {
                Add-Diagnosis 'docker-start' $hit 'wsl --update + restart Docker Desktop'
                Invoke-WslUpdate
                return [int]$sw.Elapsed.TotalSeconds
            }
        }
        Stop-WithDiagnosis 'docker-start' $info 'Docker Desktop did not start within 5 minutes'
    }
    [int]$sw.Elapsed.TotalSeconds
}

function Test-DockerGpu {
    $r = Invoke-Native 'docker' @('run', '--rm', '--gpus', 'all', $CudaTestImage, 'nvidia-smi', '-L') -Quiet
    if ($r.Code -ne 0 -or $r.Out -notmatch 'GPU 0') {
        $hit = Find-Failure 'docker-gpu' $r.Out
        if ($hit -and $hit.Entry.Fix -eq 'wsl-update') {
            Show-Diagnosis $hit
            if (Confirm-Fix 'Run "wsl --update" now (needs administrator rights), restart Docker Desktop and test again?') {
                Add-Diagnosis 'docker-gpu' $hit 'wsl --update + restart Docker Desktop'
                Invoke-WslUpdate
                $r = Invoke-Native 'docker' @('run', '--rm', '--gpus', 'all', $CudaTestImage, 'nvidia-smi', '-L') -Quiet
            }
        }
        if ($r.Code -ne 0 -or $r.Out -notmatch 'GPU 0') {
            Stop-WithDiagnosis 'docker-gpu' $r.Out 'Docker cannot see the GPU'
        }
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
        $haveRepo = Test-Path (Join-Path $RepoDir '.git\HEAD')
        $haveEnv  = Test-Path (Join-Path $RepoDir '.env')
        $haveData = @('saved_models', 'mongodb', 'tfrecords' | Where-Object { -not (Test-Path (Join-Path $InstallDir $_)) }).Count -eq 0
        if ($haveRepo -and $haveEnv -and $haveData) {
            $sha = (Invoke-Native 'git' @('-C', $RepoDir, 'rev-parse', '--short', 'HEAD') -Quiet).Out.Trim()
            $br  = (Invoke-Native 'git' @('-C', $RepoDir, 'rev-parse', '--abbrev-ref', 'HEAD') -Quiet).Out.Trim()
            return "repo at $RepoDir ($br@$sha), .env and data folders exist"
        }
    } -Install {
        $did = @()
        if (-not (Test-Path (Join-Path $RepoDir '.git\HEAD'))) {
            # Docker recreates missing bind-mount sources as empty folders when
            # containers outlive a deleted install, and the trainer then writes
            # its log there; such a leftover blocks the clone.
            if (Test-Path $RepoDir) {
                $content = @(Get-ChildItem $RepoDir -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '\.(out|log)$' })
                if ($content.Count -eq 0) {
                    Remove-Item -LiteralPath $RepoDir -Recurse -Force
                    $did += 'removed a leftover repo folder that held only logs'
                }
            }
            New-Item -ItemType Directory -Force $InstallDir | Out-Null
            $r = Invoke-Native 'git' @('clone', '--branch', $Branch, $RepoUrl, $RepoDir)
            if ($r.Code -ne 0) {
                $hit = Find-Failure 'git' $r.Out
                if ($hit -and $hit.Entry.Fix -eq 'retry') {
                    Show-Diagnosis $hit
                    Add-Diagnosis 'git' $hit 'retry'
                    Start-Sleep -Seconds 15
                    $r = Invoke-Native 'git' @('clone', '--branch', $Branch, $RepoUrl, $RepoDir)
                }
                if ($r.Code -ne 0) { Stop-WithDiagnosis 'git' $r.Out "git clone $RepoUrl failed" }
            }
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
            # Mongo only applies the password when it first creates the
            # database, so a reused mongodb folder keeps its old one.
            $mongoInUse = @(Get-ChildItem (Join-Path $InstallDir 'mongodb') -Force -ErrorAction SilentlyContinue).Count -gt 0
            if ($mongoInUse) {
                Write-Warn 'The mongodb folder already holds a database. Not generating a new password: set MONGO_ROOT_PASSWORD in .env to the password that database was created with (the default is "example").'
                $did += 'wrote .env (kept the default Mongo password: existing database found)'
            } else {
                $lines += "MONGO_ROOT_PASSWORD=$(New-RandomSecret)"
                $did += 'wrote .env with a random Mongo password'
            }
            if ($ProjectName) { $lines += "COMPOSE_PROJECT_NAME=$ProjectName" }
            if ($ImagePrefix) { $lines += "IMAGE_PREFIX=$ImagePrefix" }
            [IO.File]::WriteAllLines($envFile, $lines)
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

$FlyCheck = @'
set -eo pipefail
python -c 'import flybrain, cupy, grpc, google.protobuf; print("flybrain", flybrain.__version__ if hasattr(flybrain, "__version__") else "ok", "grpc", grpc.__version__)'
'@

function Test-Images {
    $images = Get-ComposeImages
    $evidence = @()
    foreach ($svc in $BuildServices) {
        $img = $images[$svc]
        if (-not (Test-Native 'docker' @('image', 'inspect', $img))) { return @{ ok = $false; why = "$img missing" } }
        $check = switch ($svc) { 'sim-controller' { $SimCheck } 'fly-brain' { $FlyCheck } default { $RosCheck } }
        $r = Invoke-InImage $img $check
        if ($r.Code -ne 0) { return @{ ok = $false; svc = $svc; why = "$img failed verification: $(($r.Out -split "`n" | Select-Object -Last 3) -join ' | ')"; out = $r.Out } }
        $evidence += "$img ok ($((($r.Out -split "`n") | Where-Object { $_ } | Select-Object -Last 2) -join ', '))"
    }
    @{ ok = $true; why = ($evidence -join '; ') }
}

# Builds one service, retrying the failures the catalog says a retry fixes.
# Returns the number of attempts used.
function Build-Service([string]$Svc, [string]$Image, [switch]$NoCache) {
    $buildLog = Join-Path $StateDir 'build.log'
    $restarted = $false
    for ($i = 1; $i -le $BuildRetries; $i++) {
        Write-Info "Building $Svc ($Image), attempt $i of $BuildRetries. The first build takes 20-60 minutes; log: $buildLog"
        $buildArgs = @('build', '--progress=plain')
        if ($NoCache -and $i -eq 1) { $buildArgs += '--no-cache' }
        $r = Invoke-Compose ($buildArgs + @($Svc)) -Quiet
        Add-Content -Path $buildLog -Value $r.Out
        if ($r.Code -eq 0) { return $i }

        $hit = Find-Failure 'build' $r.Out
        if (-not $hit) {
            $err = ($r.Out -split "`n" | Where-Object { $_ -match 'ERROR' } | Select-Object -Last 2) -join ' | '
            Write-Warn "attempt $i failed with an unrecognised error, retrying: $err"
            continue
        }
        Show-Diagnosis $hit
        switch ($hit.Entry.Fix) {
            'retry' { Add-Diagnosis 'build' $hit "retry (attempt $i)"; Start-Sleep -Seconds 10 }
            'restart-docker' {
                if (-not $restarted -and (Confirm-Fix 'Restart Docker Desktop now? Running containers stop and restart with it.')) {
                    Add-Diagnosis 'build' $hit 'restart Docker Desktop, retry'
                    Restart-DockerDesktop
                    $restarted = $true
                } else {
                    Add-Diagnosis 'build' $hit 'advised'
                    Stop-Phase "building $Svc failed [$($hit.Id)] $($hit.Entry.Advice)"
                }
            }
            default { Add-Diagnosis 'build' $hit 'advised'; Stop-Phase "building $Svc failed [$($hit.Id)] $($hit.Entry.Advice)" }
        }
    }
    Stop-Phase "building $Svc failed $BuildRetries times; see $buildLog"
}

# Verifies the images; on a catalogued failure that a clean rebuild fixes,
# clears the build cache and rebuilds that image once. Returns evidence.
function Confirm-ImagesHealthy {
    $t = Test-Images
    if ($t.ok) { return $t.why }
    if (-not $t.ContainsKey('out')) { Stop-Phase $t.why }
    $hit = Find-Failure 'verify' $t.out
    if (-not $hit -or $hit.Entry.Fix -ne 'rebuild-clean') { Stop-WithDiagnosis 'verify' $t.out $t.why }
    Show-Diagnosis $hit
    if (-not (Confirm-Fix "Clear Docker's build cache (for all projects on this PC) and rebuild $($t.svc) without cache?")) {
        Add-Diagnosis 'verify' $hit 'advised'
        Stop-Phase "$($t.why) [$($hit.Id)] $($hit.Entry.Advice)"
    }
    Add-Diagnosis 'verify' $hit "prune build cache, rebuild $($t.svc) without cache"
    $img = (Get-ComposeImages)[$t.svc]
    [void](Invoke-Native 'docker' @('rmi', '-f', $img) -Quiet)
    [void](Invoke-Native 'docker' @('builder', 'prune', '-af') -Quiet)
    $script:BuildAttempts[$t.svc] = Build-Service $t.svc $img -NoCache
    $t = Test-Images
    if (-not $t.ok) { Stop-Phase "$($t.why) (still failing after a clean rebuild)" }
    "$($t.why) (repaired by a clean rebuild)"
}

function Invoke-ImagesPhase {
    Invoke-Phase 'images' "Docker images ($($BuildServices -join ', '))" -Detect {
        if ($Rebuild) { return $null }
        $t = Test-Images
        if ($t.ok) { return $t.why }
        Write-Info $t.why
    } -Install {
        $images = Get-ComposeImages
        foreach ($svc in $BuildServices) {
            $script:BuildAttempts[$svc] = Build-Service $svc $images[$svc] -NoCache:$Rebuild
        }
        $health = Confirm-ImagesHealthy
        $attempts = ($script:BuildAttempts.GetEnumerator() | ForEach-Object { "$($_.Key) in $($_.Value) attempt(s)" }) -join ', '
        "built $attempts; $health"
    }
}

# ---------------------------------------------------------------------------
# Phase 6 - Unity client
# ---------------------------------------------------------------------------

function Get-GymMarker([string]$Dir) {
    $marker = Join-Path $Dir '.roboracer-client.json'
    if (-not (Test-Path (Join-Path $Dir $GymExeName)) -or -not (Test-Path $marker)) { return $null }
    try { Get-Content $marker -Raw | ConvertFrom-Json } catch { $null }
}

function Install-Gym($Gym) {
    $dest = Join-Path $UnityBinaryDir $Gym.Folder
    $asset = Get-VerifiedAsset "roboracer-gym-$($Gym.Folder).zip"
    $tmp = Join-Path $StateDir 'unity-unpack'
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -LiteralPath $asset.Zip -DestinationPath $tmp
    $root = $tmp
    $top = @(Get-ChildItem $tmp)
    if ($top.Count -eq 1 -and $top[0].PSIsContainer) { $root = $top[0].FullName }
    if (-not (Test-Path (Join-Path $root $GymExeName))) { Stop-Phase "$($asset.Source) has no '$GymExeName' at its top level" }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Force $UnityBinaryDir | Out-Null
    Move-Item -LiteralPath $root -Destination $dest
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    [pscustomobject]@{ gym = $Gym.Name; source = $asset.Source; sha256 = $asset.Sha256; installedAt = (Get-Date).ToString('s') } |
        ConvertTo-Json | Set-Content -Path (Join-Path $dest '.roboracer-client.json') -Encoding UTF8
    "$($Gym.Name) ($($asset.Check))"
}

function Invoke-UnityPhase {
    $latest = Join-Path $RepoDir 'unity\Builds\latest'
    $default = $Gyms[0]
    Invoke-Phase 'unity' "Unity gyms ($(($Gyms | ForEach-Object { $_.Name }) -join ', '))" -Detect {
        $missing = @($Gyms | Where-Object { -not (Get-GymMarker (Join-Path $UnityBinaryDir $_.Folder)) })
        $m = Get-GymMarker $latest
        if (-not $missing -and $m -and ($m.PSObject.Properties.Name -contains 'gym') -and $m.gym -eq $default.Name) {
            return "$($Gyms.Count) gyms in $UnityBinaryDir; default $($default.Name) in unity\Builds\latest"
        }
    } -Install {
        $done = @()
        foreach ($g in $Gyms) {
            if (-not (Get-GymMarker (Join-Path $UnityBinaryDir $g.Folder))) { $done += Install-Gym $g }
        }
        $stopped = Stop-UnityClients
        if ($stopped) { Write-Info "stopped $stopped running Unity client process(es) to replace unity\Builds\latest" }
        if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
        New-Item -ItemType Directory -Force $latest | Out-Null
        Copy-Item -Path (Join-Path $UnityBinaryDir "$($default.Folder)\*") -Destination $latest -Recurse -Force
        $installed = if ($done) { "unpacked $($done -join '; ')" } else { 'gyms already unpacked' }
        "$installed; default $($default.Name) copied into unity\Builds\latest"
    }
}

# Resolves a release asset (from -AssetDir, or by default the GitHub release)
# to a local zip and verifies "<zip>.sha256" when one is published next to it.
function Get-VerifiedAsset([string]$AssetName) {
    $Source = if ($AssetDir) { Join-Path $AssetDir $AssetName } else { "https://github.com/$GithubRepo/releases/download/$UnityReleaseTag/$AssetName" }
    $zip = $Source; $expected = $null
    if ($Source -match '^https?://') {
        $zip = Join-Path $StateDir $AssetName
        Write-Info "Downloading $Source"
        Invoke-WebRequest -Uri $Source -OutFile $zip -UseBasicParsing
        try { $expected = ((Invoke-WebRequest -Uri "$Source.sha256" -UseBasicParsing).Content -split '\s+')[0] } catch { $expected = $null }
    } elseif (Test-Path "$Source.sha256") {
        $expected = ((Get-Content "$Source.sha256" -Raw) -split '\s+')[0]
    }
    if (-not (Test-Path $zip)) { Stop-Phase "$AssetName not found: $zip" }
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($expected) {
        if ($actual -ne $expected.ToLower()) { Stop-Phase "checksum mismatch for $zip (expected $expected, got $actual)" }
        $check = 'checksum verified'
    } else {
        Write-Warn "No .sha256 found next to $AssetName; skipping checksum verification."
        $check = 'no checksum available'
    }
    [pscustomobject]@{ Source = $Source; Zip = $zip; Sha256 = $actual; Check = $check }
}

# ---------------------------------------------------------------------------
# Phase 6b - expert demonstrations
# ---------------------------------------------------------------------------

function Invoke-DemosPhase {
    $dir = Join-Path $InstallDir "tfrecords\job_$DemoJobId"
    Invoke-Phase 'demos' "Expert demonstrations for the donut course (job $DemoJobId)" -Detect {
        $files = @(Get-ChildItem $dir -Filter '*.tfrecord' -File -ErrorAction SilentlyContinue)
        if ($files) { return "$($files.Count) recording(s), $([math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB)) MB in tfrecords\job_$DemoJobId" }
    } -Install {
        $asset = Get-VerifiedAsset $DemoAssetName
        $tmp = Join-Path $StateDir 'demo-unpack'
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -LiteralPath $asset.Zip -DestinationPath $tmp
        $records = @(Get-ChildItem $tmp -Recurse -Filter '*.tfrecord' -File)
        if (-not $records) { Stop-Phase "$DemoAssetName contains no .tfrecord files" }
        New-Item -ItemType Directory -Force $dir | Out-Null
        $records | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $dir -Force }
        Remove-Item $tmp -Recurse -Force
        "unpacked $($records.Count) recording(s), $([math]::Round(($records | Measure-Object Length -Sum).Sum / 1MB)) MB into tfrecords\job_$DemoJobId from $($asset.Source) ($($asset.Check))"
    }
}

# ---------------------------------------------------------------------------
# Phase 7 - start
# ---------------------------------------------------------------------------

# The ROS-TCP bridge opens a short-lived connection per message, so an
# established socket is only visible by chance. ros-server logs a handshake
# each time a Unity client starts; look for one newer than the client.
function Get-UnityRosConnection {
    foreach ($p in @(Get-UnityClientProcess)) {
        $c = Get-NetTCPConnection -OwningProcess $p.Id -RemotePort 10000 -State Established -ErrorAction SilentlyContinue
        if ($c) { return "Unity pid $($p.Id) connected to 127.0.0.1:10000" }
        if (-not $p.StartTime) { continue }
        $since = $p.StartTime.ToUniversalTime().AddSeconds(-5).ToString('yyyy-MM-ddTHH:mm:ssZ')
        $logs = (Invoke-Compose @('logs', '--no-color', '--since', $since, 'ros-server') -Quiet).Out
        $line = ($logs -split "`n" | Where-Object { $_ -match 'ROS-Unity Handshake received' } | Select-Object -Last 1)
        if ($line) { return "Unity pid $($p.Id) completed the ROS handshake ($(($line -replace '^.*?\|\s*', '').Trim()))" }
    }
    $null
}

function Get-RosServerStartTime {
    $id = ((Invoke-Compose @('ps', '-q', 'ros-server') -Quiet).Out -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 1)
    if (-not $id) { return $null }
    $raw = (Invoke-Native 'docker' @('inspect', '-f', '{{.State.StartedAt}}', $id.Trim()) -Quiet).Out.Trim()
    if ($raw -notmatch '^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)') { return $null }
    [datetime]::SpecifyKind([datetime]::ParseExact($Matches[1], 'yyyy-MM-ddTHH:mm:ss', [Globalization.CultureInfo]::InvariantCulture), 'Utc')
}

function Get-FailingServices {
    Start-Sleep -Seconds 20
    $rows = (Invoke-Compose @('ps', '-a', '--format', '{{.Service}}|{{.State}}') -Quiet).Out -split "`n"
    @($rows | ForEach-Object {
        $svc, $state = $_.Trim() -split '\|', 2
        if ($svc -and $StartServices -contains $svc -and $state -match 'restarting|exited|dead') { $svc }
    })
}

# A service that crashes on start still leaves "compose up" exiting 0, so
# look for restart loops and diagnose them from the service's own log.
function Get-ServiceLogs([string[]]$Services) {
    $out = (Invoke-Compose (@('logs', '--no-color', '--tail', '60') + $Services) -Quiet).Out
    $out -replace "\x1b\[[0-9;]*m", ''
}

# Mongo runs as uid 1001, but a data folder Docker created on a Windows drive
# comes out root-owned and not writable by others.
function Set-MongoDataOwner {
    $r = Invoke-Compose @('run', '--rm', '--no-deps', '--user', 'root', '--entrypoint', 'chown', 'mongo', '-R', '1001:1001', '/bitnami/mongodb') -Quiet
    if ($r.Code -ne 0) { Write-Warn "could not change the owner of the mongodb folder: $(($r.Out -split "`n" | Select-Object -Last 1))" }
    $r.Code -eq 0
}

function Assert-NoCrashLoop {
    $failing = Get-FailingServices
    if (-not $failing) { return }
    $logs = Get-ServiceLogs $failing
    $hit = Find-Failure 'container' $logs
    if ($hit -and $hit.Entry.Fix -in 'recreate', 'mongo-owner') {
        Show-Diagnosis $hit
        if ($hit.Entry.Fix -eq 'mongo-owner') { [void](Set-MongoDataOwner) }
        Add-Diagnosis 'container' $hit "$(if ($hit.Entry.Fix -eq 'mongo-owner') { 'gave mongodb folder to uid 1001, ' })recreated $($failing -join ', ')"
        [void](Invoke-Compose (@('up', '-d', '--no-build', '--force-recreate') + $failing) -Quiet)
        $failing = Get-FailingServices
        if (-not $failing) { return }
        $logs = Get-ServiceLogs $failing
    }
    Stop-WithDiagnosis 'container' $logs "these services keep crashing: $($failing -join ', ')"
}

function Invoke-StartPhase {
    Invoke-Phase 'start' 'Start the stack and one Unity client' -NoExpectClean -Detect {
        if (-not (Test-Path $RepoDir)) { return $null }
        $running = @((Invoke-Compose @('ps', '--status', 'running', '--services') -Quiet).Out -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        $missing = @($StartServices | Where-Object { $running -notcontains $_ })
        $conn = Get-UnityRosConnection
        if ($missing.Count -eq 0 -and $conn) { return "services running: $($StartServices -join ', '); $conn" }
    } -Install {
        $mongoDir = Join-Path $InstallDir 'mongodb'
        if (-not (Get-ChildItem $mongoDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
            if (Set-MongoDataOwner) { Write-Info 'gave the empty mongodb folder to the Mongo user (uid 1001)' }
        }
        $r = Invoke-Compose (@('up', '-d', '--no-build') + $StartServices)
        if ($r.Code -ne 0) {
            $hit = Find-Failure 'compose-up' $r.Out
            if ($hit -and $hit.Entry.Fix -eq 'port-owner') {
                Show-Diagnosis $hit
                if (Resolve-PortConflict $hit.Port) {
                    Add-Diagnosis 'compose-up' $hit "freed port $($hit.Port), retried"
                    $r = Invoke-Compose (@('up', '-d', '--no-build') + $StartServices)
                }
            }
            if ($r.Code -ne 0) { Stop-WithDiagnosis 'compose-up' $r.Out 'docker compose up failed' }
        }
        Assert-NoCrashLoop
        if (-not (Wait-Until { Test-TcpPort 10000 } -TimeoutSec 120 -What 'ros-server on port 10000')) { Stop-Phase 'ros-server did not open port 10000 within 2 minutes' }
        if (-not (Wait-Until { Test-TcpPort 6006 } -TimeoutSec 180 -What 'TensorBoard on port 6006')) { Stop-Phase 'TensorBoard did not open port 6006 within 3 minutes' }
        # First start generates its gRPC stubs and downloads the ~260 MB connectome.
        $flyUp = { (Invoke-Compose @('exec', '-T', 'fly-brain', 'python', '-c', "import socket; socket.create_connection(('127.0.0.1', 50061), 2)") -Quiet).Code -eq 0 }
        if (-not (Wait-Until $flyUp -TimeoutSec 900 -IntervalSec 10 -What 'fly-brain on port 50061 (first start downloads the connectome)')) {
            Stop-WithDiagnosis 'container' (Get-ServiceLogs @('fly-brain')) 'fly-brain did not open port 50061 within 15 minutes'
        }
        Start-Sleep -Seconds 8
        # Unity sends its ROS handshake only at startup, so a client that
        # outlived a ros-server restart never connects to the new one.
        $rosStarted = Get-RosServerStartTime
        $stale = @(Get-UnityClientProcess | Where-Object { $rosStarted -and $_.StartTime -and $_.StartTime.ToUniversalTime() -lt $rosStarted })
        if ($stale) {
            Write-Info "restarting $($stale.Count) Unity client(s) started before the current ros-server"
            $stale | Stop-Process -Force -ErrorAction SilentlyContinue
            [void](Wait-Until { [bool](Get-UnityClientProcess) } -TimeoutSec 45 -What 'the client wrapper to relaunch Unity')
        }
        if (-not (Get-UnityClientProcess)) {
            $wrapper = Join-Path $RepoDir 'scripts\RunClientWrapper.ps1'
            Start-Process -FilePath 'powershell.exe' -WindowStyle Minimized -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$wrapper`"", '-Index', '0',
                '-GymPollSeconds', '10', '-GymSource', "`"$DefaultGymExe`"")
        }
        if (-not (Wait-Until { [bool](Get-UnityRosConnection) } -TimeoutSec 180 -What 'the Unity client to connect to ros-server')) {
            $playerLog = Join-Path $RepoDir 'unity\Builds\latest\Player.log'
            $text = if (Test-Path $playerLog) { (Get-Content $playerLog -Tail 300) -join "`n" } else { '' }
            $hit = Find-Failure 'unity-client' $text
            if ($hit -and $hit.Entry.Fix -eq 'port-owner') {
                Show-Diagnosis $hit
                Add-Diagnosis 'unity-client' $hit 'advised'
                $owner = Get-PortOwner 5005
                if ($owner) { Write-Host "      port 5005 is used by $($owner.Text)" -ForegroundColor Yellow }
                Stop-Phase "the Unity client could not open its port 5005 [$($hit.Id)]; stop the program using it, then re-run"
            }
            Stop-WithDiagnosis 'unity-client' $text "the Unity client did not connect to ros-server within 3 minutes (log: $playerLog)"
        }
        $dash = if (Wait-Until { Test-TcpPort 80 } -TimeoutSec 300 -IntervalSec 10 -What 'the dashboard (first start runs npm install)') { 'dashboard up' } else { 'dashboard still starting' }
        "services started; $(Get-UnityRosConnection); $dash"
    }
}

# ---------------------------------------------------------------------------
# Phase 7b - starter documents
# ---------------------------------------------------------------------------

function Get-SeedJson {
    $path = Join-Path $RepoDir 'scripts\install\seed\seed.json'
    if (-not (Test-Path $path)) { Stop-Phase "seed file missing: $path" }
    (Get-Content $path -Raw).Replace('{{UNITY_BINARY_DIR}}', $UnityBinaryDir.Replace('\', '\\'))
}

# On an empty data folder Mongo runs its first-time setup and then restarts,
# so one successful ping is not enough; require three in a row.
function Wait-MongoReady([int]$TimeoutSec = 240) {
    $streak = 0
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $r = Invoke-Mongo 'print(db.runCommand({ping: 1}).ok)'
        if ($r.Code -eq 0 -and $r.Out -match '(?m)^1\s*$') { $streak++ } else { $streak = 0 }
        if ($streak -ge 3) { return $true }
        Start-Sleep -Seconds 5
    }
    $false
}

function Invoke-SeedPhase {
    Invoke-Phase 'seed' 'Starter gyms, reward design and experiment designs' -Detect {
        if (-not (Test-Path $RepoDir)) { return $null }
        $seed = (Get-SeedJson | ConvertFrom-Json)
        $want = 0; $checks = @()
        foreach ($coll in $seed.PSObject.Properties.Name) {
            foreach ($doc in $seed.$coll) {
                $want++
                $id = if ($doc._id -is [string]) { "'$($doc._id)'" } else { "ObjectId('$($doc._id.'$oid')')" }
                $checks += "db.getCollection('$coll').countDocuments({_id: $id})"
            }
        }
        $r = Invoke-Mongo "print($($checks -join ' + '))"
        $have = ($r.Out -split "`n" | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -Last 1)
        if ($r.Code -eq 0 -and $have -and [int]$have -eq $want) { return "all $want starter documents present" }
    } -Install {
        Write-Info 'waiting for MongoDB to finish starting (a new database restarts once after setup) ...'
        if (-not (Wait-MongoReady)) {
            Stop-WithDiagnosis 'container' (Get-ServiceLogs @('mongo')) 'MongoDB did not answer within 4 minutes'
        }
        $tmp = Join-Path $StateDir 'rl-seed.json'
        [IO.File]::WriteAllText($tmp, (Get-SeedJson), (New-Object Text.UTF8Encoding $false))
        foreach ($copy in @(@($tmp, 'mongo:/tmp/rl-seed.json'), @((Join-Path $RepoDir 'scripts\install\seed\seed.js'), 'mongo:/tmp/rl-seed.js'))) {
            $c = Invoke-Compose @('cp', $copy[0], $copy[1]) -Quiet
            if ($c.Code -ne 0) { Stop-Phase "could not copy $(Split-Path $copy[0] -Leaf) into the mongo container: $($c.Out)" }
        }
        $r = Invoke-Mongo '' -File '/tmp/rl-seed.js'
        $line = ($r.Out -split "`n" | Where-Object { $_ -match '^SEEDED ' } | Select-Object -Last 1)
        if ($r.Code -ne 0 -or -not $line) { Stop-Phase "seeding failed: $($r.Out)" }
        "inserted (new/total) $($line -replace '^SEEDED ', '')"
    }
}

# ---------------------------------------------------------------------------
# Phase 8 - first TRAIN job
# ---------------------------------------------------------------------------

# Tag count across all TensorBoard runs, or -1 while TensorBoard is unreachable.
function Get-TensorBoardTagCount {
    try {
        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:6006/data/plugin/scalars/tags' -TimeoutSec 10
        $n = 0
        foreach ($run in $tags.PSObject.Properties) { $n += @($run.Value.PSObject.Properties).Count }
        $n
    } catch { -1 }
}

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

function Stop-TrainerFailure([string]$Message) {
    $out = Join-Path $RepoDir 'rl_agent\robotaxi.out'
    $text = if (Test-Path $out) { (Get-Content $out -Tail 300) -join "`n" } else { '' }
    Stop-WithDiagnosis 'trainer' $text "$Message (log: $out)"
}

function Invoke-FirstJobPhase {
    Invoke-Phase 'firstjob' "First TRAIN job ($FirstJobIterations iterations on $($Gyms[0].Name))" -Detect {
        if (-not (Test-Path $RepoDir)) { return $null }
        $s = Get-FirstJobStatus
        if ($s -in 'IN_PROGRESS', 'DONE') { return "installer's first job already exists (status $s)" }
    } -Install {
        if ((Get-FirstJobStatus) -eq 'FAILED') {
            [void](Invoke-Mongo "db.jobs.updateMany({installer_first_job: true, status: 'FAILED'}, {`$set: {installer_first_job: false}})")
            Write-Info 'the previous first job failed; queueing a new one'
        }
        if ((Get-FirstJobStatus) -eq 'NONE') {
            # Straight into SAC: no BC pretrain on the demos, no eval before
            # the first training step.
            $js = "const g = db.gyms.findOne({_id: ObjectId('$($Gyms[0].Id)')}); " +
                  "db.jobs.insertOne({job_type: 'TRAIN', model_type: 'SacAgent', robot_type: 'robotaxi', " +
                  "num_iterations: $FirstJobIterations, status: 'NOT_STARTED', create_date: new Date(), " +
                  "demo_job_id: '', pass_through_actions: false, nn_size_x: '', nn_size_y: '', seed: 0, " +
                  "skip_first_eval: true, bc_pretrain_steps: 0, " +
                  "gym_id: g ? g._id.toString() : '', gym_name: g ? g.name : '', gym_file_path: g ? g.file_path : '', " +
                  "percent_complete: 0, installer_first_job: true}).insertedId.toString()"
            $r = Invoke-Mongo $js
            if ($r.Code -ne 0) { Stop-Phase "could not insert the job: $($r.Out)" }
            Write-Info "Queued job $(($r.Out -split "`n" | Select-Object -Last 1).Trim())"
        }
        if (-not (Wait-Until { (Get-FirstJobStatus) -in 'IN_PROGRESS', 'DONE', 'FAILED' } -TimeoutSec 600 -IntervalSec 10 -What 'the trainer to pick up the job')) {
            Stop-TrainerFailure "the trainer did not start the job within 10 minutes (status $(Get-FirstJobStatus))"
        }
        $ok = Wait-Until { @(Get-TrainLines).Count -ge 2 -or (Get-FirstJobStatus) -eq 'FAILED' } -TimeoutSec ($FirstJobWaitMinutes * 60) -IntervalSec 20 -What 'training steps'
        if ((Get-FirstJobStatus) -eq 'FAILED') { Stop-TrainerFailure 'the first training job failed' }
        if (-not $ok) { Stop-TrainerFailure "no training steps after $FirstJobWaitMinutes minutes" }
        if (-not (Wait-Until { (Get-TensorBoardTagCount) -gt 0 } -TimeoutSec 300 -IntervalSec 15 -What 'TensorBoard scalars')) {
            Stop-TrainerFailure "training is running but TensorBoard shows no scalars after 5 minutes (http://127.0.0.1:6006)"
        }
        $lines = @(Get-TrainLines)
        "job $(Get-FirstJobStatus); $($lines.Count) training iterations so far; $(Get-TensorBoardTagCount) TensorBoard scalar tags; last: $($lines[-1])"
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

$script:RebootNeeded = $false
$script:SignOutNeeded = $false
$script:BuildAttempts = @{}
$exitCode = 0

if ($SelfTest) {
    $selfTestFailures = Invoke-SelfTest
    exit [int]($selfTestFailures -gt 0)
}

New-Item -ItemType Directory -Force $StateDir | Out-Null
$LogPath = Join-Path $StateDir "install-$RunStamp.log"
Start-Transcript -Path $LogPath -Append | Out-Null

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
        Invoke-DemosPhase
        if (-not $DryRun) {
            Invoke-StartPhase
            Invoke-SeedPhase
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
    if ($script:Diagnoses.Count) {
        Write-Host 'Known problems recognised during this run:'
        $script:Diagnoses | Format-Table phase, id, action -AutoSize | Out-String | Write-Host
    }
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
