<#
.SYNOPSIS
    InstallZero uninstaller for rl-roboracer: removes what Install.ps1 put on
    this PC for this Windows account.

.DESCRIPTION
    Removes, after asking once:

      - the Unity clients and their supervisor windows
      - the install's Docker containers, networks and volumes, and the images
        it built (an image another container still uses is kept)
      - the Startup shortcut and any pending resume entry
      - the code, the Unity gyms and the .env in the install folder (only
        the folders the installer creates; the install folder itself only
        when nothing else is left in it)
      - the installer's state, logs and report, and the doctor's transcripts
      - trained models, the database and recorded demos (saved_models,
        mongodb, tfrecords) only if you say so, or with -RemoveData

    Docker containers are only taken down when they were started from this
    install folder: another checkout can use the same compose project name.

    It leaves what other software may use: WSL, Git, Docker Desktop, the
    base images Docker downloaded (mongo, mongo-express) and Docker's build
    cache (-PruneBuildCache removes that too).

.PARAMETER InstallDir
    The -InstallDir the install used. Default %USERPROFILE%\rl-roboracer.

.PARAMETER RemoveData
    Also delete saved_models, mongodb and tfrecords without asking.

.PARAMETER KeepData
    Keep them without asking.

.PARAMETER PruneBuildCache
    Also empty Docker's build cache, which every Docker build on this
    account shares. Makes the next install build from scratch.

.PARAMETER Yes
    Do not ask for confirmation. Data is kept unless -RemoveData.

.EXAMPLE
    .\Uninstall.ps1
    Asks, then removes the install in %USERPROFILE%\rl-roboracer.

.EXAMPLE
    .\Uninstall.ps1 -InstallDir D:\rl -RemoveData -Yes
    Removes everything, including trained models, without asking.

.EXAMPLE
    .\Uninstall.ps1 -WhatIf
    Lists what would be removed.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'rl-roboracer'),
    [switch]$RemoveData,
    [switch]$KeepData,
    [switch]$PruneBuildCache,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$whatIf = $WhatIfPreference; $WhatIfPreference = $false; Import-Module CimCmdlets; $WhatIfPreference = $whatIf

$InstallDir = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$repo      = Join-Path $InstallDir 'rl-roboracer'
$stateDir  = Join-Path $env:LOCALAPPDATA 'rl-roboracer-install'
$doctorDir = Join-Path $env:LOCALAPPDATA 'rl-roboracer'
$dataDirs  = @('saved_models', 'mongodb', 'tfrecords' | ForEach-Object { Join-Path $InstallDir $_ } | Where-Object { Test-Path $_ })
$dockerExe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
$env:Path += ";$(Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin')"
$interactive = [Environment]::UserInteractive -and $Host.Name -eq 'ConsoleHost' -and -not $Yes

function Step([string]$Text) { Write-Host "  $Text" }
function Invoke-Quiet([scriptblock]$Block) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $Block 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $old }
}
function Remove-Path([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if ($PSCmdlet.ShouldProcess($Path, 'delete')) { Step "deleting $Path"; Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Continue }
}
function Get-EnvValue([string]$Key) {
    $envFile = Join-Path $repo '.env'
    if (-not (Test-Path $envFile)) { return $null }
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -Last 1
    if ($line) { ($line -split '=', 2)[1].Trim() }
}
function Get-SizeGB([string[]]$Paths) {
    $bytes = ($Paths | ForEach-Object { Get-ChildItem $_ -Recurse -File -Force -ErrorAction SilentlyContinue } | Measure-Object Length -Sum).Sum
    [math]::Round([double]$bytes / 1GB, 1)
}

if (-not (Test-Path $InstallDir) -and -not (Test-Path $stateDir)) {
    Write-Host "Nothing to remove: no install at $InstallDir and no installer state in $stateDir."
    exit 0
}
if ((Test-Path $InstallDir) -and -not (Test-Path (Join-Path $repo 'docker-compose.yml')) -and -not $dataDirs) {
    throw "$InstallDir does not look like an rl-roboracer install (no rl-roboracer\docker-compose.yml, saved_models, mongodb or tfrecords). Pass the folder the install used with -InstallDir."
}

Write-Host "InstallZero: uninstall rl-roboracer from $InstallDir" -ForegroundColor Cyan
if ($interactive -and -not $WhatIfPreference) {
    if ((Read-Host '  Remove it? [y/N]') -notmatch '^(y|yes)$') { Write-Host '  Nothing removed.'; exit 0 }
}
$deleteData = $RemoveData -and -not $KeepData
if ($dataDirs -and -not $RemoveData -and -not $KeepData -and $interactive -and -not $WhatIfPreference) {
    $deleteData = (Read-Host "  Also delete trained models, the database and recorded demos ($(Get-SizeGB $dataDirs) GB in saved_models, mongodb, tfrecords)? [y/N]") -match '^(y|yes)$'
}

# 1. Unity clients; their supervisors relaunch them, so those go first.
$sup = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine.IndexOf($repo, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
foreach ($p in $sup) { if ($PSCmdlet.ShouldProcess("supervisor pid $($p.ProcessId)", 'stop')) { Step "stopping supervisor pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } }
$unity = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith("$InstallDir\", 'OrdinalIgnoreCase') })
foreach ($p in $unity) { if ($PSCmdlet.ShouldProcess("$($p.Name) pid $($p.ProcessId)", 'stop')) { Step "stopping $($p.Name) pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } }

# 2. Docker: this install's containers, volumes and built images.
$dockerLeft = $false
if ((Test-Path (Join-Path $repo 'docker-compose.yml')) -and (Get-Command docker -ErrorAction SilentlyContinue)) {
    $mySession = (Get-Process -Id $PID).SessionId
    $others = @(Get-CimInstance Win32_Process -Filter "Name='Docker Desktop.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -ne $mySession })
    [void](Invoke-Quiet { docker info --format '{{.ServerVersion}}' }); $up = $LASTEXITCODE -eq 0
    if ($others) {
        Write-Host '  Docker Desktop is running for another Windows account, so docker commands here could reach its engine. Skipping Docker; quit it there and run this again to remove the containers and images.' -ForegroundColor Yellow
        $dockerLeft = $true; $up = $false
    } elseif (-not $up -and $WhatIfPreference) {
        Write-Host '  Docker is not running; a real run would start it to remove the containers and images.'
    } elseif (-not $up -and (Test-Path $dockerExe)) {
        Step 'starting Docker Desktop to remove the containers and images ...'
        Start-Process $dockerExe | Out-Null
        $deadline = (Get-Date).AddMinutes(4)
        while (-not $up -and (Get-Date) -lt $deadline) { Start-Sleep 10; [void](Invoke-Quiet { docker info --format '{{.ServerVersion}}' }); $up = $LASTEXITCODE -eq 0 }
        if (-not $up) { Write-Host '  Docker did not start; its containers and images are left.' -ForegroundColor Yellow; $dockerLeft = $true }
    }
    if ($up) {
        Push-Location $repo
        try {
            $files = @('-f', 'docker-compose.yml'); if (Test-Path 'compose\scale.yml') { $files += @('-f', 'compose\scale.yml') }
            $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            $cfgText = (docker compose @files config --format json 2>$null) -join "`n"
            $ErrorActionPreference = $old
            $cfg = if ($LASTEXITCODE -eq 0) { try { $cfgText | ConvertFrom-Json } catch { $null } } else { $null }
            $project = if ($cfg) { $cfg.name } else { $null }
            if (-not $project) { $project = Get-EnvValue 'COMPOSE_PROJECT_NAME' }
            if (-not $project) { $project = 'rl-roboracer' }
            $built = if ($cfg) { @($cfg.services.PSObject.Properties | Where-Object { $_.Value.PSObject.Properties.Name -contains 'build' -and $_.Value.image } | ForEach-Object { $_.Value.image } | Sort-Object -Unique) } else { @() }

            # No quotes in the template: Windows PowerShell 5.1 strips them from native arguments.
            $labels = @(Invoke-Quiet { docker ps -a --filter "label=com.docker.compose.project=$project" --format '{{.Labels}}' })
            $dirs = @($labels | ForEach-Object { [regex]::Match($_, 'com\.docker\.compose\.project\.working_dir=([^,]+)').Groups[1].Value } | Where-Object { $_ } | Sort-Object -Unique)
            $foreign = @($dirs | Where-Object { $_.TrimEnd('\') -ne $repo })
            if ($foreign) {
                Write-Host "  Compose project '$project' has containers started from another folder ($($foreign -join ', ')); leaving its containers and volumes alone." -ForegroundColor Yellow
                $dockerLeft = $true
            } elseif ($PSCmdlet.ShouldProcess("compose project '$project'", 'down --volumes')) {
                Step "removing compose project '$project' (containers, networks, volumes)"
                [void](Invoke-Quiet { docker compose @files down --volumes --remove-orphans })
            }
            foreach ($img in $built) {
                $users = @(Invoke-Quiet { docker ps -a --filter "ancestor=$img" --format '{{.Names}}' } | Where-Object { $_ -and $_ -notmatch '^(Error|error)' })
                if ($users) { Step "keeping image $img (used by $($users -join ', '))"; continue }
                if (-not (Invoke-Quiet { docker image inspect --format '{{.Id}}' $img }) -or $LASTEXITCODE -ne 0) { continue }
                if ($PSCmdlet.ShouldProcess($img, 'remove image')) { Step "removing image $img"; [void](Invoke-Quiet { docker rmi $img }) }
            }
            if ($PruneBuildCache -and $PSCmdlet.ShouldProcess("Docker's build cache", 'prune')) {
                Step 'pruning Docker build cache'
                [void](Invoke-Quiet { docker builder prune --all --force })
            }
        } finally { Pop-Location }
    }
}

# 3. Sign-in hooks.
$startup = [Environment]::GetFolderPath('Startup')
$shell = New-Object -ComObject WScript.Shell
Get-ChildItem $startup -Filter 'rl-roboracer*.lnk' -ErrorAction SilentlyContinue |
    Where-Object { $shell.CreateShortcut($_.FullName).Arguments.IndexOf($repo, [StringComparison]::OrdinalIgnoreCase) -ge 0 } |
    ForEach-Object { Remove-Path $_.FullName }
$runOnce = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
if ((Get-ItemProperty $runOnce -ErrorAction SilentlyContinue).PSObject.Properties.Name -contains 'rl-roboracer-install') {
    if ($PSCmdlet.ShouldProcess('RunOnce rl-roboracer-install', 'remove')) { Step 'removing the pending resume entry'; Remove-ItemProperty $runOnce -Name 'rl-roboracer-install' }
}

# 4. Files.
# Only what Install.ps1 creates; anything else in the folder is someone's own.
Remove-Path $repo
Remove-Path (Join-Path $InstallDir 'UnityBinary')
if ($deleteData) { $dataDirs | ForEach-Object { Remove-Path $_ } }
if ((Test-Path $InstallDir) -and -not @(Get-ChildItem $InstallDir -Force -ErrorAction SilentlyContinue) -and -not $WhatIfPreference) { Remove-Path $InstallDir }
Remove-Path $stateDir
Remove-Path $doctorDir

if ($WhatIfPreference) { Write-Host 'Preview only (-WhatIf): nothing was removed.'; exit 0 }
$left = @(@($repo, $stateDir, $doctorDir) | Where-Object { Test-Path $_ })
if ($left) { Write-Host "  Could not delete: $($left -join ', ') (a file may be in use). Close the Unity window or sign out and back in, then run this again." -ForegroundColor Yellow; exit 1 }
Write-Host 'rl-roboracer is uninstalled.' -ForegroundColor Green
if (-not $deleteData -and $dataDirs) { Write-Host "Kept your data: $($dataDirs -join ', '). Delete those folders, or run again with -RemoveData, to remove it." }
if ($dockerLeft) { exit 1 }
Write-Host 'WSL, Git and Docker Desktop are still installed; remove them from Settings > Apps if nothing else needs them.'
