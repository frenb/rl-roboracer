<#
.SYNOPSIS
    Remove everything Install.ps1 put in a test Windows account (normally
    rltest), so the next install there starts from nothing.

.DESCRIPTION
    Run it signed in as the test account, in a normal (not administrator)
    PowerShell. It removes, for this account only:

      - Unity clients and their supervisor windows
      - the compose project's containers and volumes, every Docker image,
        volume and build cache in this account's Docker Desktop
      - the Startup shortcut and any pending RunOnce resume entry
      - the install folder (repo, saved_models, mongodb, tfrecords, gyms)
      - the installer's state, logs and report, and the doctor's transcripts
      - with -ResetDocker: this account's Docker Desktop data and settings
        (its docker-desktop WSL distro, %APPDATA%\Docker,
        %LOCALAPPDATA%\Docker, ~\.docker), so Docker Desktop starts as on
        first use

    It does not uninstall what the whole machine shares: WSL, Git and the
    Docker Desktop program stay, and the account stays in docker-users (see
    docs/rltest-clean-test.md to undo those).

    Docker Desktop's engine pipe is shared by all Windows accounts. If
    another account has Docker Desktop running, docker commands here could
    reach that account's engine, so the script refuses to touch Docker
    until it is quit there.

.PARAMETER Account
    The account this must run in. A guard against running it in your main
    account. Default rltest.

.PARAMETER InstallDir
    The -InstallDir the test install used. Default %USERPROFILE%\rl-roboracer.

.PARAMETER SaveResultsTo
    Copy the installer's report and logs and the doctor's results here first.
    Default C:\Users\Public\rl-roboracer-test\results\<timestamp>.

.PARAMETER NoSaveResults
    Do not copy the results anywhere.

.PARAMETER ResetDocker
    Also delete this account's Docker Desktop data and settings.

.PARAMETER WhatIf
    Show what would be removed; remove nothing.

.EXAMPLE
    .\Reset-TestAccount.ps1 -WhatIf
    Lists what would be removed.

.EXAMPLE
    .\Reset-TestAccount.ps1 -ResetDocker
    Full reset, including this account's Docker Desktop data and settings.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Account = 'rltest',
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'rl-roboracer'),
    [string]$SaveResultsTo = ("C:\Users\Public\rl-roboracer-test\results\" + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [switch]$NoSaveResults,
    [switch]$ResetDocker
)

$ErrorActionPreference = 'Stop'
$whatIf = $WhatIfPreference; $WhatIfPreference = $false; Import-Module CimCmdlets; $WhatIfPreference = $whatIf
if ($env:USERNAME -ne $Account) { throw "This removes the whole test install of account '$Account', but you are signed in as '$env:USERNAME'. Sign in as $Account, or pass -Account $env:USERNAME if this really is a test account." }

$repo     = Join-Path $InstallDir 'rl-roboracer'
$stateDir = Join-Path $env:LOCALAPPDATA 'rl-roboracer-install'
$doctorDir = Join-Path $env:LOCALAPPDATA 'rl-roboracer'
$dockerExe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
$env:Path += ";$(Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin')"

function Step([string]$Text) { Write-Host "  $Text" }
function Invoke-Quiet([scriptblock]$Block) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $Block 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $old }
}
function Remove-Path([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if ($PSCmdlet.ShouldProcess($Path, 'delete')) { Step "deleting $Path"; Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Continue }
}

Write-Host "Resetting test account $Account (install folder $InstallDir)" -ForegroundColor Cyan

# 1. Results first, while they still exist.
if (-not $NoSaveResults) {
    $items = @(Get-ChildItem $stateDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.json', '.log', '.txt' }) +
             @(Get-ChildItem (Join-Path $doctorDir 'doctor') -File -ErrorAction SilentlyContinue)
    if ($items -and $PSCmdlet.ShouldProcess($SaveResultsTo, "copy $($items.Count) result files")) {
        New-Item -ItemType Directory -Force $SaveResultsTo | Out-Null
        $items | Copy-Item -Destination $SaveResultsTo -Force
        Step "saved $($items.Count) report, log and doctor files to $SaveResultsTo"
    }
}

# 2. Unity clients; their supervisors relaunch them, so those go first.
$sup = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'RunClientWrapper|RunNClients|Start-ClientAtLogon|Start-Clients' })
foreach ($p in $sup) { if ($PSCmdlet.ShouldProcess("supervisor pid $($p.ProcessId)", 'stop')) { Step "stopping supervisor pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } }
$unity = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDir, 'OrdinalIgnoreCase') -and $_.ProcessId -ne $PID })
foreach ($p in $unity) { if ($PSCmdlet.ShouldProcess("$($p.Name) pid $($p.ProcessId)", 'stop')) { Step "stopping $($p.Name) pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } }

# 3. Docker, only when the engine is this account's.
$mySession = (Get-Process -Id $PID).SessionId
$others = @(Get-CimInstance Win32_Process -Filter "Name='Docker Desktop.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -ne $mySession })
if ($others) {
    Write-Host "  Docker Desktop is running in another Windows session (session $(($others.SessionId | Sort-Object -Unique) -join ', ')). Quit it there (tray icon > Quit Docker Desktop, or sign that account out), then run this again." -ForegroundColor Yellow
    exit 1
}
if (Test-Path $dockerExe) {
    $up = (Invoke-Quiet { docker info --format '{{.ServerVersion}}' }) -and $LASTEXITCODE -eq 0
    if (-not $up -and -not $ResetDocker) {
        Step 'starting Docker Desktop to remove containers, images and volumes ...'
        Start-Process $dockerExe | Out-Null
        $deadline = (Get-Date).AddMinutes(5)
        while (-not $up -and (Get-Date) -lt $deadline) { Start-Sleep 10; [void](Invoke-Quiet { docker info --format '{{.ServerVersion}}' }); $up = $LASTEXITCODE -eq 0 }
        if (-not $up) { Write-Host '  Docker Desktop did not start; skipping the Docker cleanup (use -ResetDocker to delete its data instead).' -ForegroundColor Yellow }
    }
    if ($up) {
        if ((Test-Path (Join-Path $repo 'docker-compose.yml')) -and $PSCmdlet.ShouldProcess('compose project', 'down --volumes')) {
            Step 'docker compose down --volumes'
            Push-Location $repo
            $files = @('-f', 'docker-compose.yml'); if (Test-Path 'compose\scale.yml') { $files += @('-f', 'compose\scale.yml') }
            [void](Invoke-Quiet { docker compose @files down --volumes --remove-orphans })
            Pop-Location
        }
        if ($PSCmdlet.ShouldProcess("this account's Docker engine", 'remove all containers, images, volumes and build cache')) {
            $ids = @(Invoke-Quiet { docker ps -aq } | Where-Object { $_ -match '^[0-9a-f]+$' })
            if ($ids) { Step "removing $($ids.Count) remaining containers"; [void](Invoke-Quiet { docker rm -f @ids }) }
            Step 'docker system prune --all --volumes (images, volumes, networks, build cache)'
            Invoke-Quiet { docker system prune --all --volumes --force } | Select-Object -Last 1 | ForEach-Object { Step "  $_" }
            [void](Invoke-Quiet { docker builder prune --all --force })
        }
    }
}
if ($ResetDocker) {
    if ($PSCmdlet.ShouldProcess("this account's Docker Desktop data and settings", 'delete')) {
        Step 'quitting Docker Desktop'
        Get-Process 'Docker Desktop', 'com.docker.backend', 'com.docker.build', 'docker-sandbox' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep 5
        [void](Invoke-Quiet { wsl.exe --shutdown })
        foreach ($d in 'docker-desktop', 'docker-desktop-data') {
            $listed = (Invoke-Quiet { wsl.exe -l -q }) -replace "`0", '' | ForEach-Object { $_.Trim() }
            if ($listed -contains $d) { Step "wsl --unregister $d"; [void](Invoke-Quiet { wsl.exe --unregister $d }) }
        }
    }
    foreach ($d in (Join-Path $env:APPDATA 'Docker'), (Join-Path $env:APPDATA 'Docker Desktop'), (Join-Path $env:LOCALAPPDATA 'Docker'), (Join-Path $env:USERPROFILE '.docker')) { Remove-Path $d }
}

# 4. Sign-in hooks.
$startup = [Environment]::GetFolderPath('Startup')
Get-ChildItem $startup -Filter 'rl-roboracer*.lnk' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Path $_.FullName }
$runOnce = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
if ((Get-ItemProperty $runOnce -ErrorAction SilentlyContinue).PSObject.Properties.Name -contains 'rl-roboracer-install') {
    if ($PSCmdlet.ShouldProcess('RunOnce rl-roboracer-install', 'remove')) { Step 'removing the RunOnce resume entry'; Remove-ItemProperty $runOnce -Name 'rl-roboracer-install' }
}

# 5. Files.
Remove-Path $InstallDir
Remove-Path $stateDir
Remove-Path $doctorDir

$left = @($InstallDir, $stateDir, $doctorDir | Where-Object { Test-Path $_ })
if ($left -and -not $WhatIfPreference) {
    Write-Host "  Could not delete everything (a file may still be in use): $($left -join ', '). Sign out and back in, then run this again." -ForegroundColor Yellow
    exit 1
}
if ($WhatIfPreference) { Write-Host 'Preview only (-WhatIf): nothing was removed.'; exit 0 }
Write-Host 'Done. Next install in this account starts from nothing.' -ForegroundColor Green
if (-not $ResetDocker) { Write-Host '(Docker Desktop settings and its first-run state were kept; add -ResetDocker to reset those too.)' }
