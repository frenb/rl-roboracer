# InstallZero one-line installer for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1 | iex
#
# With installer options (Get-Help on the downloaded Install.ps1 lists them):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1))) -Doctor xai
#
# Downloads scripts/install/Install.ps1 and the install doctor into
# %LOCALAPPDATA%\rl-roboracer-install and runs it in Windows PowerShell,
# passing the options through. RL_INSTALL_BRANCH picks another branch.

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'frenb/rl-roboracer'
    $branch = if ($env:RL_INSTALL_BRANCH) { $env:RL_INSTALL_BRANCH } else { 'main' }
    $raw = "https://raw.githubusercontent.com/$repo/$branch"
    $dir = Join-Path $env:LOCALAPPDATA 'rl-roboracer-install'
    $doctorDir = Join-Path $dir 'doctor'

    Write-Host "InstallZero: downloading the rl-roboracer installer ($branch) to $dir ..."
    New-Item -ItemType Directory -Force $dir | Out-Null
    Invoke-WebRequest -UseBasicParsing "$raw/scripts/install/Install.ps1" -OutFile (Join-Path $dir 'Install.ps1')

    # The doctor diagnoses failures, including ones before the repo is cloned.
    # Without it the install still runs, with catalog diagnoses only.
    function Get-DoctorFiles([string]$Path) {
        foreach ($e in Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$repo/contents/${Path}?ref=$branch") {
            if ($e.type -eq 'dir' -and $e.name -notin 'eval', 'assets') { Get-DoctorFiles $e.path }
            elseif ($e.type -eq 'file') { $e.path }
        }
    }
    try {
        $files = @(Get-DoctorFiles 'scripts/install/doctor')
        if (Test-Path $doctorDir) { Remove-Item $doctorDir -Recurse -Force }
        foreach ($f in $files) {
            $dest = Join-Path $doctorDir (($f -replace '^scripts/install/doctor/', '') -replace '/', '\')
            New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
            Invoke-WebRequest -UseBasicParsing "$raw/$f" -OutFile $dest
        }
    } catch {
        Write-Host "The InstallZero doctor could not be downloaded ($($_.Exception.Message)); continuing without it." -ForegroundColor Yellow
    }

    $pass = @($args)
    if (-not ($pass | Where-Object { $_ -match '^-Branch(:|$)' })) { $pass += @('-Branch', $branch) }
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir 'Install.ps1') @pass
} @args
