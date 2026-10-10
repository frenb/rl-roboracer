# InstallZero one-line installer for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1 | iex
#
# With installer options (Get-Help on the downloaded Install.ps1 lists them):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1))) -Doctor xai
#
# Downloads scripts/install/Install.ps1, its failure catalog and the install doctor into
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

    Write-Host "InstallZero: downloading the rl-roboracer installer ($branch) to $dir ..."
    New-Item -ItemType Directory -Force $dir | Out-Null
    Invoke-WebRequest -UseBasicParsing "$raw/scripts/install/Install.ps1" -OutFile (Join-Path $dir 'Install.ps1')

    function Get-RepoFiles([string]$Path) {
        # Assigned first: Windows PowerShell 5.1 emits the JSON array as one object.
        $entries = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$repo/contents/${Path}?ref=$branch"
        foreach ($e in $entries) {
            if ($e.type -eq 'dir' -and $e.name -notin 'eval', 'assets') { Get-RepoFiles $e.path }
            elseif ($e.type -eq 'file') { $e.path }
        }
    }
    function Save-RepoFolder([string]$Path, [string]$Dest) {
        $files = @(Get-RepoFiles $Path)
        if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
        foreach ($f in $files) {
            $to = Join-Path $Dest (($f.Substring($Path.Length + 1)) -replace '/', '\')
            New-Item -ItemType Directory -Force (Split-Path $to) | Out-Null
            Invoke-WebRequest -UseBasicParsing "$raw/$f" -OutFile $to
        }
    }

    # The failure catalog and the doctor diagnose failures, including ones
    # before the repo is cloned. Without them the install still runs.
    try { Save-RepoFolder 'scripts/install/catalog' (Join-Path $dir 'catalog') }
    catch { Write-Host "The failure catalog could not be downloaded ($($_.Exception.Message)); known failures are recognised once the repo is cloned." -ForegroundColor Yellow }
    try { Save-RepoFolder 'scripts/install/doctor' (Join-Path $dir 'doctor') }
    catch { Write-Host "The InstallZero doctor could not be downloaded ($($_.Exception.Message)); continuing without it." -ForegroundColor Yellow }

    $pass = @($args)
    if (-not ($pass | Where-Object { $_ -match '^-Branch(:|$)' })) { $pass += @('-Branch', $branch) }
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir 'Install.ps1') @pass
} @args
