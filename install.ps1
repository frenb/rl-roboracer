# InstallZero one-line installer for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1 | iex
#
# With installer options (Get-Help on the downloaded Install.ps1 lists them):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1))) -Doctor xai
#
# Downloads scripts/install/Install.ps1, its requirements, the project's failure catalog entries and
# installzero.json (the pinned InstallZero engine, which Install.ps1 downloads and verifies) into
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
    Invoke-WebRequest -UseBasicParsing "$raw/scripts/install/requirements.json" -OutFile (Join-Path $dir 'requirements.json')
    Invoke-WebRequest -UseBasicParsing "$raw/scripts/install/installzero.json" -OutFile (Join-Path $dir 'installzero.json')

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

    # The project's own failure catalog entries, for failures before the repo
    # is cloned. Without them the install still runs.
    try { Save-RepoFolder 'scripts/install/catalog' (Join-Path $dir 'catalog') }
    catch { Write-Host "The project's failure catalog could not be downloaded ($($_.Exception.Message)); its entries are recognised once the repo is cloned." -ForegroundColor Yellow }
    Remove-Item (Join-Path $dir 'doctor') -Recurse -Force -ErrorAction SilentlyContinue

    $pass = @($args)
    if (-not ($pass | Where-Object { $_ -match '^-Branch(:|$)' })) { $pass += @('-Branch', $branch) }
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir 'Install.ps1') @pass
} @args
