# InstallZero one-line installer for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1 | iex
#
# With installzero install options (installzero install --help lists them):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1))) --model google
#
# Downloads the InstallZero CLI (the version below, checked against the
# release's SHA256SUMS) into %LOCALAPPDATA%\installzero\bin and runs
# installzero install frenb/rl-roboracer, which gets the repo into
# %USERPROFILE%\rl-roboracer\rl-roboracer (git clone, or GitHub's archive when
# Git is not installed yet; install.md installs it) and follows its install.md.
# The same CLI is attached to each release as installzero.exe, which does this
# when opened. RL_INSTALL_BRANCH picks another branch, INSTALLZERO_VERSION
# another InstallZero release (or latest).

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $project = 'frenb/rl-roboracer'
    if ($env:RL_INSTALL_BRANCH) { $project += "@$env:RL_INSTALL_BRANCH" }
    $version = if ($env:INSTALLZERO_VERSION) { $env:INSTALLZERO_VERSION } else { 'v0.1.9' }

    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { 'arm64' } else { 'amd64' }
    $asset = "installzero-windows-$arch.exe"
    $release = if ($version -eq 'latest') { 'https://github.com/InstallZero/installzero/releases/latest/download' } else { "https://github.com/InstallZero/installzero/releases/download/$version" }
    $binDir = Join-Path $env:LOCALAPPDATA 'installzero\bin'
    $bin = Join-Path $binDir 'installzero.exe'

    Write-Host "InstallZero: downloading installzero $version ($asset) ..."
    New-Item -ItemType Directory -Force $binDir | Out-Null
    $tmp = Join-Path $binDir "installzero.$PID.download"
    Invoke-WebRequest -UseBasicParsing "$release/$asset" -OutFile $tmp
    $sums = (Invoke-WebRequest -UseBasicParsing "$release/SHA256SUMS").Content
    if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
    $want = ($sums -split "`n" | Where-Object { $_ -match "^([0-9a-f]{64})\s+\*?$([regex]::Escape($asset))\s*$" } | ForEach-Object { $Matches[1] }) | Select-Object -First 1
    $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
    if (-not $want -or $got -ne $want) {
        Remove-Item $tmp -Force
        throw "installzero $version ($asset): sha256 $got does not match the release's SHA256SUMS ($want)"
    }
    Move-Item -Force $tmp $bin

    & $bin install $project @args
    Write-Host "installzero is in $bin; run it again with the same arguments to resume, or $bin uninstall rl-roboracer to remove what it installed."
} @args
