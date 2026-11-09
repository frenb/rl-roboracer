# InstallZero one-line installer for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1 | iex
#
# With installzero install options (installzero install --help lists them):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/install.ps1))) --model google
#
# Downloads the InstallZero CLI (the version below, checked against the
# release's SHA256SUMS) into %LOCALAPPDATA%\installzero\bin, gets the repo into
# %USERPROFILE%\rl-roboracer\rl-roboracer (git clone, or the branch's zip when
# Git is not installed yet; install.md installs it), and runs installzero
# install there, which follows install.md. RL_INSTALL_BRANCH picks another
# branch, INSTALLZERO_VERSION another InstallZero release (or latest).

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'frenb/rl-roboracer'
    $branch = if ($env:RL_INSTALL_BRANCH) { $env:RL_INSTALL_BRANCH } else { 'main' }
    $version = if ($env:INSTALLZERO_VERSION) { $env:INSTALLZERO_VERSION } else { 'v0.1.8' }
    $installDir = Join-Path $env:USERPROFILE 'rl-roboracer'
    $repoDir = Join-Path $installDir 'rl-roboracer'

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

    if (Test-Path (Join-Path $repoDir 'install.md')) {
        Write-Host "Using the rl-roboracer checkout in $repoDir"
    } elseif (Test-Path (Join-Path $repoDir '.git')) {
        Write-Host "Updating the rl-roboracer checkout in $repoDir (it has no install.md yet) ..."
        git -C $repoDir pull --ff-only
        if (-not (Test-Path (Join-Path $repoDir 'install.md'))) { throw "$repoDir has no install.md after git pull; update it to $branch and run this again." }
    } elseif (Test-Path $repoDir) {
        throw "$repoDir exists but holds no rl-roboracer checkout; move it aside and run this again."
    } elseif (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "Cloning $repo ($branch) into $repoDir ..."
        New-Item -ItemType Directory -Force $installDir | Out-Null
        git clone --branch $branch "https://github.com/$repo.git" $repoDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE)" }
    } else {
        Write-Host "Git is not installed yet (install.md installs it); downloading $repo ($branch) as a zip into $repoDir ..."
        $zip = Join-Path $env:TEMP "rl-roboracer-$PID.zip"
        $unz = Join-Path $env:TEMP "rl-roboracer-$PID"
        Invoke-WebRequest -UseBasicParsing "https://codeload.github.com/$repo/zip/refs/heads/$branch" -OutFile $zip
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $unz)
        New-Item -ItemType Directory -Force $installDir | Out-Null
        Move-Item (Get-ChildItem $unz -Directory | Select-Object -First 1).FullName $repoDir
        Remove-Item $zip, $unz -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "This copy is not a git checkout; to update it later, git clone https://github.com/$repo.git" -ForegroundColor Yellow
    }

    Push-Location $repoDir
    try { & $bin install @args }
    finally { Pop-Location }
    Write-Host "installzero is in $bin; from $repoDir run it again to resume, or $bin uninstall rl-roboracer to remove what it installed."
} @args
