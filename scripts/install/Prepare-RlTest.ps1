<#
.SYNOPSIS
    Stage the current working tree, the installer and the doctor's model
    files where the test account (rltest) can use them.

.DESCRIPTION
    Run it in your own account. It writes to C:\Users\Public, which every
    account can read:

      <StageDir>\rl-roboracer.bundle   the working tree as it is now,
                                       including uncommitted and untracked
                                       files (not gitignored ones such as the
                                       env file), as branch "rltest". Made with a
                                       temporary git index: your branch,
                                       index and working tree are untouched.
      <StageDir>\Install.ps1, requirements.json, installzero.json,
      Reset-TestAccount.ps1, catalog\  the installer, its GPU requirements,
                                       the pinned InstallZero engine (the
                                       installer downloads it), the reset
                                       script and the project's own failure
                                       catalog entries
      <StageDir>\installzero\          agent\ and doctor\ from the InstallZero
                                       checkout, for the probe (step 3)
      <StageDir>\installzero-engine\   with -LocalEngine only: the checkout's
                                       doctor, catalog and schemas, used by
                                       the installer instead of the pinned
                                       release
      <StageDir>\RUN-TEST.md           docs/rltest-clean-test.md
      <DoctorAssetsDir>                llama.cpp builds (unpacked) and the
                                       chosen local models

    The release assets (gym zips and roboracer-demos-donut.zip, each with
    its sha256 file) are expected in <StageDir> already; it warns when one
    is missing.

.PARAMETER LocalModels
    Local doctor models to stage: any of gpt-oss, gemma, qwen, granite, or
    all, or none. Default gemma, gpt-oss (about 16 GB). Copies are skipped
    when the file is already there with the same size.

.PARAMETER TestAccount
    Account given modify rights on the staged folders (it unpacks llama.cpp
    logs there and saves its results). Default rltest.

.PARAMETER InstallZeroDir
    The InstallZero checkout. Default: installzero beside this repo.

.PARAMETER LocalEngine
    Test the checkout's doctor and catalog instead of the released engine
    pinned in installzero.json.

.EXAMPLE
    .\Prepare-RlTest.ps1
    Stages the code with the gemma and gpt-oss local models.

.EXAMPLE
    .\Prepare-RlTest.ps1 -LocalModels none
    Code only, for testing hosted doctor models.
#>
[CmdletBinding()]
param(
    [string]$StageDir = 'C:\Users\Public\rl-roboracer-test',
    [string]$DoctorAssetsDir = 'C:\Users\Public\doctor-assets',
    [string[]]$LocalModels = @('gemma', 'gpt-oss'),
    [string]$SourceAssets = '',
    [string]$TestAccount = 'rltest',
    [string]$InstallZeroDir = '',
    [switch]$LocalEngine
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $SourceAssets) { $SourceAssets = Join-Path (Split-Path $repo -Parent) 'doctor-assets' }
if (-not $InstallZeroDir) { $InstallZeroDir = Join-Path (Split-Path $repo -Parent) 'installzero' }
if (-not (Test-Path (Join-Path $InstallZeroDir 'doctor\Llm.ps1'))) { throw "No InstallZero checkout at $InstallZeroDir (it needs doctor\ and agent\); pass -InstallZeroDir." }
$InstallZeroDir = (Resolve-Path $InstallZeroDir).Path
$LocalModels = @($LocalModels | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
$modelFiles = [ordered]@{ 'gpt-oss' = 'gpt-oss-20b-MXFP4.gguf'; gemma = 'gemma-4-E4B-it-Q4_K_M.gguf'; qwen = 'Qwen3.5-9B-Q4_K_M.gguf'; granite = 'granite-4.1-8b-Q4_K_M.gguf' }
if ($LocalModels -contains 'all') { $LocalModels = @($modelFiles.Keys) }
if ($LocalModels -contains 'none') { $LocalModels = @() }
foreach ($m in $LocalModels) { if (-not $modelFiles.Contains($m)) { throw "Unknown local model '$m'; use $(@($modelFiles.Keys) -join ', '), all or none." } }

function Invoke-Git([string[]]$GitArgs) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = & git -C $repo @GitArgs 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed: $($out -join ' ')" }
    $out
}

New-Item -ItemType Directory -Force $StageDir | Out-Null

# 1. Snapshot of the working tree, as branch rltest in a bundle.
Write-Host 'Snapshotting the working tree ...' -ForegroundColor Cyan
$index = Join-Path $env:TEMP "rltest-index-$PID"
$bare = Join-Path $env:TEMP "rltest-bare-$PID"
try {
    $env:GIT_INDEX_FILE = $index
    [void](Invoke-Git @('read-tree', 'HEAD'))
    [void](Invoke-Git @('add', '-A'))
    $tree = (Invoke-Git @('write-tree')) | Select-Object -Last 1
    Remove-Item Env:\GIT_INDEX_FILE
    $secretFiles = @(Invoke-Git @('ls-tree', '-r', '--name-only', $tree) | Where-Object { $_ -match '(^|/)\.env$|(^|/)secrets/(?!\.git(ignore|keep)$)|\.(pem|pfx|p12)$' })
    if ($secretFiles) { throw "The snapshot would contain files with secrets: $($secretFiles -join ', '). Add them to .gitignore first." }
    $head = (Invoke-Git @('rev-parse', '--short', 'HEAD')) | Select-Object -Last 1
    $branch = (Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')) | Select-Object -Last 1
    $changed = @(Invoke-Git @('status', '--porcelain')).Count
    $commit = (Invoke-Git @('commit-tree', $tree, '-p', 'HEAD', '-m', "rltest snapshot of $branch@$head plus $changed uncommitted change(s), $(Get-Date -Format s)")) | Select-Object -Last 1
    $oldEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & git init --bare --quiet $bare 2>&1 | Out-Null
    & git -C $repo push --quiet $bare "${commit}:refs/heads/rltest" 2>&1 | Out-Null
    $pushed = $LASTEXITCODE
    $bundle = Join-Path $StageDir 'rl-roboracer.bundle'
    & git -C $bare bundle create $bundle --all 2>&1 | Out-Null
    $bundled = $LASTEXITCODE
    $ErrorActionPreference = $oldEap
    if ($pushed -ne 0 -or $bundled -ne 0) { throw 'creating the bundle failed' }
    Write-Host "  $bundle : branch rltest = $branch@$head + $changed uncommitted change(s) ($([math]::Round((Get-Item $bundle).Length / 1MB)) MB)"
} finally {
    Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue
    Remove-Item $index -Force -ErrorAction SilentlyContinue
    Remove-Item $bare -Recurse -Force -ErrorAction SilentlyContinue
}

# 2. Installer, reset script, project catalog, InstallZero, instructions.
Write-Host 'Copying the installer, reset script, failure catalog, InstallZero and instructions ...' -ForegroundColor Cyan
Copy-Item (Join-Path $PSScriptRoot 'Install.ps1'), (Join-Path $PSScriptRoot 'requirements.json'), (Join-Path $PSScriptRoot 'installzero.json'), (Join-Path $PSScriptRoot 'Reset-TestAccount.ps1') $StageDir -Force
Get-ChildItem (Join-Path $PSScriptRoot 'rltest') -File | Where-Object Name -match '^[2-9]-.*\.ps1$' | Copy-Item -Destination $StageDir -Force
Remove-Item (Join-Path $StageDir 'doctor') -Recurse -Force -ErrorAction SilentlyContinue
function Copy-Folder([string]$From, [string]$To, [string[]]$Exclude = @()) {
    if (Test-Path $To) { Remove-Item $To -Recurse -Force }
    New-Item -ItemType Directory -Force $To | Out-Null
    Get-ChildItem $From -Exclude $Exclude | Copy-Item -Destination $To -Recurse -Force
}
$izStage = Join-Path $StageDir 'installzero'
Copy-Folder (Join-Path $InstallZeroDir 'agent') (Join-Path $izStage 'agent')
Copy-Folder (Join-Path $InstallZeroDir 'doctor') (Join-Path $izStage 'doctor') @('assets')
$izHead = "$(& git -C $InstallZeroDir rev-parse --short HEAD 2>$null)".Trim()
Write-Host "  $izStage : the probe from $InstallZeroDir@$izHead"
$engineStage = Join-Path $StageDir 'installzero-engine'
Remove-Item $engineStage -Recurse -Force -ErrorAction SilentlyContinue
if ($LocalEngine) {
    Copy-Folder (Join-Path $InstallZeroDir 'doctor') (Join-Path $engineStage 'doctor') @('assets', 'eval')
    Copy-Folder (Join-Path $InstallZeroDir 'catalog') (Join-Path $engineStage 'catalog')
    Copy-Folder (Join-Path $InstallZeroDir 'schemas') (Join-Path $engineStage 'schemas')
    "local`n$izHead" | Set-Content (Join-Path $engineStage 'VERSION') -Encoding ASCII
    Write-Host "  $engineStage : the engine from the checkout, instead of the pinned release"
} else {
    Write-Host "  engine: $((Get-Content (Join-Path $PSScriptRoot 'installzero.json') -Raw | ConvertFrom-Json).version) from installzero.json (the installer downloads it)"
}
$catalogStage = Join-Path $StageDir 'catalog'
if (Test-Path $catalogStage) { Remove-Item $catalogStage -Recurse -Force }
New-Item -ItemType Directory -Force $catalogStage | Out-Null
Copy-Item (Join-Path $PSScriptRoot 'catalog\*.json') $catalogStage -Force
Copy-Item (Join-Path $repo 'docs\rltest-clean-test.md') (Join-Path $StageDir 'RUN-TEST.md') -Force
Remove-Item (Join-Path $StageDir 'RUN-TEST.txt') -ErrorAction SilentlyContinue
foreach ($asset in 'roboracer-gym-wCourseJetRacer2026.09.20-v13.zip', 'roboracer-gym-wCourseJetRacer2026.09.28-v49.zip', 'roboracer-demos-donut.zip') {
    if (-not (Test-Path (Join-Path $StageDir $asset))) { Write-Host "  WARNING: $asset is missing from $StageDir; the install will try to download it from the GitHub release." -ForegroundColor Yellow }
}

# 3. Doctor assets: llama.cpp builds (unpacked once here) and local models.
Write-Host 'Staging doctor assets ...' -ForegroundColor Cyan
New-Item -ItemType Directory -Force $DoctorAssetsDir | Out-Null
$wanted = @(Get-ChildItem $SourceAssets -Filter '*.zip' -File) + @($LocalModels | ForEach-Object { Get-Item (Join-Path $SourceAssets $modelFiles[$_]) -ErrorAction SilentlyContinue })
$missing = @($LocalModels | Where-Object { -not (Test-Path (Join-Path $SourceAssets $modelFiles[$_])) })
if ($missing) { Write-Host "  WARNING: not in $SourceAssets : $(@($missing | ForEach-Object { $modelFiles[$_] }) -join ', ')" -ForegroundColor Yellow }
$need = ($wanted | Where-Object { $d = Join-Path $DoctorAssetsDir $_.Name; -not ((Test-Path $d) -and (Get-Item $d).Length -eq $_.Length) } | Measure-Object Length -Sum).Sum
$free = (Get-PSDrive ($DoctorAssetsDir.Substring(0, 1))).Free
if ($need -gt $free - 10GB) { throw "Copying $([math]::Round($need / 1GB, 1)) GB of doctor assets would leave less than 10 GB free on $($DoctorAssetsDir.Substring(0, 2))." }
foreach ($f in $wanted) {
    $dest = Join-Path $DoctorAssetsDir $f.Name
    if ((Test-Path $dest) -and (Get-Item $dest).Length -eq $f.Length) { continue }
    Write-Host "  copying $($f.Name) ($([math]::Round($f.Length / 1GB, 2)) GB)"
    Copy-Item $f.FullName $dest -Force
}
. (Join-Path $InstallZeroDir 'doctor\Llm.ps1')
foreach ($b in 'cuda', 'vulkan', 'cpu') {
    try { [void](Install-LlamaBackend $DoctorAssetsDir $b); Write-Host "  llama.cpp $b build ready" } catch { Write-Host "  llama.cpp $b build not staged: $_" -ForegroundColor Yellow }
}
$staleLogs = @(Get-ChildItem $DoctorAssetsDir -Filter 'llama-server-*.log*' -File -ErrorAction SilentlyContinue)
$staleLogs | Remove-Item -Force -ErrorAction SilentlyContinue

# 4. Let the test account write logs and results there.
$ErrorActionPreference = 'Continue'
foreach ($d in $StageDir, $DoctorAssetsDir) {
    $out = & icacls.exe $d /grant "${TestAccount}:(OI)(CI)M" /Q 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "  WARNING: could not give $TestAccount write access to ${d}: $out" -ForegroundColor Yellow }
}

Write-Host ''
Write-Host 'Staged. Signed in as rltest, follow C:\Users\Public\rl-roboracer-test\RUN-TEST.md, or run the steps:' -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File $StageDir\2-Update.ps1    (install or update, -Clients 2)"
Write-Host "  powershell -ExecutionPolicy Bypass -File $StageDir\3-Probe.ps1     (probe; -Set compare|local|hosted|all, -Runs 2)"
Write-Host "  powershell -ExecutionPolicy Bypass -File $StageDir\4-Results.ps1   (summary; -Facts for recall per fact)"
if ((Split-Path $DoctorAssetsDir -Parent) -ne (Split-Path $StageDir -Parent)) { Write-Host "  (add -DoctorAssets ${DoctorAssetsDir}: it is not beside $StageDir, so the installer will not find it by itself)" }
