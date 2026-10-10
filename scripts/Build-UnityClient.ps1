<#
.SYNOPSIS
    Build the Unity client (Windows 64-bit player) from unity/ without
    opening the editor by hand.

.DESCRIPTION
    Runs the editor version named in unity/ProjectSettings/ProjectVersion.txt
    with -quit -executeMethod BuildClient.Build (unity/Assets/Editor/
    BuildClient.cs) and builds the scenes enabled in Build Settings into
    unity/Builds/<Name>/.

    The editor runs windowed (minimized), not with -batchmode. Unity 2020.3
    with a Unity Personal licence refuses batch mode ("Missing or bad
    username or password"); the windowed editor picks up the licence Unity
    Hub signed in with. Unity Hub does not need to be running.

    The project must not be open in the editor at the same time.

    Typical workflow:
      .\scripts\Build-UnityClient.ps1 -Name wCourseJetRacer2026.10.08-v50 -Promote -Zip

.PARAMETER Name
    Build folder name under unity/Builds/, also used for the release zip
    (roboracer-gym-<Name>.zip). Default: the current date and time.

.PARAMETER Promote
    After a successful build, replace unity/Builds/latest/ with it (what
    RunClientWrapper.ps1 and RunNClients.ps1 launch). Refuses while a
    client is running from unity/Builds/latest/.

.PARAMETER Zip
    Also write unity/Builds/release/roboracer-gym-<Name>.zip and
    "<zip>.sha256" in the layout scripts/install/Install.ps1 expects (the
    .exe at the top level of the zip). Player.log and Unity's
    *_DoNotShip debug folders are left out.

.PARAMETER UnityExe
    Path to Unity.exe. Default: the Unity Hub install of the project's
    editor version.

.PARAMETER TimeoutMinutes
    Kill the editor if the build has not finished by then. Default 60.

    Exit codes: 0 built, 1 build failed, 2 could not start.
#>
[CmdletBinding()]
param(
    [string]$Name = (Get-Date -Format 'yyyy.MM.dd-HHmm'),
    [switch]$Promote,
    [switch]$Zip,
    [string]$UnityExe = '',
    [int]$TimeoutMinutes = 60
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$repoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$project    = Join-Path $repoRoot 'unity'
$buildsDir  = Join-Path $project 'Builds'
$outDir     = Join-Path $buildsDir $Name
$latestDir  = Join-Path $buildsDir 'latest'
$releaseDir = Join-Path $buildsDir 'release'
$log        = Join-Path $buildsDir "$Name.build.log"

function Fail([string]$Message, [int]$Code) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit $Code
}

if ($Name -in @('latest', 'instances', 'release')) { Fail "-Name '$Name' is reserved under unity\Builds" 2 }
if (Test-Path $outDir) { Fail "unity\Builds\$Name already exists; pick another -Name or delete it" 2 }

if (-not $UnityExe) {
    $versionLine = Select-String -Path (Join-Path $project 'ProjectSettings\ProjectVersion.txt') -Pattern '^m_EditorVersion:\s*(\S+)'
    $editorVersion = $versionLine.Matches[0].Groups[1].Value
    $UnityExe = Join-Path $env:ProgramFiles "Unity\Hub\Editor\$editorVersion\Editor\Unity.exe"
    if (-not (Test-Path $UnityExe)) { Fail "Unity $editorVersion not found at $UnityExe; install it with Unity Hub or pass -UnityExe" 2 }
}

# The editor holds Temp\UnityLockfile open exclusively while the project is open.
$lockFile = Join-Path $project 'Temp\UnityLockfile'
if (Test-Path $lockFile) {
    try { [IO.File]::Open($lockFile, 'Open', 'ReadWrite', 'None').Dispose() }
    catch { Fail 'the project is open in the Unity editor; close it first' 2 }
}

if ($Promote) {
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($latestDir, 'OrdinalIgnoreCase') })
    if ($running) { Fail "a Unity client is running from unity\Builds\latest (pid $($running.Id -join ', ')); stop it (scripts\Stop-Stack.ps1) before -Promote" 2 }
}

Write-Host "Building unity\Builds\$Name with $UnityExe"
Write-Host "  log: $log"
$started = Get-Date
$unityArgs = @('-quit', '-projectPath', "`"$project`"", '-executeMethod', 'BuildClient.Build',
               '-buildOutput', "`"$outDir`"", '-logFile', "`"$log`"")
$proc = Start-Process -FilePath $UnityExe -ArgumentList $unityArgs -WindowStyle Minimized -PassThru
if (-not $proc.WaitForExit($TimeoutMinutes * 60 * 1000)) {
    $proc.Kill()
    Fail "build did not finish within $TimeoutMinutes minutes; see $log" 1
}
$elapsed = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

$exe = Join-Path $outDir 'robotaxi gym level 1.exe'
if ($proc.ExitCode -ne 0 -or -not (Test-Path $exe)) {
    Write-Host "Build failed after $elapsed min (editor exit code $($proc.ExitCode)). From the log:" -ForegroundColor Red
    Select-String -Path $log -Pattern 'error CS\d+|BuildClient:|Build Finished, Result|executeMethod|licen[cs]e' -ErrorAction SilentlyContinue |
        Select-Object -Last 20 | ForEach-Object { "  $($_.Line.Trim())" }
    exit 1
}
$sizeMb = [math]::Round((Get-ChildItem $outDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Built unity\Builds\$Name ($sizeMb MB) in $elapsed min" -ForegroundColor Green

if ($Promote) {
    if (Test-Path $latestDir) { Remove-Item $latestDir -Recurse -Force }
    Copy-Item -Path $outDir -Destination $latestDir -Recurse
    Write-Host "Promoted to unity\Builds\latest"
}

if ($Zip) {
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Force $releaseDir | Out-Null
    $zipPath = Join-Path $releaseDir "roboracer-gym-$Name.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    $archive = [IO.Compression.ZipFile]::Open($zipPath, 'Create')
    try {
        Get-ChildItem $outDir -Recurse -File |
            Where-Object { $_.Name -ne 'Player.log' -and $_.FullName -notmatch '_DoNotShip\\' } |
            ForEach-Object {
                $entry = $_.FullName.Substring($outDir.Length + 1).Replace('\', '/')
                [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $_.FullName, $entry, 'Optimal')
            }
    } finally { $archive.Dispose() }
    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    Set-Content -Path "$zipPath.sha256" -Value "$hash  $(Split-Path $zipPath -Leaf)" -Encoding ASCII
    Write-Host "Wrote unity\Builds\release\$(Split-Path $zipPath -Leaf) ($([math]::Round((Get-Item $zipPath).Length / 1MB)) MB) and .sha256"
}
