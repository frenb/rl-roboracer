<#
.SYNOPSIS
    Promote the most-recently-modified Unity build folder under unity/Builds/
    to "latest", archiving any prior "latest" to its modification timestamp.

.DESCRIPTION
    A Unity Windows build is a folder, not a single file: <game>.exe sits next
    to <game>_Data/, UnityPlayer.dll, MonoBleedingEdge/, etc. So we promote
    *folders*, not files.

    Workflow:
      1. In Unity: File > Build Settings > Build to a fresh subfolder of
         unity/Builds/ (e.g. unity/Builds/2026-05-02-multiagent/).
      2. Run this script from anywhere:  .\scripts\PromoteLatestBuild.ps1
      3. The previous unity/Builds/latest/ (if any) is renamed to its own
         LastWriteTime stamp (yyyy-MM-dd_HH-mm-ss), and the freshly-built
         folder is renamed to latest/.

    RunClientWrapper.ps1 should point at unity/Builds/latest/<exe-name>.exe.

.PARAMETER BuildsDir
    Path to the Unity Builds folder. Defaults to ../unity/Builds relative
    to this script.
#>
[CmdletBinding()]
param(
    [string]$BuildsDir = (Join-Path $PSScriptRoot '..\unity\Builds')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $BuildsDir)) {
    New-Item -ItemType Directory -Path $BuildsDir -Force | Out-Null
    Write-Host "Created $BuildsDir"
    Write-Host "No builds present yet. Build from Unity into a subfolder of this directory, then re-run."
    return
}

$BuildsDir = (Resolve-Path -LiteralPath $BuildsDir).Path
$latestPath = Join-Path $BuildsDir 'latest'

# Find candidate: newest subdirectory that isn't already 'latest'.
$candidate = Get-ChildItem -LiteralPath $BuildsDir -Directory |
    Where-Object { $_.Name -ne 'latest' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $candidate) {
    Write-Host "No promotable build folder found in $BuildsDir."
    Write-Host "Build from Unity into a subfolder of this directory first."
    return
}

# If 'latest' already exists, decide whether to archive it.
if (Test-Path -LiteralPath $latestPath) {
    $existingLatest = Get-Item -LiteralPath $latestPath
    if ($existingLatest.LastWriteTime -ge $candidate.LastWriteTime) {
        Write-Host "Existing 'latest/' (modified $($existingLatest.LastWriteTime)) is at least as new as $($candidate.Name) (modified $($candidate.LastWriteTime)). Nothing to promote."
        return
    }

    $stamp = $existingLatest.LastWriteTime.ToString('yyyy-MM-dd_HH-mm-ss')
    $archiveName = $stamp
    $i = 1
    while (Test-Path -LiteralPath (Join-Path $BuildsDir $archiveName)) {
        $archiveName = "${stamp}_$i"
        $i++
    }

    Write-Host "Archiving existing 'latest/' -> $archiveName/"
    Rename-Item -LiteralPath $latestPath -NewName $archiveName
}

Write-Host "Promoting $($candidate.Name)/ -> latest/"
Rename-Item -LiteralPath $candidate.FullName -NewName 'latest'

# Surface the exe path for convenience.
$exe = Get-ChildItem -LiteralPath $latestPath -Filter '*.exe' -File | Select-Object -First 1
if ($exe) {
    Write-Host "Done. Launch with:"
    Write-Host "  & '$($exe.FullName)'"
} else {
    Write-Host "Done. unity/Builds/latest/ is now in place (no .exe found at top level - check Unity build output)."
}
