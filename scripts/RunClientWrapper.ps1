<#
.SYNOPSIS
    Launch and supervise the Unity gym client. Restarts it if the process
    becomes unresponsive or exits.

.DESCRIPTION
    By default, looks for a single .exe at the top of unity/Builds/latest/
    (the destination of scripts/PromoteLatestBuild.ps1). You can override
    with -Path "<full exe path>" if you need to run a different binary.

.PARAMETER Path
    Optional explicit path to the Unity .exe. If omitted, the script globs
    unity/Builds/latest/*.exe and expects exactly one match.

.PARAMETER PollSeconds
    How often (seconds) to check whether the process is still responding.
    Default 5.
#>
[CmdletBinding()]
param(
    [string]$Path,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = 'Stop'

if (-not $Path) {
    $latestDir = Join-Path $PSScriptRoot '..\unity\Builds\latest'
    if (-not (Test-Path -LiteralPath $latestDir)) {
        throw "No build at $latestDir. Build from Unity into a subfolder of unity/Builds/, then run scripts/PromoteLatestBuild.ps1."
    }
    $exes = @(Get-ChildItem -LiteralPath $latestDir -Filter '*.exe' -File)
    if ($exes.Count -eq 0) {
        throw "No .exe found at top of $latestDir."
    }
    if ($exes.Count -gt 1) {
        $names = ($exes | ForEach-Object { $_.Name }) -join ', '
        throw "Expected exactly one .exe in $latestDir, found $($exes.Count): $names. Pass -Path to disambiguate."
    }
    $Path = $exes[0].FullName
}

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Binary not found: $Path"
}

# `Get-Process -Name` wants the executable name without the .exe extension.
$processName = [System.IO.Path]::GetFileNameWithoutExtension($Path)

Start-Process -FilePath $Path
Write-Host "$processName started ($Path)"

while ($true) {
    $procs = Get-Process -Name $processName -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($p in $procs) {
            if ($p.Responding) {
                Write-Host "working fine"
            } else {
                Write-Host "Status = Not Responding: Kill & Restart..."
                $p.Kill()
                Start-Process -FilePath $Path
            }
        }
    } else {
        Write-Host "Status = Restart..."
        Start-Process -FilePath $Path
    }

    Start-Sleep -Seconds $PollSeconds
}

# Source: https://community.idera.com/database-tools/powershell/ask_the_experts/f/powershell_for_windows-12/7002/how-to-detect-process-not-responding
