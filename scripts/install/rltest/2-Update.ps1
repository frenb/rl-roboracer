<#
.SYNOPSIS
    Step 2, run as rltest: install or update the stack from the staged code.

.DESCRIPTION
    Runs the staged Install.ps1 with -Clients 2 (it fast-forwards the clone
    to the staged snapshot and leaves a running stack alone), logs to
    results\, then confirms the clone is at the staged commit.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\2-Update.ps1
#>
[CmdletBinding()]
param([ValidateRange(1, 4)][int]$Clients = 2)

$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'rltest') { Write-Host "Warning: step 2 is meant for the rltest account (you are $env:USERNAME)." -ForegroundColor Yellow }
$stage = $PSScriptRoot
$results = Join-Path $stage 'results'
New-Item -ItemType Directory -Force $results | Out-Null
$log = Join-Path $results "install-update-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

& powershell -ExecutionPolicy Bypass -File (Join-Path $stage 'Install.ps1') -Clients $Clients *>&1 | Tee-Object $log

$clone = Join-Path $env:USERPROFILE 'rl-roboracer\rl-roboracer'
$want = "$(& git bundle list-heads (Join-Path $stage 'rl-roboracer.bundle') refs/heads/rltest 2>$null)".Split(' ')[0]
$have = "$(& git -C $clone rev-parse HEAD 2>$null)".Trim()
Write-Host ''
if ($want -and $have -eq $want) {
    Write-Host "Code is up to date: the clone is at the staged commit $($want.Substring(0, 7))." -ForegroundColor Green
    Write-Host 'Next: powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\3-Probe.ps1'
} else {
    Write-Host "The clone is at $(if ($have) { $have.Substring(0, 7) } else { '(none)' }), not the staged commit $(if ($want) { $want.Substring(0, 7) } else { '(unknown)' })." -ForegroundColor Red
    Write-Host "Look for 'The repo has local edits' in $log, and send me that part of the log."
    exit 1
}
