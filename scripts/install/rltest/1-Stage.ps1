<#
.SYNOPSIS
    Step 1, run in your own account: stage the current working tree for rltest.

.DESCRIPTION
    Runs Prepare-RlTest.ps1 (snapshot bundle, installer, the probe from the
    InstallZero checkout, models, and the step 2-4 scripts in
    C:\Users\Public\rl-roboracer-test). Then sign in as rltest and run step 2.
    The installer uses the InstallZero engine pinned in installzero.json;
    -LocalEngine stages the checkout's instead.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install\rltest\1-Stage.ps1
#>
[CmdletBinding()]
param([string[]]$LocalModels = @('gemma', 'gpt-oss', 'qwen', 'granite'), [switch]$LocalEngine)

$ErrorActionPreference = 'Stop'
if ($env:USERNAME -eq 'rltest') { throw 'Run step 1 in your own account, not as rltest.' }
& (Join-Path $PSScriptRoot '..\Prepare-RlTest.ps1') -LocalModels $LocalModels -LocalEngine:$LocalEngine
Write-Host ''
Write-Host 'Next: sign in as rltest and run' -ForegroundColor Green
Write-Host '  powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\2-Update.ps1'
