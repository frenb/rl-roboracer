<#
.SYNOPSIS
    Step 3, run as rltest: probe the running install with a set of models.

.DESCRIPTION
    Runs Invoke-ProbeEval.ps1 -Probe on rl-roboracer for each model, one
    round of all models per run, copies results and transcripts to
    results\ after every probe, then shows the summary (step 4).

    Sets: compare (Opus 5.5, GPT 5.6 sol, gpt-oss-20b), local (the four
    open-weights models), hosted (Opus, GPT 5.6, Grok 4.7, Gemini 3.1 Pro,
    Gemini 4 Argon), all. -Models overrides the set. Hosted models whose API
    key is not set are skipped.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\3-Probe.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\3-Probe.ps1 -Set local -Runs 3
#>
[CmdletBinding()]
param(
    [ValidateSet('compare', 'local', 'hosted', 'all')][string]$Set = 'compare',
    [int]$Runs = 2,
    [string[]]$Models = @()
)

$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'rltest') { Write-Host "Warning: step 3 is meant for the rltest account (you are $env:USERNAME)." -ForegroundColor Yellow }
$stage = $PSScriptRoot
$results = Join-Path $stage 'results'
New-Item -ItemType Directory -Force $results | Out-Null
$clone = Join-Path $env:USERPROFILE 'rl-roboracer\rl-roboracer'
if (-not (Test-Path $clone)) { throw "No install at $clone; run step 2 first." }
$eval = Join-Path $stage 'installzero\agent\eval\Invoke-ProbeEval.ps1'
if (-not (Test-Path $eval)) { throw "No probe staged in $stage\installzero; run step 1 again in your own account." }
$env:RL_DOCTOR_ASSETS = 'C:\Users\Public\doctor-assets'

$local = 'gpt-oss-20b-MXFP4.gguf', 'Qwen3.5-9B-Q4_K_M.gguf', 'gemma-4-E4B-it-Q4_K_M.gguf', 'granite-4.1-8b-Q4_K_M.gguf'
$hosted = 'anthropic:claude-opus-5-5', 'openai:gpt-5.6-sol', 'xai:grok-4.7', 'google:gemini-3.1-pro-preview', 'google:gemini-4-argon'
$list = if ($Models) { $Models } else {
    switch ($Set) {
        'compare' { 'anthropic:claude-opus-5-5', 'openai:gpt-5.6-sol', 'gpt-oss-20b-MXFP4.gguf' }
        'local' { $local }
        'hosted' { $hosted }
        'all' { $local + $hosted }
    }
}

# Hosted models need their key in the environment or the clone's .env.
$keyVars = @{ anthropic = 'ANTHROPIC_API_KEY'; openai = 'OPENAI_API_KEY'; xai = 'XAI_API_KEY'; google = 'GOOGLE_GEMINI_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY' }
$envText = if (Test-Path (Join-Path $clone '.env')) { Get-Content (Join-Path $clone '.env') -Raw } else { '' }
$list = @($list | Where-Object {
    if ($_ -notmatch '^(\w+):') { return $true }
    $vars = $keyVars[$Matches[1]] -split '\|'
    $ok = @($vars | Where-Object { $v = $_; ('Process', 'User', 'Machine' | Where-Object { [Environment]::GetEnvironmentVariable($v, $_) }) -or $envText -match "(?m)^\s*$v\s*=\s*\S" }).Count
    if (-not $ok) { Write-Host "Skipping $_ : set $($vars[0]) first." -ForegroundColor Yellow }
    [bool]$ok
})
if (-not $list) { throw 'No models to run.' }

$started = Get-Date
Write-Host "Probing rl-roboracer: $($list -join ', '); $Runs run(s) each, started $started" -ForegroundColor Cyan
foreach ($i in 1..$Runs) {
    foreach ($m in $list) {
        Write-Host ''
        Write-Host "=== Run $i of $Runs : $m ===" -ForegroundColor Cyan
        $modelArg = if ($m -match '^\w+:') { @('-Remote', $m) } else { @('-Model', $m) }
        & powershell -ExecutionPolicy Bypass -File $eval -Repo rl-roboracer -Probe -Target $clone -NoDocs @modelArg
        Copy-Item "$env:LOCALAPPDATA\installzero\eval\*", "$env:LOCALAPPDATA\installzero\probe\*" $results -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ''
& (Join-Path $stage '4-Results.ps1') -Since $started
