<#
.SYNOPSIS
    Step 4, either account: summarize the probe results per model.

.DESCRIPTION
    Reads results\results.jsonl in the staging folder. Per model: runs,
    delivered install.md files, weighted fact recall (average, min, max),
    clean files (no format problems and no failing checks), median time,
    and the blindness audit (refused attempts to reach install files, lines
    of install-file content withheld). -Facts adds, for each fact some
    file missed, how often each model found it.
    Only probe-5 runs (the probe blind to install files) by default;
    -PromptVersion any includes earlier ones.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\4-Results.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\4-Results.ps1 -Since '2026-10-14 22:00' -Facts
#>
[CmdletBinding()]
param(
    [datetime]$Since = [datetime]::MinValue,
    [string]$PromptVersion = 'probe-5',
    [switch]$Facts,
    [string]$ResultsDir = ''
)

$ErrorActionPreference = 'Stop'
if ($PromptVersion -eq 'any') { $PromptVersion = '' }
if (-not $ResultsDir) { $ResultsDir = if (Test-Path (Join-Path $PSScriptRoot 'results\results.jsonl')) { Join-Path $PSScriptRoot 'results' } else { 'C:\Users\Public\rl-roboracer-test\results' } }
$file = Join-Path $ResultsDir 'results.jsonl'
if (-not (Test-Path $file)) { throw "No results in $ResultsDir yet; run step 3." }
$rows = @(Get-Content $file | ForEach-Object { $_ | ConvertFrom-Json } |
    Where-Object { $_.repo -eq 'rl-roboracer' -and (-not $PromptVersion -or $_.prompt_version -eq $PromptVersion) -and [datetime]$_.time -ge $Since })
if (-not $rows) { Write-Host "No rl-roboracer results$(if ($PromptVersion) { " for $PromptVersion" })$(if ($Since -gt [datetime]::MinValue) { " since $Since" })." -ForegroundColor Yellow; return }

Write-Host "rl-roboracer probe results$(if ($PromptVersion) { " ($PromptVersion)" }): $($rows.Count) run(s), $($rows[0].time) to $($rows[-1].time)" -ForegroundColor Cyan
$table = foreach ($g in ($rows | Group-Object model | Sort-Object Name)) {
    $ok = @($g.Group | Where-Object delivered)
    $rec = @($ok | ForEach-Object { [double]$_.recall })
    $secs = @($g.Group | ForEach-Object { [double]$_.probe.stats.seconds } | Sort-Object)
    $clean = @($ok | Where-Object { -not @($_.format_problems).Count -and $_.live_checks -and [int]$_.live_checks.failed -eq 0 }).Count
    [pscustomobject]@{
        Model     = $g.Name -replace ' \(cuda\)', ''
        Runs      = $g.Count
        Delivered = $ok.Count
        Recall    = if ($rec) { '{0:P0}' -f ($rec | Measure-Object -Average).Average } else { '-' }
        Range     = if ($rec) { '{0:P0}-{1:P0}' -f ($rec | Measure-Object -Minimum).Minimum, ($rec | Measure-Object -Maximum).Maximum } else { '-' }
        Clean     = "$clean/$($ok.Count)"
        MedianSec = if ($secs) { [int]$secs[[int][math]::Floor($secs.Count / 2)] } else { 0 }
        Refused   = ($g.Group | ForEach-Object { [int]$_.probe.blind.hidden_attempts } | Measure-Object -Sum).Sum
        Withheld  = ($g.Group | ForEach-Object { [int]$_.probe.blind.withheld_lines } | Measure-Object -Sum).Sum
    }
}
$table | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
Write-Host 'Recall: weighted facts found. Clean: no format problems and no failing checks. Refused / Withheld: the blindness audit (attempts to reach install files, and lines of install-file content kept out of tool output).'

if ($Facts) {
    $ids = @($rows | Where-Object delivered | ForEach-Object { @($_.misses.id) } | Where-Object { $_ } | Select-Object -Unique)
    if (-not $ids) { Write-Host 'Every delivered file found every fact.'; return }
    Write-Host 'Facts missed at least once (share of delivered files that found each):'
    $models = @($rows | Group-Object model | Sort-Object Name)
    $factTable = foreach ($id in $ids) {
        $o = [ordered]@{ Fact = $id }
        foreach ($g in $models) {
            $ok = @($g.Group | Where-Object delivered)
            $o[($g.Name -replace ' \(cuda\)', '' -replace '^\w+:', '')] = if ($ok) { '{0:P0}' -f (1 - @($ok | Where-Object { @($_.misses.id) -contains $id }).Count / $ok.Count) } else { '-' }
        }
        [pscustomobject]$o
    }
    $factTable | Format-Table -AutoSize | Out-String -Width 250 | Write-Host
}
