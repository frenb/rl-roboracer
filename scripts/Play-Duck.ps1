<#
.SYNOPSIS
    Drive a trained microduck policy in the Unity MicroDuck player.

.DESCRIPTION
    Picks a microduck model from the console (the newest compatible one by
    default), maps its ONNX from the duck-trainer container path
    (/microduck_rl/logs/...) to the host (saved_models\microduck\...), and
    starts MicroDuck.exe with -policy, so a freshly trained policy plays
    without copying files or rebuilding the player.

    The player simulates the flat-ground walking duck (Velocity-Flat scene,
    61 observations, 14 joint targets). Models of other robots or
    observation layouts, such as the Rollers task, are skipped.

    With the dashboard down it falls back to the newest policy_*.onnx under
    saved_models\microduck\rsl_rl\velocity.

.PARAMETER Model
    Model id (whole, or its first or last 8 characters) or its TRAIN job id.
    Default: newest compatible model.

.PARAMETER Follow
    Keep running: when a newer compatible model appears, restart the player
    with it. Checks every -PollSeconds.

.PARAMETER List
    Print the compatible models and exit.

.PARAMETER Width
    Window width. Default 1280.

.PARAMETER Height
    Window height. Default 720.

.PARAMETER Fullscreen
    Borderless fullscreen instead of a window.

.PARAMETER Exe
    Player to run. Default: $env:MICRODUCK_EXE, else
    ..\spikes\unity-duck\build\MicroDuck\MicroDuck.exe next to this repo.

.PARAMETER Task
    Regex of mjlab task ids the player can run.

.EXAMPLE
    .\scripts\Play-Duck.ps1
.EXAMPLE
    .\scripts\Play-Duck.ps1 -Model 6af2684c
.EXAMPLE
    .\scripts\Play-Duck.ps1 -Follow
.EXAMPLE
    .\scripts\Play-Duck.ps1 -List
#>
param(
    [string]$Model,
    [switch]$Follow,
    [int]$PollSeconds = 30,
    [switch]$List,
    [int]$Width = 1280,
    [int]$Height = 720,
    [switch]$Fullscreen,
    [string]$Exe,
    [string]$Task = '^Mjlab-Velocity-(Flat|Rough)-MicroDuck$',
    # Windows PowerShell tries localhost over IPv6 first and times out on
    # Docker Desktop's port forward; the IPv4 loopback answers at once.
    [string]$DashboardUrl = 'http://127.0.0.1'
)

$ErrorActionPreference = 'Stop'
if ($Follow -and $Model) { throw '-Follow plays the newest model; it cannot be combined with -Model.' }
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Workspace = Split-Path -Parent $RepoRoot
# compose/duck.yml binds ../saved_models/microduck (from the repo root) to this.
$ContainerLogs = '/microduck_rl/logs/'
$HostLogs = Join-Path $Workspace 'saved_models\microduck'
$ObsDim = 61
$ActDim = 14

if (-not $Exe) {
    $Exe = if ($env:MICRODUCK_EXE) { $env:MICRODUCK_EXE } else {
        Join-Path $Workspace 'spikes\unity-duck\build\MicroDuck\MicroDuck.exe'
    }
}
if (-not (Test-Path $Exe)) {
    throw "MicroDuck player not found at $Exe. Build it (spikes\unity-duck, DuckBuild.Build) or pass -Exe."
}

function ConvertTo-HostPath([string]$path) {
    if (-not $path -or -not $path.StartsWith($ContainerLogs)) { return $null }
    return Join-Path $HostLogs ($path.Substring($ContainerLogs.Length) -replace '/', '\')
}

# Compatible models, newest first, each with the host path of its ONNX.
function Get-DuckModels {
    $rows = Invoke-RestMethod "$DashboardUrl/get_models?force=true" -TimeoutSec 30
    $rows |
        Where-Object { $_.robot -eq 'microduck' -and $_.onnx_location -and $_.task_id -match $Task } |
        Where-Object { (-not $_.obs_dim -or $_.obs_dim -eq $ObsDim) -and (-not $_.act_dim -or $_.act_dim -eq $ActDim) } |
        ForEach-Object {
            $onnx = ConvertTo-HostPath $_.onnx_location
            if ($onnx -and (Test-Path $onnx)) {
                $stamp = if ($_.updated_at) { $_.updated_at } else { $_.create_date }
                $short = ($_.task_id -replace '^Mjlab-', '' -replace '-MicroDuck$', '')
                $iters = if ($null -ne $_.training_iterations) { " ($([int]$_.training_iterations + 1) it)" } else { '' }
                [pscustomobject]@{
                    Id = [string]$_._id
                    JobId = [string]$_.job_id
                    Task = $_.task_id
                    Iterations = $_.training_iterations
                    Updated = [string]$stamp
                    Onnx = $onnx
                    Label = "$short $(([string]$_._id).Substring(([string]$_._id).Length - 8))$iters"
                }
            }
        } |
        Sort-Object Updated -Descending
}

function Get-DiskFallback {
    $f = Get-ChildItem (Join-Path $HostLogs 'rsl_rl\velocity') -Recurse -Filter 'policy_*.onnx' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return $null }
    [pscustomobject]@{ Id = ''; Updated = $f.LastWriteTime.ToString('o'); Onnx = $f.FullName
                       Label = "$($f.Directory.Name.Substring(0, [Math]::Min(19, $f.Directory.Name.Length))) $($f.BaseName)" }
}

function Select-DuckModel {
    try {
        $models = @(Get-DuckModels)
    } catch {
        if ($Model) { throw "Dashboard unreachable at $DashboardUrl, so model '$Model' can't be looked up: $_" }
        Write-Warning "Dashboard unreachable at $DashboardUrl; using the newest policy on disk."
        return Get-DiskFallback
    }
    if ($Model) {
        $hit = $models | Where-Object {
            $_.Id.EndsWith($Model) -or $_.Id.StartsWith($Model) -or $_.JobId.EndsWith($Model) } | Select-Object -First 1
        if (-not $hit) { throw "No compatible microduck model matches '$Model' (try -List)." }
        return $hit
    }
    return $models | Select-Object -First 1
}

function Start-Player($m) {
    $playerArgs = @('-policy', "`"$($m.Onnx)`"", '-label', "`"$($m.Label)`"",
                    '-screen-width', $Width, '-screen-height', $Height,
                    '-screen-fullscreen', $(if ($Fullscreen) { 1 } else { 0 }))
    Write-Host "MicroDuck: $($m.Label)  ($($m.Onnx))"
    return Start-Process $Exe -ArgumentList $playerArgs -PassThru
}

if ($List) {
    Get-DuckModels | Format-Table Id, Task, Iterations, Updated, Onnx -AutoSize
    return
}

$current = Select-DuckModel
if (-not $current) { throw "No compatible microduck policy found (dashboard models or $HostLogs)." }
$proc = Start-Player $current
if (-not $Follow) { return }

Write-Host "Following new policies every $PollSeconds s; Ctrl+C to stop (the player keeps running)."
while ($true) {
    Start-Sleep -Seconds $PollSeconds
    if ($proc.HasExited) { Write-Host 'Player closed; stopping.'; return }
    try { $newest = Get-DuckModels | Select-Object -First 1 } catch { continue }
    if ($newest -and ($newest.Onnx -ne $current.Onnx)) {
        Write-Host "$(Get-Date -Format T) newer policy: $($newest.Label)"
        Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
        $proc.WaitForExit(10000) | Out-Null
        $current = $newest
        $proc = Start-Player $current
    }
}
