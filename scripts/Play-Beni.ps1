<#
.SYNOPSIS
    Drive a trained OpenBeni policy in the Unity OpenBeni player.

.DESCRIPTION
    Resolves a policy ONNX (explicit -Policy, dashboard model id, or the
    pretrained asset shipped with open-beni) and starts OpenBeni.exe with
    -policy, matching the MicroDuck Play-Duck.ps1 UX.

.PARAMETER Policy
    Path to a policy.onnx (recurrent GRU-512, 40-D obs / 7-D action).

.PARAMETER Model
    Dashboard model id (or last-8 suffix) whose onnx_location to play.

.PARAMETER Exe
    Path to OpenBeni.exe. Default: ..\spikes\unity-beni\build\OpenBeni\OpenBeni.exe

.PARAMETER List
    List recent openbeni models from the dashboard and exit.

.EXAMPLE
    .\scripts\Play-Beni.ps1

.EXAMPLE
    .\scripts\Play-Beni.ps1 -Policy ..\rl-roboracer\third_party\open-beni\assets\models\policy.onnx
#>
[CmdletBinding()]
param(
    [string]$Policy,
    [string]$Model,
    [string]$Exe,
    [string]$Label,
    [switch]$List
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Workspace = Split-Path $RepoRoot -Parent

if (-not $Exe) {
    $Exe = Join-Path $Workspace 'spikes\unity-beni\build\OpenBeni\OpenBeni.exe'
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "OpenBeni player not found at $Exe. Build it (spikes\unity-beni, BeniBuild.Build) or pass -Exe."
}

$HostLogs = Join-Path $Workspace 'saved_models\openbeni'
$Pretrained = Join-Path $RepoRoot 'third_party\open-beni\assets\models\policy.onnx'

function Get-BeniModels {
    $rows = Invoke-RestMethod -Uri 'http://localhost/get_models?force=true' -TimeoutSec 10
    if ($rows -is [string]) { return @() }
    @($rows) |
        Where-Object { $_.robot -eq 'openbeni' -and $_.onnx_location } |
        Sort-Object { $_.updated_at } -Descending |
        ForEach-Object {
            [pscustomobject]@{
                Id = [string]$_._id
                Task = $_.task_id
                Iterations = $_.training_steps
                Updated = $_.updated_at
                Onnx = $_.onnx_location
                Label = if ($_.name) { $_.name } else { $_.task_id }
            }
        }
}

function Resolve-HostOnnx([string]$ContainerPath) {
    if (-not $ContainerPath) { return $null }
    if (Test-Path -LiteralPath $ContainerPath) { return (Resolve-Path $ContainerPath).Path }
    $prefix = '/openbeni_rl/logs/'
    if ($ContainerPath.StartsWith($prefix)) {
        $rel = $ContainerPath.Substring($prefix.Length) -replace '/', '\'
        $hostPath = Join-Path $HostLogs $rel
        if (Test-Path -LiteralPath $hostPath) { return (Resolve-Path $hostPath).Path }
    }
    return $null
}

if ($List) {
    Get-BeniModels | Format-Table Id, Task, Iterations, Updated, Onnx -AutoSize
    return
}

$onnx = $null
$policyLabel = $Label
if ($Policy) {
    $onnx = (Resolve-Path -LiteralPath $Policy).Path
    if (-not $policyLabel) { $policyLabel = Split-Path $onnx -Leaf }
} elseif ($Model) {
    $hit = Get-BeniModels | Where-Object { $_.Id -eq $Model -or $_.Id.EndsWith($Model) } | Select-Object -First 1
    if (-not $hit) { throw "No openbeni model matches '$Model'." }
    $onnx = Resolve-HostOnnx $hit.Onnx
    if (-not $onnx) { throw "Model $($hit.Id) onnx not on disk: $($hit.Onnx)" }
    if (-not $policyLabel) { $policyLabel = $hit.Label }
} else {
    try {
        $newest = Get-BeniModels | Select-Object -First 1
        if ($newest) {
            $onnx = Resolve-HostOnnx $newest.Onnx
            if ($onnx -and -not $policyLabel) { $policyLabel = $newest.Label }
        }
    } catch { }
    if (-not $onnx) {
        if (-not (Test-Path -LiteralPath $Pretrained)) {
            throw "No openbeni ONNX found (dashboard models, $HostLogs, or $Pretrained)."
        }
        $onnx = (Resolve-Path $Pretrained).Path
        if (-not $policyLabel) { $policyLabel = 'pretrained policy.onnx' }
    }
}

# Unity Inference Engine has no GRU operator; unroll it into Gemm/Sigmoid/Tanh first.
$unroll = Join-Path $Workspace 'spikes\unity-beni\tools\unroll_gru.py'
$unityOnnx = Join-Path $env:TEMP ("openbeni_" + [IO.Path]::GetFileNameWithoutExtension($onnx) + "_" +
    (Get-FileHash -LiteralPath $onnx -Algorithm SHA256).Hash.Substring(0, 12) + "_unity.onnx")
if (-not (Test-Path -LiteralPath $unityOnnx)) {
    & py -3.13 $unroll $onnx $unityOnnx --check
    if ($LASTEXITCODE -ne 0) { throw "unroll_gru.py failed for $onnx (needs: py -3.13 -m pip install onnx onnxruntime)" }
}
if (Test-Path -LiteralPath $unityOnnx) { $onnx = $unityOnnx }

Write-Host "OpenBeni: $policyLabel"
Write-Host "  policy $onnx"
Write-Host "  exe    $Exe"
Start-Process -FilePath $Exe -ArgumentList @('-policy', $onnx, '-label', $policyLabel) -WorkingDirectory (Split-Path $Exe)
