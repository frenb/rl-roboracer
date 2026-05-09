<#
.SYNOPSIS
    Kill the Unity clients, their PowerShell supervisor tabs/windows,
    and the Windows Terminal window that hosted them.
    Leaves the Docker stack alone.

.DESCRIPTION
    Equivalent to ``Stop-Stack.ps1 -KeepDocker`` - this script is just
    a more discoverable name for that operation. The Stop-Stack script
    handles all four phases (capture wt host PIDs, kill supervisors,
    kill Unity, optionally close wt) and skips the docker-down phase
    when ``-KeepDocker`` is passed.

.PARAMETER CloseWtWindow
    Forwarded to Stop-Stack.ps1. Default $true. Pass
    -CloseWtWindow:$false to leave the Windows Terminal window open
    after killing its supervisor tabs (e.g. for inspecting last log
    lines before closing manually).

.EXAMPLE
    .\scripts\Stop-Clients.ps1

.EXAMPLE
    .\scripts\Stop-Clients.ps1 -CloseWtWindow:$false
#>
[CmdletBinding()]
param(
    [bool]$CloseWtWindow = $true
)

$stopStack = Join-Path $PSScriptRoot 'Stop-Stack.ps1'
if (-not (Test-Path -LiteralPath $stopStack)) {
    throw "Stop-Stack.ps1 not found at $stopStack"
}

& $stopStack -KeepDocker -CloseWtWindow:$CloseWtWindow
