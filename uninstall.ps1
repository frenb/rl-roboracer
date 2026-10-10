# InstallZero one-line uninstaller for rl-roboracer on Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/uninstall.ps1 | iex
#
# With options (-InstallDir, -RemoveData, -KeepData, -PruneBuildCache, -Yes, -WhatIf):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/frenb/rl-roboracer/main/uninstall.ps1))) -RemoveData
#
# Downloads scripts/install/Uninstall.ps1 and runs it in Windows PowerShell,
# passing the options through. RL_INSTALL_BRANCH picks another branch.

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $branch = if ($env:RL_INSTALL_BRANCH) { $env:RL_INSTALL_BRANCH } else { 'main' }
    $script = Join-Path $env:TEMP 'rl-roboracer-Uninstall.ps1'
    Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/frenb/rl-roboracer/$branch/scripts/install/Uninstall.ps1" -OutFile $script
    $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $ps -NoProfile -ExecutionPolicy Bypass -File $script @args
    Remove-Item $script -ErrorAction SilentlyContinue
} @args
