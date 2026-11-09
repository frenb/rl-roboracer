<#
.SYNOPSIS
    Open the live microduck 3D view in its own window, like the Unity
    clients do for the roboracer.

.DESCRIPTION
    Starts rl_agent/duck/live_view.py in the duck-trainer container under a
    supervisor (a minimized PowerShell window that restarts it if it exits)
    and opens http://localhost:8090 as a standalone app window (Edge or
    Chrome --app mode: no tabs or address bar), 960x540 like a Unity client.

    The view follows the running microduck TRAIN job, loading each new
    checkpoint as it is saved, and moves on to the next job by itself. With
    no duck job yet it shows a waiting message.

    The view shares the GPU with training; it simulates a handful of
    robots, so the slowdown is small. Start-Stack.ps1 -Duck runs this
    script; Stop-Stack.ps1 runs it with -Stop.

.PARAMETER JobId
    Pin the view to one microduck TRAIN job instead of following.

.PARAMETER NumEnvs
    Robots shown. Default 2. The view idles while its window is closed.

.PARAMETER WindowWidth
    App window width. Default 960, the Unity clients' default.

.PARAMETER WindowHeight
    App window height. Default 540.

.PARAMETER NoBrowser
    Start the view and its supervisor without opening a window.

.PARAMETER Stop
    Close the window, stop the supervisor and the view.

.PARAMETER Supervise
    Internal: run the supervisor loop in this process.

.EXAMPLE
    .\scripts\Watch-Duck.ps1
.EXAMPLE
    .\scripts\Watch-Duck.ps1 -JobId 6af2304da205b116dc617a35
.EXAMPLE
    .\scripts\Watch-Duck.ps1 -Stop
#>
param(
    [string]$JobId,
    [int]$NumEnvs = 2,
    [int]$WindowWidth = 960,
    [int]$WindowHeight = 540,
    [switch]$NoBrowser,
    [switch]$Stop,
    [switch]$Supervise
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
# Absolute paths so callers (Start-Stack, Stop-Stack) keep their location;
# compose resolves relative paths from the first file's directory.
$composeArgs = @('docker-compose.yml', 'compose/scale.yml', 'compose/duck.yml') |
    ForEach-Object { '-f'; Join-Path $RepoRoot $_ }
$url = 'http://localhost:8090'
# Windows PowerShell tries localhost over IPv6 first and times out on
# Docker Desktop's port forward; the IPv4 loopback answers at once.
$probeUrl = 'http://127.0.0.1:8090'
$viewerLog = '/microduck_rl/logs/jobs/live_view.log'
# A dedicated browser profile makes the app window its own process tree,
# so -Stop can close it without touching the user's other browser windows.
$profileDir = Join-Path $env:LOCALAPPDATA 'rl-roboracer\duck-viewer-browser'
# "[l]ive_view" keeps the pattern from matching the shell running pgrep/pkill.
$viewerPattern = "'[l]ive_view.py'"

function Invoke-Duck([string[]]$Command, [switch]$Detached) {
    $execArgs = @('compose') + $composeArgs + @('exec', '-T', '-w', '/microduck_rl')
    if ($Detached) { $execArgs += '-d' }
    & docker @($execArgs + 'duck-trainer' + $Command)
}

function Test-ViewerRunning {
    try {
        $out = Invoke-Duck @('sh', '-c', "pgrep -f $viewerPattern >/dev/null && echo yes || echo no") 2>$null
    } catch {
        return $false
    }
    return (($out | Out-String).Trim() -eq 'yes')
}

function Start-Viewer {
    $viewArgs = "--num-envs $NumEnvs" + $(if ($JobId) { " --job-id $JobId" } else { '' })
    Invoke-Duck -Detached @('sh', '-c',
        "python -u /python_ws/src/duck/live_view.py $viewArgs >> $viewerLog 2>&1")
}

function Get-SupervisorProcesses {
    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -like '*Watch-Duck.ps1*-Supervise*' -and $_.ProcessId -ne $PID }
}

function Stop-All {
    Get-SupervisorProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe' OR Name = 'chrome.exe'" |
        Where-Object { $_.CommandLine -like "*$profileDir*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # Fails harmlessly when the duck-trainer container isn't running.
    try { Invoke-Duck @('sh', '-c', "pkill -f $viewerPattern || true") 2>$null | Out-Null } catch { }
}

function Find-Browser {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

if ($Supervise) {
    $Host.UI.RawUI.WindowTitle = 'duck live view supervisor'
    while ($true) {
        if (-not (Test-ViewerRunning)) {
            Write-Host "$(Get-Date -Format T) starting live view"
            Start-Viewer
        }
        Start-Sleep -Seconds 15
    }
}

Stop-All
if ($Stop) {
    Write-Host 'Duck live view stopped.'
    return
}

$superviseArgs = @('-NoExit', '-File', $PSCommandPath, '-Supervise', '-NumEnvs', $NumEnvs)
if ($JobId) { $superviseArgs += @('-JobId', $JobId) }
Start-Process powershell -ArgumentList $superviseArgs -WindowStyle Minimized
Write-Host 'Starting the duck live view...'

$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    try {
        Invoke-WebRequest -Uri $probeUrl -UseBasicParsing -TimeoutSec 3 | Out-Null
        break
    } catch { }
}
if ((Get-Date) -ge $deadline) {
    throw "Live view did not answer on $url within 3 minutes; see $viewerLog in the duck-trainer container."
}

if ($NoBrowser) {
    Write-Host "Live view ready: $url"
    return
}
$browser = Find-Browser
if (-not $browser) {
    Write-Warning "No Edge or Chrome found; opening $url in the default browser instead."
    Start-Process $url
    return
}
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
Start-Process $browser -ArgumentList @(
    "--app=$url",
    "--window-size=$WindowWidth,$WindowHeight",
    "--user-data-dir=$profileDir",
    # Policy-installed extensions otherwise open onboarding tabs in the
    # fresh profile next to the viewer.
    '--disable-extensions',
    # WebGL on the integrated GPU: on the NVIDIA one, Windows time-slices it
    # against training's CUDA work and training steps get several times slower.
    '--force_low_power_gpu',
    '--no-first-run',
    '--no-default-browser-check'
)

# Edge opens a fresh app window maximized whatever --window-size says, so
# restore and size it once it exists. This (DPI-unaware) process works in
# scaled pixels, so the size matches the Unity windows' on-screen footprint
# at 100% scaling and grows with display scaling.
if (-not ('DuckView.Win32' -as [type])) {
    Add-Type -Namespace DuckView -Name Win32 -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
[DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
[DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr lParam);
[DllImport("user32.dll")] static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder s, int n);
[DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
[DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
public static IntPtr FindWindow(uint[] pids, string title) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h, _) => {
        uint pid;
        GetWindowThreadProcessId(h, out pid);
        if (!IsWindowVisible(h) || Array.IndexOf(pids, pid) < 0) return true;
        var s = new System.Text.StringBuilder(256);
        GetWindowText(h, s, 256);
        if (s.ToString() != title) return true;
        found = h;
        return false;
    }, IntPtr.Zero);
    return found;
}
'@
}
# viser titles its page "Viser"; only windows of this profile's browser are
# considered, so the user's own browser windows are never touched.
$deadline = (Get-Date).AddSeconds(20)
$hwnd = [IntPtr]::Zero
while ($hwnd -eq [IntPtr]::Zero -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    [uint32[]]$pids = Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe' OR Name = 'chrome.exe'" |
        Where-Object { $_.CommandLine -like "*$profileDir*" } | ForEach-Object { $_.ProcessId }
    if ($pids) { $hwnd = [DuckView.Win32]::FindWindow($pids, 'Viser') }
}
if ($hwnd -ne [IntPtr]::Zero) {
    $SW_RESTORE = 9
    $SWP_NOZORDER = 0x0004
    [DuckView.Win32]::ShowWindow($hwnd, $SW_RESTORE) | Out-Null
    [DuckView.Win32]::SetWindowPos($hwnd, [IntPtr]::Zero, 40, 40,
        $WindowWidth, $WindowHeight, $SWP_NOZORDER) | Out-Null
} else {
    Write-Warning 'Could not find the viewer window to size it; it may have opened maximized.'
}
Write-Host "Duck live view window opened ($url). Close it with .\scripts\Watch-Duck.ps1 -Stop"
