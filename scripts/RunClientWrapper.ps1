<#
.SYNOPSIS
    Launch and supervise one Unity gym client. Restarts it if the process
    becomes unresponsive or exits.

.DESCRIPTION
    By default, looks for a single .exe at the top of unity/Builds/latest/
    (the destination of scripts/PromoteLatestBuild.ps1).

    Multi-actor support: pass -Index 0..N-1 to derive a unique ROS port
    (10000 + Index) and have RosBootstrap.cs route the client to the
    matching ros-server-N container brought up by compose/scale.yml.

.PARAMETER Index
    Actor index (0 = the base ros-server, 1.. = ros-server-1, etc.).
    Determines the default --ros-port and is also used as a label in
    log output. Default 0 (single-client behavior).

.PARAMETER RosIp
    --ros-ip value passed to the Unity exe. Default 127.0.0.1, which
    is how the Unity client (running on the Windows host) reaches the
    ros-server container's published port. Note: this is NOT
    host.docker.internal - that name only resolves from inside a
    container looking out at the host, which is the opposite direction.

.PARAMETER RosPort
    --ros-port value. Defaults to 10000 + Index.

.PARAMETER UnityPort
    --unity-port value. Defaults to 5005 + Index. ROS-TCP-Connector is
    bidirectional: Unity opens a local TcpListener on this port for the
    matching ros-server to push messages back to. Multiple Unity
    instances on the same host MUST use different unityPorts - if they
    collide, only the first instance binds successfully and the rest
    fail with "Address already in use" out of
    ROSConnection.StartMessageServer (Player.log shows a flood of
    SocketException stack traces) and silently never connect to ROS at
    all. The corresponding ros-server reads the chosen port from the
    Unity-side handshake and dials back host.docker.internal:<unityPort>.

.PARAMETER Path
    Optional explicit path to the Unity .exe. If omitted, the script globs
    unity/Builds/latest/*.exe, excludes Unity's bundled crash handler
    (UnityCrashHandler*.exe), and expects exactly one remaining match.

.PARAMETER PollSeconds
    How often (seconds) to check whether the process is still responding.
    Default 15. Raised from the original 5 because under multi-actor GPU
    contention a Unity main thread can legitimately stall its Windows
    message pump for >5s during scene-init, GfxDevice creation, or
    synchronous ROS-TCP waits, and we don't want the health probe to
    false-positive into a Kill().

.PARAMETER UnresponsiveStrikes
    Number of consecutive PollSeconds windows that the Unity process
    must fail .Responding before the supervisor kills it. Default 3 -
    so a Unity main thread has up to ~3*PollSeconds (~45s by default)
    of message-pump stall before being recycled. Single .Responding
    misses are extremely common under multi-actor load and rarely
    indicate an actual hang.

.PARAMETER WindowWidth
.PARAMETER WindowHeight
.PARAMETER WindowQuality
    Window flags forwarded to Unity. Defaults render to a 960x540 normal
    resizable window at "Fastest" quality. Pair with -Popup for the
    historical 320x240 borderless popup (cheap when several actors share
    one GPU); pass -WindowWidth 1280 etc. to override.

.PARAMETER Popup
    Use Unity's -popupwindow flag: borderless, fixed-size, no title bar.
    Useful when running -RunNClients for multi-actor training and you
    want N tiny clients tiling the screen rather than N normal windows.
    Implies a 320x240 default if -WindowWidth/-WindowHeight aren't set.

.PARAMETER LogFile
    Path forwarded to Unity's -logfile flag. By default each client writes
    to <exe-dir>/Player.log so multi-instance runs don't overwrite each
    other's logs (Unity normally points all instances of the same product
    at the same %LOCALAPPDATA%/<Co>/<Product>/Player.log, which makes
    diagnosing concurrent-client crashes impossible). Pass an empty
    string to fall back to Unity's default location.

.PARAMETER GridCols
.PARAMETER GridRows
    When both are > 0, the spawned Unity window is moved into the
    grid cell at column ($Index % GridCols), row ($Index / GridCols)
    of the primary monitor's working area. Tile size is
    (screen_width / GridCols) by (screen_height / GridRows) so the
    N tiles fill the screen exactly with no overlap. Default 0/0
    leaves Unity's spawn position untouched (it stacks new windows on
    top of each other - which is annoying when running 4 actors).

    Re-applies on supervisor respawn so a recycled Unity client
    doesn't drift back to the default spawn location.
#>
[CmdletBinding()]
param(
    [int]$Index = 0,
    [string]$RosIp = '127.0.0.1',
    [int]$RosPort = 0,
    [int]$UnityPort = 0,
    [string]$Path,
    [int]$PollSeconds = 15,
    [int]$UnresponsiveStrikes = 3,
    [int]$GridCols = 0,
    [int]$GridRows = 0,
    [int]$WindowWidth = 0,
    [int]$WindowHeight = 0,
    [string]$WindowQuality = 'Fastest',
    [switch]$Popup,
    [string]$LogFile
)

$ErrorActionPreference = 'Stop'

if ($RosPort   -le 0) { $RosPort   = 10000 + $Index }
if ($UnityPort -le 0) { $UnityPort = 5005  + $Index }

if ($WindowWidth  -le 0) { $WindowWidth  = if ($Popup) { 320 } else { 960 } }
if ($WindowHeight -le 0) { $WindowHeight = if ($Popup) { 240 } else { 540 } }

if (-not $Path) {
    $latestDir = Join-Path $PSScriptRoot '..\unity\Builds\latest'
    if (-not (Test-Path -LiteralPath $latestDir)) {
        throw "No build at $latestDir. Build from Unity into a subfolder of unity/Builds/, then run scripts/PromoteLatestBuild.ps1."
    }
    $allExes = @(Get-ChildItem -LiteralPath $latestDir -Filter '*.exe' -File)
    if ($allExes.Count -eq 0) {
        throw "No .exe found at top of $latestDir."
    }
    # Unity ships UnityCrashHandler64.exe alongside the player; it's not the game.
    $exes = @($allExes | Where-Object { $_.Name -notlike 'UnityCrashHandler*.exe' })
    if ($exes.Count -eq 0) {
        $names = ($allExes | ForEach-Object { $_.Name }) -join ', '
        throw "No game .exe found at top of $latestDir (only Unity helpers: $names). Pass -Path to disambiguate."
    }
    if ($exes.Count -gt 1) {
        $names = ($exes | ForEach-Object { $_.Name }) -join ', '
        throw "Expected exactly one game .exe in $latestDir, found $($exes.Count): $names. Pass -Path to disambiguate."
    }
    $Path = $exes[0].FullName
}

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Binary not found: $Path"
}

# Default each client to its own Player.log next to its .exe. PowerShell
# binds an unset [string] parameter as either $null or '' depending on
# version, so guard with a single -not check that handles both.
$exeDir = Split-Path -Parent $Path
if (-not $LogFile) {
    $LogFile = Join-Path $exeDir 'Player.log'
}

# Start-Process -ArgumentList joins elements with spaces and does NOT
# auto-quote, so a path containing spaces (e.g. a workspace dir whose
# name has a space in it) would be split. Sidestep the quoting question
# entirely: launch Unity
# with cwd = $exeDir and pass a relative -logfile filename. The log
# file ends up at $LogFile but no spaces traverse the command line.
$logFileFileName = [System.IO.Path]::GetFileName($LogFile)
$logFileDir      = Split-Path -Parent $LogFile
if (-not (Test-Path -LiteralPath $logFileDir)) {
    New-Item -ItemType Directory -Force -Path $logFileDir | Out-Null
}
# The -WorkingDirectory passed to Start-Process must equal $logFileDir
# for the relative -logfile arg to resolve to $LogFile.
$cwdForUnity = $logFileDir

$exeArgs = @(
    '--ros-ip', $RosIp,
    '--ros-port', $RosPort,
    '--unity-port', $UnityPort,
    '-screen-width', $WindowWidth,
    '-screen-height', $WindowHeight,
    '-screen-quality', $WindowQuality,
    '-logfile', $logFileFileName
)
if ($Popup) { $exeArgs += '-popupwindow' }

# Grid-positioning plumbing. Loaded only once, even if the wrapper
# script gets re-sourced. Pulls in System.Windows.Forms for screen
# dimensions and adds a tiny user32.dll shim for SetWindowPos.
if ($GridCols -gt 0 -and $GridRows -gt 0) {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
    if (-not ('Win32WindowPlacement' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class Win32WindowPlacement {
    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool SetWindowPos(
        IntPtr hWnd, IntPtr hWndInsertAfter,
        int X, int Y, int cx, int cy, uint uFlags);
    public const uint SWP_NOZORDER = 0x0004;
    public const uint SWP_NOACTIVATE = 0x0010;
}
'@
    }
}

function Move-ProcessWindowToGrid {
    param([int]$ProcessId, [int]$Index, [int]$Cols, [int]$Rows)

    if ($Cols -le 0 -or $Rows -le 0) { return }

    # Unity takes a few seconds to create its main window. Poll for
    # MainWindowHandle to become non-zero, with a deadline. If Unity
    # crashes during boot we exit early so the supervisor's normal
    # restart logic still kicks in.
    $deadline = (Get-Date).AddSeconds(45)
    $hwnd = [IntPtr]::Zero
    while ((Get-Date) -lt $deadline) {
        $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $p) {
            Write-Host "[$Index] grid: process exited before window appeared; skipping placement"
            return
        }
        if ($p.MainWindowHandle -ne [IntPtr]::Zero) {
            $hwnd = $p.MainWindowHandle
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if ($hwnd -eq [IntPtr]::Zero) {
        Write-Host "[$Index] grid: main window didn't appear in 45s; skipping placement"
        return
    }

    $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $tileW = [int]($screen.Width / $Cols)
    $tileH = [int]($screen.Height / $Rows)
    $col = $Index % $Cols
    $row = [int][Math]::Floor($Index / $Cols)
    $x = $screen.X + $col * $tileW
    $y = $screen.Y + $row * $tileH

    $flags = [Win32WindowPlacement]::SWP_NOZORDER -bor [Win32WindowPlacement]::SWP_NOACTIVATE
    [void][Win32WindowPlacement]::SetWindowPos(
        $hwnd, [IntPtr]::Zero,
        $x, $y, $tileW, $tileH, $flags)

    Write-Host ("[{0}] grid: cell=(c{1},r{2}) -> ({3},{4}) size {5}x{6}" -f `
        $Index, $col, $row, $x, $y, $tileW, $tileH)
}

function Start-Client {
    $p = Start-Process -FilePath $Path -ArgumentList $exeArgs -WorkingDirectory $cwdForUnity -PassThru
    Move-ProcessWindowToGrid -ProcessId $p.Id -Index $Index -Cols $GridCols -Rows $GridRows
    return $p
}

$proc = Start-Client
$mode = if ($Popup) { 'popup' } else { 'windowed' }
Write-Host ("[{0}] started PID={1}, ROS endpoint {2}:{3} <-> unityPort {4}, {5} {6}x{7} {8}" -f `
    $Index, $proc.Id, $RosIp, $RosPort, $UnityPort, $mode, $WindowWidth, $WindowHeight, $WindowQuality)
Write-Host "[$Index] log: $LogFile"

# Track consecutive .Responding misses so a transient main-thread stall
# under multi-actor GPU contention doesn't get punished with a Kill().
# Only a sustained N-poll-window unresponsive streak is treated as a
# real hang (e.g. Unity is genuinely deadlocked, not just slow).
$missStreak = 0

while ($true) {
    Start-Sleep -Seconds $PollSeconds

    # Re-fetch so .Responding reads live state.
    $current = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue

    if (-not $current) {
        Write-Host "[$Index] Status = Exited: Restart..."
        $proc = Start-Client
        Write-Host "[$Index] respawned PID=$($proc.Id)"
        $missStreak = 0
        continue
    }

    if (-not $current.Responding) {
        $missStreak++
        if ($missStreak -ge $UnresponsiveStrikes) {
            Write-Host "[$Index] Status = Not Responding x$missStreak (PID=$($current.Id)): Kill & Restart..."
            try { $current.Kill() } catch { Write-Host "[$Index] kill failed: $_" }
            $proc = Start-Client
            Write-Host "[$Index] respawned PID=$($proc.Id)"
            $missStreak = 0
        } else {
            Write-Host "[$Index] Status = Not Responding x$missStreak (PID=$($current.Id)): tolerating, scene-init / GPU / ROS sync stalls are common under load"
        }
    } else {
        if ($missStreak -gt 0) {
            Write-Host "[$Index] working fine again (PID=$($current.Id)) after $missStreak unresponsive poll(s)"
        } else {
            Write-Host "[$Index] working fine (PID=$($current.Id))"
        }
        $missStreak = 0
    }
}

# Source: https://community.idera.com/database-tools/powershell/ask_the_experts/f/powershell_for_windows-12/7002/how-to-detect-process-not-responding
