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

.PARAMETER GymPollSeconds
    How often (seconds) to check whether the dashboard has requested
    a different Unity binary (gym hot-swap). Default 10. Set to 0
    to disable gym polling entirely (useful when the gym is fixed for
    the whole session).

.PARAMETER DashboardUrl
    Base URL of the dashboard HTTP server that exposes
    GET /get_desired_gym?index=<N> and POST /set_desired_gym.
    Default http://localhost (the dashboard container's published
    port 80). Change if the dashboard runs on a different host/port.

.PARAMETER GymSource
    The SOURCE .exe path the initial -Path instance copy was mirrored
    from (RunNClients.ps1 passes the unity/Builds/latest/<game>.exe it
    copied into unity/Builds/instances/<Index>/). Used as the baseline
    for gym-switch comparison so a job whose gym points at the same
    source doesn't trigger a needless restart on the first poll. When a
    gym switch IS needed, the new source build is mirrored into this
    actor's per-index instance dir (unity/Builds/instances/<Index>/) so
    multiple actors never collide on Unity's "Force Single Instance"
    mutex (which locks on the .exe path). Empty means "no known source"
    - the first non-empty desired gym then triggers one switch.
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
    [string]$LogFile,
    [int]$GymPollSeconds = 10,
    # Use the IPv4 loopback explicitly. 'localhost' resolves to the IPv6
    # ::1 first in .NET, but Docker Desktop publishes the dashboard port
    # only on IPv4 127.0.0.1, so an http://localhost request hangs until
    # timeout. 127.0.0.1 connects immediately.
    [string]$DashboardUrl = 'http://127.0.0.1',
    [string]$GymSource = ''
)

$ErrorActionPreference = 'Stop'

# Self-register with the shared stack-state directory so Stop-Stack
# can find this supervisor by PID without needing to ask WMI. The
# matching Unregister-Supervisor call lives in the try/finally
# wrapping the supervise loop at the bottom of the file. See
# scripts/_StackState.ps1 for the rationale (WMI's Win32_Process
# queries hang indefinitely when the Windows WMI service is wedged,
# and PowerShell's -OperationTimeoutSec doesn't reliably cancel a
# stuck call).
. (Join-Path $PSScriptRoot '_StackState.ps1')

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

# ---- Mutable launch state -------------------------------------------
# These script-scoped variables are updated by Set-ActiveExePath
# whenever a gym hot-swap changes the binary path. Start-Client and
# the supervise loop always read the current values, not the values
# that were set at parameter-parse time.
# ---------------------------------------------------------------------
$script:ActivePath    = $Path
$script:ActiveExeArgs = $null   # set by Set-ActiveExePath below
$script:ActiveCwd     = $null
# The SOURCE build exe the currently-running binary was mirrored from.
# Gym switching compares the dashboard's desired source against this so
# we only restart when the requested build genuinely differs.
$script:ActiveGymSource = $GymSource

function Set-ActiveExePath {
    param([string]$NewPath)
    if (-not (Test-Path -LiteralPath $NewPath)) {
        throw "Binary not found: $NewPath"
    }

    $script:ActivePath = $NewPath
    $exeDir = Split-Path -Parent $NewPath

    # Default the log next to the .exe (avoids spaces on the command line).
    $logPath = if ($LogFile) { $LogFile } else { Join-Path $exeDir 'Player.log' }
    $logFileName = [System.IO.Path]::GetFileName($logPath)
    $logDir      = Split-Path -Parent $logPath
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    $script:ActiveCwd = $logDir

    $script:ActiveExeArgs = @(
        '--ros-ip',     $RosIp,
        '--ros-port',   $RosPort,
        '--unity-port', $UnityPort,
        '-screen-width',    $WindowWidth,
        '-screen-height',   $WindowHeight,
        '-screen-quality',  $WindowQuality,
        '-logfile',     $logFileName
    )
    if ($Popup) { $script:ActiveExeArgs += '-popupwindow' }
}

# Initialise with the resolved path from the parameter block above.
Set-ActiveExePath -NewPath $Path

# ---- Gym hot-swap helpers -------------------------------------------
function Get-DesiredGymPath {
    # Ask the dashboard for the file_path the current job wants.
    # Falls back to '' on any error so the caller can treat a missing
    # or unreachable dashboard as "no switch needed".
    #
    # We use a raw HttpWebRequest with Proxy=$null rather than
    # Invoke-RestMethod: on Windows PowerShell 5.1 Invoke-RestMethod can
    # hang on proxy auto-detection, and resolving 'localhost' to IPv6 ::1
    # (which Docker Desktop doesn't bind) makes it time out. The explicit
    # request below, paired with the IPv4 DashboardUrl default, connects
    # immediately.
    if ($GymPollSeconds -le 0) { return '' }
    try {
        $uri = "${DashboardUrl}/get_desired_gym?index=${Index}"
        $req = [System.Net.HttpWebRequest]::Create($uri)
        $req.Method  = 'GET'
        $req.Timeout = 4000
        $req.Proxy   = $null
        $resp   = $req.GetResponse()
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
        $json   = $reader.ReadToEnd()
        $reader.Close()
        $resp.Close()

        $obj = $json | ConvertFrom-Json
        $p = [string]($obj.file_path)

        # Strip a single pair of surrounding quotes a pasted path may
        # carry (Explorer's "Copy as path" wraps in double quotes), so
        # Test-Path / robocopy below get a clean filesystem path even if
        # an older gym record was stored with the quotes intact.
        $p = $p.Trim()
        if ($p.Length -ge 2 -and $p[0] -eq $p[-1] -and ($p[0] -eq '"' -or $p[0] -eq "'")) {
            $p = $p.Substring(1, $p.Length - 2).Trim()
        }
        return $p
    } catch {
        # Dashboard not reachable / no gym set yet — not an error.
        return ''
    }
}

function Switch-ToGym {
    <#
      Mirror the gym's SOURCE build into this actor's per-index instance
      dir (unity/Builds/instances/<Index>/) and point the launch state at
      the per-index copy. Per-index copies are mandatory: Unity's "Force
      Single Instance" Player Setting locks on the .exe path, so N actors
      sharing one .exe would let only the first launch and silently block
      the rest. Mirroring the same way RunNClients.ps1 does keeps multi-
      actor gym switches collision-free.

      $SourceExePath : absolute path to the gym's .exe (the build the user
                       registered on the Gyms tab). Its parent dir is the
                       full Unity build folder that gets mirrored.
    #>
    param([string]$SourceExePath)

    $sourceDir = Split-Path -Parent $SourceExePath
    $exeName   = Split-Path -Leaf   $SourceExePath

    $repoRoot    = Split-Path $PSScriptRoot -Parent
    $instanceDir = Join-Path $repoRoot ("unity\Builds\instances\{0}" -f $Index)
    if (-not (Test-Path -LiteralPath $instanceDir)) {
        New-Item -ItemType Directory -Force -Path $instanceDir | Out-Null
    }

    Write-Host "[$Index] gym switch: mirroring '$sourceDir' -> '$instanceDir' ..."
    & robocopy $sourceDir $instanceDir /MIR /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    # robocopy exit codes 0..7 are success-with-info; >=8 is a real error.
    if ($LASTEXITCODE -ge 8) {
        throw "[$Index] robocopy failed mirroring gym build (exit $LASTEXITCODE): $sourceDir -> $instanceDir"
    }
    $global:LASTEXITCODE = 0

    $instanceExe = Join-Path $instanceDir $exeName
    if (-not (Test-Path -LiteralPath $instanceExe)) {
        throw "[$Index] gym build mirrored but exe missing: $instanceExe"
    }

    Set-ActiveExePath -NewPath $instanceExe
    $script:ActiveGymSource = $SourceExePath
}

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

# Seconds to hold the cross-supervisor launch lock waiting for the new
# Unity client to create its graphics device (signalled by its main
# window appearing). Bounds how long one actor blocks the others; a
# wedged client that never makes a window releases the slot after this.
$LaunchSettleSec = 30

function Wait-ForClientWindow {
    param([int]$ProcessId, [int]$TimeoutSec = 30)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $p) { return $false }                       # exited
        if ($p.MainWindowHandle -ne [IntPtr]::Zero) { return $true }  # device + window up
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Start-Client {
    # Serialize device creation across ALL actors: only one Unity client
    # creates its D3D device at a time, so a respawn can't collide with a
    # running sibling and deadlock at "GfxDevice: creating device client".
    $gotSlot = Enter-LaunchSlot -TimeoutSec 90 -StaleSec 60
    if (-not $gotSlot) {
        Write-Host "[$Index] launch slot wait timed out; launching anyway."
    }
    try {
        $p = Start-Process `
                -FilePath        $script:ActivePath `
                -ArgumentList    $script:ActiveExeArgs `
                -WorkingDirectory $script:ActiveCwd `
                -PassThru
        # Hold the slot until this client's window (= graphics device
        # created) appears, so the next actor's launch starts only after
        # this one is past the contended GfxDevice phase. A wedged client
        # that never makes a window releases the slot after the timeout.
        $up = Wait-ForClientWindow -ProcessId $p.Id -TimeoutSec $LaunchSettleSec
        if (-not $up) {
            Write-Host "[$Index] window not up within ${LaunchSettleSec}s of launch (may still be initializing or wedged)."
        }
    }
    finally {
        Exit-LaunchSlot
    }
    Move-ProcessWindowToGrid -ProcessId $p.Id -Index $Index -Cols $GridCols -Rows $GridRows
    return $p
}

# Self-register before doing anything else that could fail, so an
# early crash during Unity launch still leaves a record Stop-Stack
# can use to clean us up. The matching Unregister-Supervisor lives in
# the finally below; together they cover normal exit, Stop-Stack
# kills (file gets removed by the consistency check on stale entries),
# and Ctrl-C.
Register-Supervisor -ProcessId $PID -Index $Index -Exe $script:ActivePath

try {
    $proc = Start-Client
    $mode = if ($Popup) { 'popup' } else { 'windowed' }
    Write-Host ("[{0}] started PID={1}, ROS endpoint {2}:{3} <-> unityPort {4}, {5} {6}x{7} {8}" -f `
        $Index, $proc.Id, $RosIp, $RosPort, $UnityPort, $mode, $WindowWidth, $WindowHeight, $WindowQuality)
    Write-Host "[$Index] log: $LogFile"
    Write-Host "[$Index] gym polling: every ${GymPollSeconds}s via ${DashboardUrl}/get_desired_gym?index=${Index}"

    $missStreak     = 0
    $gymLastChecked = [DateTime]::UtcNow.AddSeconds(-$GymPollSeconds) # check on first poll

    while ($true) {
        Start-Sleep -Seconds $PollSeconds

        # ---- Gym hot-swap check ----------------------------------------
        # If the dashboard signals a different gym SOURCE build, mirror it
        # into this actor's per-index instance dir and restart Unity from
        # the copy. Compares against $script:ActiveGymSource (the source
        # the running binary was mirrored from) so an identical request
        # doesn't trigger a needless restart. Only runs when
        # GymPollSeconds > 0 and enough time has elapsed.
        if ($GymPollSeconds -gt 0) {
            $now = [DateTime]::UtcNow
            if (($now - $gymLastChecked).TotalSeconds -ge $GymPollSeconds) {
                $gymLastChecked = $now
                $desiredSource = Get-DesiredGymPath
                if ($desiredSource -and $desiredSource -ne $script:ActiveGymSource) {
                    if (Test-Path -LiteralPath $desiredSource) {
                        Write-Host ("[$Index] GYM SWITCH: {0} -> {1}" -f $script:ActiveGymSource, $desiredSource)
                        # Kill the running Unity process cleanly before switching.
                        $current = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
                        if ($current) {
                            try {
                                $current.Kill()
                                $current.WaitForExit(10000)
                            } catch {
                                Write-Host "[$Index] gym switch: kill failed (process may have already exited): $_"
                            }
                        }
                        # Mirror the new build into instances/<Index>/ and
                        # restart from the per-index copy.
                        try {
                            Switch-ToGym -SourceExePath $desiredSource
                            $proc = Start-Client
                            Write-Host "[$Index] gym switch complete. New PID=$($proc.Id) -> $($script:ActivePath)"
                        } catch {
                            Write-Host "[$Index] gym switch FAILED: $_"
                            # Fall back to relaunching whatever we had so the
                            # actor isn't left dead. ActivePath is unchanged
                            # if Switch-ToGym threw before Set-ActiveExePath.
                            $proc = Start-Client
                        }
                        $missStreak = 0
                        continue
                    } else {
                        Write-Host "[$Index] Desired gym binary not found at '$desiredSource'; keeping current."
                    }
                }
            }
        }
        # ----------------------------------------------------------------

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
} finally {
    # Drop the registration on normal exit / Ctrl-C. Hard kills
    # (Stop-Process -Force) bypass finally; the stale file is
    # harmless and gets reaped on the next Get-RegisteredSupervisors
    # call (it checks the PID is still alive AND still a powershell).
    Unregister-Supervisor -ProcessId $PID
}

# Source: https://community.idera.com/database-tools/powershell/ask_the_experts/f/powershell_for_windows-12/7002/how-to-detect-process-not-responding
