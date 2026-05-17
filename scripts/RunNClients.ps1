<#
.SYNOPSIS
    Spawn N supervised Unity clients (RunClientWrapper.ps1 instances), each
    in its own PowerShell window with a unique -Index.

.DESCRIPTION
    Pairs with compose/scale.yml: each client connects to the matching
    ros-server-N container by passing --ros-port (10000 + i).

    Each child wrapper opens its own console so its log output stays
    legible. Close any console window to stop supervising that client.

    Per-instance build copies: Unity's "Force Single Instance" Player
    Setting is baked into the build and locks on the .exe path. Without
    a workaround, only one instance of unity/Builds/latest/<game>.exe can
    run at a time. This script mirrors unity/Builds/latest/ into
    unity/Builds/instances/0..N-1/ (incremental robocopy, only changed
    files are copied) and points each supervisor at its own .exe path.
    Disk cost is one full build per index; rebuild + re-run picks up
    changes automatically.

    Typical workflow:
      1. docker compose -f docker-compose.yml -f compose/scale.yml up -d
      2. .\scripts\RunNClients.ps1 -N 4

    Make sure no other Unity client is running first (it will hold the
    instances/0 mutex, blocking actor 0 from launching).

.PARAMETER N
    Number of clients to spawn. Default 2. Each gets Index 0..N-1.

.PARAMETER RosIp
    Forwarded to each RunClientWrapper.ps1. Default 127.0.0.1, since
    Unity runs on the Windows host and reaches the ros-server containers
    via their published ports.

.PARAMETER WindowWidth
.PARAMETER WindowHeight
.PARAMETER WindowQuality
.PARAMETER Popup
    Forwarded to each RunClientWrapper.ps1. Defaults match the wrapper:
    960x540 windowed, "Fastest" quality. Pass -Popup for the historical
    320x240 borderless tile (good for stacking many actors on one
    screen during long training runs).

.PARAMETER StaggerSeconds
    Seconds to wait between supervisor spawns. Without a stagger, all N
    Unity instances race their D3D / Mono / Subsystems initialization
    simultaneously and lose under contention - some processes silently
    exit before any logger initializes, others wedge mid-CreateDevice
    with no main window and no further log output (you'll see Player.log
    stop at "GfxDevice: creating device client; threaded=1"). 15s is
    the conservative-but-reliable default and is plenty for actor i to
    finish CreateDevice + scene init before actor i+1 starts touching
    the GPU on typical hardware. Set to 0 to disable.

    The previous default of 3s was tuned on a faster GPU and bit users
    on more contended systems where CreateDevice alone takes 5-10s; the
    failure mode there is a permanent two-process deadlock that can
    only be cleared with Stop-Stack. Most of the launch time you "save"
    by lowering this you'll pay back in retries when the first attempt
    wedges. Lower it only if you've confirmed your specific GPU + driver
    + N is happy with the shorter window.

.PARAMETER GridCols
.PARAMETER GridRows
    Lays out the N Unity windows in a GridCols x GridRows grid on the
    primary monitor's working area instead of letting Unity stack them
    all at the default spawn position. Defaults: GridCols =
    ceil(sqrt(N)) and GridRows = ceil(N / GridCols), so N=4 is 2x2,
    N=2 is 2x1, N=9 is 3x3. Pass GridCols=0 (or GridRows=0) to opt out
    and keep the old stack-on-top behavior.

.PARAMETER Minimized
    Start the supervisor consoles minimized to the taskbar so they
    don't cover the Unity windows on launch. Default $true. Pass
    -Minimized:$false to keep them in the foreground (useful when
    actively debugging a supervisor).

.PARAMETER UseWindowsTerminal
    Group the N supervisor consoles into one Windows Terminal window
    with N tabs (named ``actor-0`` ... ``actor-N-1``) instead of
    spawning N separate powershell.exe console windows. Default
    $true; auto-detected (falls back to separate consoles if
    ``wt.exe`` isn't on PATH). Each tab still runs RunClientWrapper.ps1
    so Stop-Stack.ps1's CommandLine-based filter cleans them up the
    same way.
#>
[CmdletBinding()]
param(
    [int]$N = 2,
    [string]$RosIp = '127.0.0.1',
    [int]$WindowWidth = 0,
    [int]$WindowHeight = 0,
    [string]$WindowQuality = 'Fastest',
    [switch]$Popup,
    [double]$StaggerSeconds = 15.0,
    [int]$GridCols = -1,
    [int]$GridRows = -1,
    [bool]$Minimized = $true,
    [bool]$UseWindowsTerminal = $true
)

# Auto-compute a square-ish default grid from $N if the caller didn't
# specify cols/rows. ceil(sqrt(N)) cols, then enough rows to cover N.
# Caller can pass GridCols=0 (or GridRows=0) to disable grid placement
# entirely; both >0 to use a custom layout.
if ($GridCols -lt 0) {
    $GridCols = [int][Math]::Ceiling([Math]::Sqrt($N))
}
if ($GridRows -lt 0) {
    $GridRows = [int][Math]::Ceiling($N / [double]$GridCols)
}

$ErrorActionPreference = 'Stop'

# Shared stack-state helpers (see scripts/_StackState.ps1 for the
# rationale - we record supervisor / wt-host PIDs on disk so
# Stop-Stack doesn't need to query WMI's Win32_Process and freeze
# when WMI is wedged).
. (Join-Path $PSScriptRoot '_StackState.ps1')

# Start from a clean slate. Any stale registrations from a previous
# Start-Clients run that wasn't properly torn down would otherwise
# confuse the next Stop-Stack into trying to kill PIDs that have
# been recycled. Get-RegisteredSupervisors already filters those
# out, but resetting upfront keeps the on-disk state small and the
# code paths predictable.
Clear-StackState

$wrapperPath = Join-Path $PSScriptRoot 'RunClientWrapper.ps1'
if (-not (Test-Path -LiteralPath $wrapperPath)) {
    throw "Wrapper not found: $wrapperPath"
}

if ($N -lt 1) { throw "N must be >= 1 (got $N)." }

# Resolve build paths.
$repoRoot      = Split-Path $PSScriptRoot -Parent
$latestDir     = Join-Path $repoRoot 'unity\Builds\latest'
$instancesRoot = Join-Path $repoRoot 'unity\Builds\instances'

if (-not (Test-Path -LiteralPath $latestDir)) {
    throw "No build at $latestDir. Build from Unity into unity/Builds/, then run scripts/PromoteLatestBuild.ps1."
}

# Find the game .exe at the top of latest/, ignoring Unity's crash handler.
$gameExes = @(
    Get-ChildItem -LiteralPath $latestDir -Filter '*.exe' -File |
        Where-Object { $_.Name -notlike 'UnityCrashHandler*.exe' }
)
if ($gameExes.Count -eq 0) {
    throw "No game .exe found at top of $latestDir."
}
if ($gameExes.Count -gt 1) {
    $names = ($gameExes | ForEach-Object { $_.Name }) -join ', '
    throw "Expected exactly one game .exe in $latestDir, found $($gameExes.Count): $names."
}
$exeName = $gameExes[0].Name

# Sync per-index copies via robocopy /MIR. /MIR mirrors changes only,
# so a steady-state re-run is fast. Quiet flags suppress robocopy's
# normal stat dump.
Write-Host "Syncing $N per-index build copies under unity\Builds\instances\..."
$indexExePaths = @()
for ($i = 0; $i -lt $N; $i++) {
    $instanceDir = Join-Path $instancesRoot $i
    if (-not (Test-Path -LiteralPath $instanceDir)) {
        New-Item -ItemType Directory -Force -Path $instanceDir | Out-Null
    }

    & robocopy $latestDir $instanceDir /MIR /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    # robocopy exit codes: 0..7 are success-with-info, >=8 is error.
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for index $i (exit code $LASTEXITCODE). Check $instanceDir."
    }
    # robocopy sets $LASTEXITCODE; reset so a benign code doesn't trip a later check.
    $global:LASTEXITCODE = 0

    $exePath = Join-Path $instanceDir $exeName
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Synced instance dir is missing the game exe: $exePath"
    }
    $indexExePaths += $exePath
}

# Hosting strategy: prefer Windows Terminal (wt.exe) if available so
# the N supervisors share one window with N tabs instead of cluttering
# the taskbar with N separate powershell consoles. Auto-falls back to
# plain powershell consoles if wt isn't on PATH or the user opted out.
# Search by name 'wt' (no extension) because Windows registers wt as
# an App Execution Alias - Get-Command 'wt.exe' won't see it, but
# Get-Command 'wt' resolves to <LocalAppData>\Microsoft\WindowsApps\wt.exe.
$wtCommand = $null
if ($UseWindowsTerminal) {
    $wtCommand = Get-Command 'wt' -ErrorAction SilentlyContinue
}
$useWt = ($null -ne $wtCommand)
$wtWindowName = 'robotaxi-stack'

$gridLayoutMsg = if ($GridCols -gt 0 -and $GridRows -gt 0) {
    "${GridCols}x${GridRows} grid on primary monitor"
} else {
    "no grid (default Unity spawn position)"
}
$hostMsg = if ($useWt) { "wt tabs in window '$wtWindowName'" } else { 'separate powershell consoles' }
$visibilityMsg = if ($Minimized) { 'minimized' } else { 'normal' }
Write-Host "Launching $N supervised Unity clients (stagger ${StaggerSeconds}s, ${gridLayoutMsg}, ${hostMsg}, ${visibilityMsg})..."

# We build the spawned-powershell argument list as a single explicitly-
# quoted string. PowerShell 5.1's Start-Process -ArgumentList @(...) joins
# array elements with spaces but does NOT auto-quote, so any path arg
# containing a space (e.g. a workspace dir whose name has a space in it)
# would be split mid-value when the child shell parses it - the
# supervisor would crash in param binding before ever launching Unity
# (and you'd see no Player.log).
for ($i = 0; $i -lt $N; $i++) {
    if ($i -gt 0 -and $StaggerSeconds -gt 0) {
        Start-Sleep -Seconds $StaggerSeconds
    }

    $exePath = $indexExePaths[$i]
    $psCmdLine = @(
        '-NoExit'
        '-File "{0}"'           -f $wrapperPath
        '-Index {0}'            -f $i
        '-RosIp "{0}"'          -f $RosIp
        '-Path "{0}"'           -f $exePath
        '-WindowWidth {0}'      -f $WindowWidth
        '-WindowHeight {0}'     -f $WindowHeight
        '-WindowQuality "{0}"'  -f $WindowQuality
        '-GridCols {0}'         -f $GridCols
        '-GridRows {0}'         -f $GridRows
    ) -join ' '
    if ($Popup) { $psCmdLine += ' -Popup' }

    $startArgs = @{ PassThru = $true }
    if ($Minimized) { $startArgs.WindowStyle = 'Minimized' }

    if ($useWt) {
        # `wt -w <name>` finds-or-creates a wt window with that name.
        # First iteration creates it; subsequent iterations attach
        # new tabs. Each tab runs powershell with the same wrapper
        # arguments as the fallback path, so RunClientWrapper.ps1 sees
        # an identical command line in either mode.
        $wtCmdLine = @(
            '-w', $wtWindowName
            'new-tab'
            '--title', "actor-$i"
            'powershell'
            $psCmdLine
        ) -join ' '
        $startArgs.FilePath     = 'wt.exe'
        $startArgs.ArgumentList = $wtCmdLine
        $proc = Start-Process @startArgs
        Write-Host "  [$i] tab created in wt window '$wtWindowName' -> $exePath"
    } else {
        $startArgs.FilePath     = 'powershell'
        $startArgs.ArgumentList = $psCmdLine
        $proc = Start-Process @startArgs
        Write-Host "  [$i] supervisor PID=$($proc.Id) -> $exePath"
    }
}

# Find the wt host PID without WMI. We rely on the supervisors having
# self-registered into the stack-state dir by the time we get here
# (RunClientWrapper.ps1 calls Register-Supervisor as its first
# action), then walk up from a registered supervisor to the wt host
# via the .NET Process class. This intentionally avoids any
# Get-CimInstance / Win32_Process call so a wedged WMI service
# doesn't freeze the script.
#
# Why we need the wt host PID:
#   * To re-minimize the wt window (the per-tab Start-Process
#     -WindowStyle Minimized only takes effect for the FIRST tab;
#     each subsequent `wt new-tab` un-minimizes the window).
#   * Stop-Stack uses it to close the wt window when tearing down.
function Get-WtHostPidByWalk {
    # Wait briefly for at least one supervisor to register itself.
    # Without this the timing-sensitive "right after wt new-tab" case
    # races against RunClientWrapper.ps1's startup.
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        $supervisors = @(Get-RegisteredSupervisors)
        foreach ($s in $supervisors) {
            $p = Get-Process -Id $s.ProcessId -ErrorAction SilentlyContinue
            if (-not $p) { continue }
            # Walk up via parent process IDs. .NET's
            # System.Diagnostics.Process doesn't expose parent on its
            # own, but NtQueryInformationProcess does. We compile a
            # tiny shim once per session.
            $wtPid = Resolve-AncestorWindowsTerminal -StartPid $p.Id -MaxDepth 6
            if ($wtPid) { return $wtPid }
        }
        Start-Sleep -Milliseconds 250
    }
    return 0
}

if (-not ('Win32ParentPid' -as [type])) {
    # Pulls the ParentProcessId out of NtQueryInformationProcess's
    # PROCESS_BASIC_INFORMATION struct. Faster and far more reliable
    # than WMI Win32_Process for "what's my parent?" lookups. The
    # underlying NT API is unaffected by the WMI service's state.
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class Win32ParentPid {
    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_BASIC_INFORMATION {
        public IntPtr Reserved1;
        public IntPtr PebBaseAddress;
        public IntPtr Reserved2_0;
        public IntPtr Reserved2_1;
        public IntPtr UniqueProcessId;
        public IntPtr InheritedFromUniqueProcessId;
    }

    [DllImport("ntdll.dll", SetLastError=true)]
    static extern int NtQueryInformationProcess(
        IntPtr ProcessHandle, int ProcessInformationClass,
        ref PROCESS_BASIC_INFORMATION ProcessInformation,
        int ProcessInformationLength, ref int ReturnLength);

    public static int Get(int pid) {
        try {
            var p = System.Diagnostics.Process.GetProcessById(pid);
            var pbi = new PROCESS_BASIC_INFORMATION();
            int returnLen = 0;
            int status = NtQueryInformationProcess(
                p.Handle, 0, ref pbi, Marshal.SizeOf(pbi), ref returnLen);
            if (status != 0) return 0;
            return pbi.InheritedFromUniqueProcessId.ToInt32();
        } catch {
            return 0;
        }
    }
}
'@
}

function Resolve-AncestorWindowsTerminal {
    param([int]$StartPid, [int]$MaxDepth = 6)
    $cur = $StartPid
    for ($d = 0; $d -lt $MaxDepth; $d++) {
        $parentPid = [Win32ParentPid]::Get($cur)
        if ($parentPid -le 0) { return 0 }
        $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
        if (-not $parent) { return 0 }
        if ($parent.ProcessName -match '^WindowsTerminal') {
            return $parentPid
        }
        $cur = $parentPid
    }
    return 0
}

if ($useWt) {
    # Give wt a moment to register the last tab before we touch it.
    Start-Sleep -Milliseconds 500
    $wtHostPid = Get-WtHostPidByWalk
    if ($wtHostPid) {
        Set-WtHostPid -ProcessId $wtHostPid
        Write-Host "Recorded wt host PID=$wtHostPid in stack state."
    } else {
        Write-Warning "Could not resolve the wt host PID. Stop-Stack will still kill supervisors; the wt window may need manual close."
    }

    # Re-minimize the wt window after all tabs are added. The first
    # `Start-Process wt -WindowStyle Minimized` creates the window
    # minimized, but each subsequent ``wt -w <name> new-tab ...`` call
    # brings the existing window to the foreground (that's wt's
    # focus-on-new-tab behavior, not configurable per-call). Net
    # effect: without this fix-up, the window ends up unminimized
    # after tab 1 even though the user asked for minimized.
    if ($Minimized -and $wtHostPid) {
        if (-not ('Win32WindowMinimize' -as [type])) {
            Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class Win32WindowMinimize {
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    public const int SW_SHOWMINNOACTIVE = 7;
}
'@
        }
        $wtProc = Get-Process -Id $wtHostPid -ErrorAction SilentlyContinue
        if ($wtProc -and $wtProc.MainWindowHandle -ne [IntPtr]::Zero) {
            [void][Win32WindowMinimize]::ShowWindow(
                $wtProc.MainWindowHandle,
                [Win32WindowMinimize]::SW_SHOWMINNOACTIVE)
            Write-Host "Re-minimized wt host window (PID=$wtHostPid)"
        }
    }
}

Write-Host ""
if ($useWt) {
    Write-Host "All $N clients launched as tabs in Windows Terminal window '$wtWindowName'."
    Write-Host "Click the wt taskbar icon to switch between actor-0 .. actor-$($N-1)."
    Write-Host "Stop-Stack.ps1 will clean them up (or close the wt window manually)."
} else {
    Write-Host "All $N clients launched as separate PowerShell windows."
    Write-Host "Close a window to stop supervising that client, or run Stop-Stack.ps1."
}
Write-Host ""
Write-Host "If the Unity windows don't appear, check the supervisor consoles for"
Write-Host "'Status = Exited: Restart...' loops - that means Unity is dying right"
Write-Host "after launch. Most common causes: another instance of the same .exe"
Write-Host "path is already running, or compose/scale.yml isn't up so port"
Write-Host "10001/10002/... has no listener."
