<#
.SYNOPSIS
    Shared registry of "things Start-Stack started" so Stop-Stack can
    find them again without needing WMI.

.DESCRIPTION
    Previously the lifecycle scripts (Start-Stack / Stop-Stack /
    Stop-Clients) located their own supervisor PowerShell processes by
    asking WMI for every powershell.exe on the system and filtering on
    CommandLine via ``Get-CimInstance Win32_Process``. When the
    Windows WMI service gets into a wedged state - which happens not-
    infrequently on Windows after suspend/resume cycles, .NET
    background work, or simply long uptime - those queries hang
    forever. The cmdlet has no enforceable timeout (the docs claim
    -OperationTimeoutSec but it doesn't reliably cancel a stuck WMI
    call), so the scripts would freeze with no signal of why.

    This module replaces the WMI dependency for the steady-state case:
    every supervisor self-registers a tiny JSON descriptor on disk
    when it starts and removes it when it exits. Stop-Stack just reads
    those files. A bounded-timeout WMI fallback (Invoke-WithTimeout)
    is still available for one-off "what if WMI is fine and there's a
    legacy supervisor from before this fix" cleanups, but if WMI is
    wedged it now degrades to a printed warning instead of an
    infinite hang.

    State dir layout, all under $env:TEMP\rl-roboracer-stack\:

        supervisors\<pid>.json    one file per supervisor process
        wt-host.json              the Windows Terminal window PID

    Each file holds a small JSON object with the PID, a created
    timestamp, and a "kind" tag we use to defend against PID re-use
    (we won't kill a PID whose current image is no longer the
    expected powershell.exe / WindowsTerminal.exe).

.NOTES
    Dot-source this file at the top of any script that needs the
    helpers:
        . (Join-Path $PSScriptRoot '_StackState.ps1')
#>

# State dir lives in $env:TEMP so it auto-cleans across reboots.
# Per-user, so multiple users on the same machine don't collide.
function Get-StackStateDir {
    $root = Join-Path $env:TEMP 'rl-roboracer-stack'
    if (-not (Test-Path -LiteralPath $root)) {
        New-Item -ItemType Directory -Force -Path $root | Out-Null
    }
    return $root
}

function Get-SupervisorsDir {
    $dir = Join-Path (Get-StackStateDir) 'supervisors'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    return $dir
}

function Get-WtHostStateFile {
    return Join-Path (Get-StackStateDir) 'wt-host.json'
}

# Register the current process (or another specified PID) as a
# supervisor we're tracking. Stored fields:
#   pid:      the PID to kill
#   index:    actor index (informational; used in Stop-Stack output)
#   exe:      target Unity .exe path (informational)
#   started:  ISO-8601 timestamp at registration
#   kind:     'supervisor' marker; lets the cleanup code defend
#             against PID re-use when the original supervisor has
#             already exited and the OS recycled its PID to e.g. an
#             unrelated explorer.exe.
function Register-Supervisor {
    param(
        [int]$ProcessId = $PID,
        [int]$Index = -1,
        [string]$Exe = ''
    )
    $entry = [ordered]@{
        pid     = $ProcessId
        index   = $Index
        exe     = $Exe
        started = (Get-Date).ToString('o')
        kind    = 'supervisor'
    }
    $path = Join-Path (Get-SupervisorsDir) "$ProcessId.json"
    # ConvertTo-Json -Depth 5 so nested objects (we don't currently
    # have any, but cheap insurance) get fully serialized rather than
    # truncating to System.Collections.Hashtable.
    $entry | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $path -Encoding UTF8
}

# Remove our own registration. Best-effort: callers should put this
# in a try/finally so it runs on Ctrl-C exit, but a hard kill bypasses
# finally - stale files are tolerated and reaped by the consistency
# check in Get-RegisteredSupervisors.
function Unregister-Supervisor {
    param([int]$ProcessId = $PID)
    $path = Join-Path (Get-SupervisorsDir) "$ProcessId.json"
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

# Return the live supervisor registrations. "Live" = the PID still
# exists AND its current image looks like a powershell.exe (so we
# don't accidentally kill a recycled PID belonging to something
# innocent). Stale .json files are removed as a side effect so the
# state dir doesn't accumulate cruft over time.
function Get-RegisteredSupervisors {
    $dir = Get-SupervisorsDir
    $entries = @()
    foreach ($file in @(Get-ChildItem -LiteralPath $dir -Filter '*.json' -File -ErrorAction SilentlyContinue)) {
        $raw = $null
        try {
            $raw = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        } catch {
            # Corrupt or partially-written file - drop it.
            Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue
            continue
        }
        if (-not $raw -or -not $raw.pid) { continue }

        $alive = Get-Process -Id $raw.pid -ErrorAction SilentlyContinue
        if (-not $alive) {
            # Supervisor died (probably crashed or was killed) without
            # running its finally cleanup. Reap the stale file.
            Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue
            continue
        }
        # PID re-use guard. If the live process is no longer a
        # powershell instance, the OS has recycled the PID since
        # registration; this is no longer our supervisor.
        if ($alive.ProcessName -notmatch '^(powershell|pwsh)$') {
            Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue
            continue
        }
        $entries += [pscustomobject]@{
            ProcessId = [int]$raw.pid
            Index     = if ($null -ne $raw.index)   { [int]$raw.index } else { -1 }
            Exe       = if ($null -ne $raw.exe)     { [string]$raw.exe } else { '' }
            Started   = if ($null -ne $raw.started) { [string]$raw.started } else { '' }
        }
    }
    # Plain return (no `,@(...)` wrapper). The historical wrap-in-a-
    # 1-element-array trick is meant to defeat PowerShell's single-
    # element-array unwrapping on function return, but when $entries
    # is itself @() the wrapper turns it into a 1-element array
    # containing @(), and the caller's foreach then iterates once
    # with an empty hashtable - which manifests as "killing supervisor
    # PID=0" in Stop-Stack. Callers always wrap with @() on receive
    # so a plain return is fine and avoids the empty-case footgun.
    return $entries
}

# Single-PID record for the Windows Terminal window that hosts the
# supervisor tabs. The wt window is shared across N supervisors so
# one file is enough.
function Set-WtHostPid {
    param([int]$ProcessId)
    if (-not $ProcessId -or $ProcessId -le 0) { return }
    $entry = [ordered]@{
        pid     = $ProcessId
        started = (Get-Date).ToString('o')
        kind    = 'wt-host'
    }
    $entry | ConvertTo-Json | Set-Content -LiteralPath (Get-WtHostStateFile) -Encoding UTF8
}

function Get-WtHostPid {
    $path = Get-WtHostStateFile
    if (-not (Test-Path -LiteralPath $path)) { return 0 }
    try {
        $raw = Get-Content -LiteralPath $path -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        return 0
    }
    if (-not $raw -or -not $raw.pid) { return 0 }
    $alive = Get-Process -Id $raw.pid -ErrorAction SilentlyContinue
    if (-not $alive -or $alive.ProcessName -notmatch '^WindowsTerminal') {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        return 0
    }
    return [int]$raw.pid
}

function Clear-WtHostPid {
    $path = Get-WtHostStateFile
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

# Wipe the entire state dir. Called at the start of a fresh
# Start-Clients run (so a previous run's stale entries can't confuse
# the next Stop-Stack).
function Clear-StackState {
    $root = Get-StackStateDir
    if (Test-Path -LiteralPath $root) {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Run a script block in a background job with a hard wall-clock
# timeout. Used to wrap WMI calls so a wedged Win32_Process query
# can't freeze the whole script - if the job doesn't finish in
# $TimeoutSec, we stop the job (the WMI call leaks in the background,
# but the foreground returns) and the caller gets $null + a warning.
#
# Returns the job's output (whatever the scriptblock emitted) on
# success, or $null on timeout.
function Invoke-WithTimeout {
    param(
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock,
        [int]$TimeoutSec = 8,
        [object[]]$ArgumentList = @(),
        [string]$Description = 'background task'
    )
    $job = Start-Job -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList
    if (Wait-Job -Job $job -Timeout $TimeoutSec) {
        $out = Receive-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        return $out
    } else {
        # PowerShell's -f operator binds tighter than string +, so the
        # placeholders {0}/{1} have to be in a single explicitly-
        # concatenated string (wrapped in parens) for the format to
        # apply across all of it.
        $msg = ("{0} timed out after {1}s; skipping. (If this keeps happening, " +
                "Windows WMI may be wedged: try 'net stop winmgmt /yes; net start winmgmt' " +
                "from an elevated shell.)") -f $Description, $TimeoutSec
        Write-Warning $msg
        # Leave the job orphaned in the background rather than blocking
        # on Stop-Job (which itself can hang on a stuck WMI runspace).
        # PowerShell process exit cleans it up; meanwhile we proceed.
        try { Stop-Job -Job $job -ErrorAction SilentlyContinue } catch { }
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch { }
        return $null
    }
}
