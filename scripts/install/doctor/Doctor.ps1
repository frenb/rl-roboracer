<#
.SYNOPSIS
    Diagnoses an rl-roboracer install problem with a local open-weights model.

.DESCRIPTION
    Starts llama-server with a bundled model (or uses -LlmUrl), then lets the
    model investigate with read-only tools until it reports a root cause.
    Diagnosis only: nothing on the machine is changed. Every run is logged
    as JSON lines under %LOCALAPPDATA%\rl-roboracer\doctor\.

    See docs/install-doctor-plan.md and scripts/install/doctor/MODELS.md.

.EXAMPLE
    .\Doctor.ps1 -Problem "Training jobs fail with: DNS resolution failed for fly-brain:50061"

.EXAMPLE
    .\Doctor.ps1 -Problem "..." -Model gemma-4-E4B-it-Q4_K_M.gguf -Backend cpu
#>
[CmdletBinding()]
param(
    [string]$Problem = '',
    # Read the problem from a file instead (the installer passes it this way).
    [string]$ProblemFile = '',
    [string]$RepoDir = '',
    [string]$Model = '',
    [ValidateSet('', 'cuda', 'vulkan', 'cpu')][string]$Backend = '',
    [string]$LlmUrl = '',
    # Hosted model for evaluation, "anthropic|openai|xai:<model-id>"; key from
    # ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY. Sends logs and paths
    # from this PC to that provider.
    [string]$Remote = '',
    [int]$MaxSteps = 25,
    [int]$TimeoutMinutes = 15,
    [string]$ResultFile = '',
    [switch]$Quiet,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'
if (-not $RepoDir) { $RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path }
$script:DoctorRepoDir = $RepoDir
. (Join-Path $PSScriptRoot 'Llm.ps1')
. (Join-Path $PSScriptRoot 'Tools.ps1')
if ($ProblemFile) { $Problem = Get-Content -LiteralPath $ProblemFile -Raw -Encoding UTF8 }
if (-not $Problem.Trim()) { throw 'Describe the problem with -Problem or -ProblemFile.' }
$Problem = Protect-Secrets $Problem

function Write-Step([string]$Text) { if (-not $Quiet) { Write-Host $Text } }

# Keeps the conversation inside a 32K-token context (~3.5 characters per
# token, with room for the reply): the oldest tool outputs are cut to a stub.
function Compress-OldToolOutput($Messages, [int]$MaxChars = 80000) {
    $total = ($Messages | ForEach-Object { "$($_.content)".Length } | Measure-Object -Sum).Sum
    foreach ($m in $Messages) {
        if ($total -le $MaxChars) { break }
        $old = "$($m.content)"
        if ($m.role -ne 'tool' -or $old.Length -le 400 -or $old.EndsWith('[older output trimmed]')) { continue }
        $m.content = $old.Substring(0, 300) + "`n[older output trimmed]"
        $total -= ($old.Length - $m.content.Length)
    }
}

function Get-SystemPrompt {
    $services = if (Test-Path (Join-Path $RepoDir 'docker-compose.yml')) {
        ((Invoke-Docker ((Get-RepoComposeArgs) + @('config', '--services'))).Out.Trim() -split "`r?`n") -join ', '
    } else { "unknown: $RepoDir has no docker-compose.yml (the repo may not be cloned yet)" }
    @"
You are the InstallZero install doctor for rl-roboracer, a reinforcement-learning racing stack on a Windows PC. Find the root cause of the problem the user reports.

How the stack works:
- Docker Desktop (WSL 2 backend) runs a docker compose project from $RepoDir with services: $services.
- sim-controller runs the trainer (robotaxi.py) and TensorBoard. ros-server (and ros-server-1..3 for extra clients) bridge to Unity over ROS-TCP on host ports 10000-10003. mongo holds jobs and settings. dashboard is the web UI on http://127.0.0.1 (port 80). fly-brain is a gRPC server that fly courses reach at fly-brain:50061. madscientist proposes experiments.
- Unity simulator clients run natively on Windows, each supervised by scripts\RunClientWrapper.ps1 in its own PowerShell window. They are started by scripts\Start-Stack.ps1 or scripts\Start-Clients.ps1, or at sign-in by scripts\install\Start-ClientAtLogon.ps1. Client N connects to 127.0.0.1:(10000+N); its ros-server dials back to host.docker.internal:(5005+N). A client must connect after its ros-server started.
- Unity builds live under unity\Builds (latest, and per-client copies in instances\N) and ..\UnityBinary.

How to work:
- Gather evidence before concluding: check what is actually running, read logs and configuration. Prefer one specific check over guessing.
- You cannot change anything; every command is read-only. Put the fix in finish.
- Secret values (API keys, passwords) are never shown: tool output has them replaced by <secret:NAME> or <secret>, and .env shows only their length. That a value is masked tells you it is set; ask the user to check its content when that matters.
- The root cause is the underlying condition, not the symptom. When something is missing, check whether it is not configured at all or configured but not running, and say which.
- Call finish as soon as the evidence supports a root cause. You have at most $MaxSteps tool calls.

This PC: $((Get-CimInstance Win32_OperatingSystem).Caption), user $env:USERNAME, $(Get-Date -Format 'yyyy-MM-dd HH:mm').
"@
}

$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $env:LOCALAPPDATA 'rl-roboracer\doctor'
New-Item -ItemType Directory -Force $logDir | Out-Null
$transcript = Join-Path $logDir "$runId.jsonl"
function Write-Transcript($Entry) { Add-Content -Path $transcript -Value ($Entry | ConvertTo-Json -Depth 20 -Compress) -Encoding UTF8 }

$server = $null
$result = [ordered]@{ run_id = $runId; problem = $Problem; model = $Model; backend = $Backend; steps = 0; tool_calls = 0; invalid_calls = 0; refused_commands = 0; nudges = 0; server_errors = 0; prompt_tokens = 0; completion_tokens = 0
    finished = $false; root_cause = ''; evidence = ''; fix = ''; confidence = ''; seconds = 0; transcript = $transcript; error = '' }
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $llm = $null
    if ($Remote) {
        $llm = Get-RemoteLlm $Remote
        $LlmUrl = $llm.Url
        $result.model = $Remote; $result.backend = 'remote'
    } elseif (-not $LlmUrl) {
        Write-Step 'Starting the local model...'
        $server = Start-LlmServer -Model $Model -Backend $Backend
        $LlmUrl = $server.Url
        $result.model = $server.Model; $result.backend = $server.Backend
        Write-Step "  $($server.Model) on $($server.Backend), ready in $($server.LoadSeconds)s"
    }

    $messages = [System.Collections.ArrayList]@(
        @{ role = 'system'; content = (Get-SystemPrompt) },
        @{ role = 'user'; content = $Problem }
    )
    Write-Transcript @{ type = 'start'; problem = $Problem; model = $result.model; url = $LlmUrl; system = $messages[0].content }

    $finish = $null; $lastText = ''
    while (-not $finish) {
        if ($sw.Elapsed.TotalMinutes -ge $TimeoutMinutes) { $result.error = "timed out after $TimeoutMinutes minutes"; break }
        if ($result.tool_calls -ge $MaxSteps) {
            if ($result.nudges -ge 3) { break }
            $result.nudges++
            [void]$messages.Add(@{ role = 'user'; content = 'You have used all your tool calls. Call finish now with your best diagnosis.' })
        }

        Compress-OldToolOutput $messages
        # llama-server answers 500 when it cannot parse the model's tool-call
        # syntax; a resample at a higher temperature usually parses. Hosted
        # APIs also answer 429/5xx when busy; those get a backoff.
        $r = $null
        $attempt = 0
        foreach ($temperature in 0.2, 0.6, 0.9, 0.9) {
            $attempt++
            try { $r = Invoke-ChatTurn $llm $LlmUrl $messages $script:DoctorToolSchemas $temperature 4096; break }
            catch {
                $err = "$_"
                $detail = try { $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream()); $sr.ReadToEnd() } catch { '' }
                if ("$err $detail" -notmatch '429|500|502|503|529|Internal Server Error|does not match|overloaded|rate limit') { throw "$err $detail" }
                $result.server_errors++
                Write-Transcript @{ type = 'server_error'; temperature = $temperature; error = "$err $detail" }
                if ($llm) { Start-Sleep -Seconds (10 * $attempt) }
            }
        }
        if (-not $r) { throw 'the model failed to produce a usable reply four times in a row' }
        $result.steps++
        if ($r.usage) { $result.prompt_tokens += [int]$r.usage.prompt_tokens; $result.completion_tokens += [int]$r.usage.completion_tokens }
        $msg = $r.choices[0].message
        $calls = if ($msg.PSObject.Properties.Name -contains 'tool_calls' -and $msg.tool_calls) { @($msg.tool_calls) } else { @() }
        $text = ConvertTo-PlainText "$($msg.content)"
        Write-Transcript @{ type = 'assistant'; content = $text; tool_calls = $calls; usage = $r.usage }

        if (-not $calls) {
            # Some models write the finish call as JSON text instead of calling it.
            if ($text -match '(?s)\{.*"root_cause".*\}') {
                try { $finish = $Matches[0] | ConvertFrom-Json; Write-Transcript @{ type = 'finish_from_text' }; break } catch { }
            }
            if ($text.Trim()) { $lastText = $text }
            if ($result.nudges -ge 2) { break }
            $result.nudges++
            [void]$messages.Add(@{ role = 'assistant'; content = $text })
            [void]$messages.Add(@{ role = 'user'; content = 'Continue: use a tool to check something, or call finish with root_cause, evidence, fix and confidence.' })
            continue
        }

        $argJson = { param($raw) if ($raw -is [string]) { $raw } elseif ($null -eq $raw) { '{}' } else { $raw | ConvertTo-Json -Compress -Depth 10 } }
        # extra_content carries Gemini's thought signatures, which must come back unchanged.
        $echo = @($calls | ForEach-Object {
            $e = @{ id = $_.id; type = 'function'; function = @{ name = $_.function.name; arguments = (& $argJson $_.function.arguments) } }
            if ($_.PSObject.Properties.Name -contains 'extra_content') { $e.extra_content = $_.extra_content }
            $e
        })
        # Anthropic rejects empty text content alongside tool calls.
        $assistant = @{ role = 'assistant'; content = $(if ($llm -and -not $text.Trim()) { $null } else { $text }); tool_calls = $echo }
        if ($msg.PSObject.Properties.Name -contains '_responses_items') { $assistant._responses_items = $msg._responses_items }
        [void]$messages.Add($assistant)
        foreach ($c in $calls) {
            $result.tool_calls++
            $name = $c.function.name
            $raw = & $argJson $c.function.arguments
            $a = $null
            try { $a = if ($raw.Trim()) { $raw | ConvertFrom-Json } else { [pscustomobject]@{} } }
            catch { $result.invalid_calls++; $out = "error: arguments are not valid JSON: $raw" }
            if ($null -ne $a) {
                if ($name -eq 'finish') {
                    $finish = $a; $out = 'Recorded.'
                } else {
                    $argText = ($a | ConvertTo-Json -Compress -Depth 5)
                    Write-Step ("  [{0}] {1} {2}" -f $result.tool_calls, $name, $(if ($argText.Length -gt 120) { $argText.Substring(0, 120) + '...' } else { $argText }))
                    try { $out = Invoke-DoctorTool $name $a } catch { $out = "error: $_" }
                    if ($name -eq 'run_command' -and "$out".StartsWith('refused:')) { $result.refused_commands++ }
                    if ($out -match '^error: unknown tool') { $result.invalid_calls++ }
                }
            }
            $out = Limit-Text (ConvertTo-PlainText "$out")
            Write-Transcript @{ type = 'tool'; name = $name; arguments = "$($c.function.arguments)"; output = $out }
            [void]$messages.Add(@{ role = 'tool'; tool_call_id = $c.id; content = $out })
        }
    }

    if ($finish) {
        $result.finished = $true
        $result.root_cause = ConvertTo-PlainText "$($finish.root_cause)"
        $result.evidence = ConvertTo-PlainText "$($finish.evidence)"
        $result.fix = ConvertTo-PlainText "$($finish.fix)"
        $result.confidence = "$($finish.confidence)"
    } elseif ($lastText) {
        $result.root_cause = $lastText
        $result.confidence = 'low'
    }
} catch {
    $result.error = "$_"
} finally {
    Stop-LlmServer $server
    $result.seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    Write-Transcript @{ type = 'result'; result = $result }
}

if ($ResultFile) { $result | ConvertTo-Json -Depth 5 | Set-Content -Path $ResultFile -Encoding UTF8 }
if (-not $Quiet) {
    Write-Host ''
    if ($result.error) { Write-Host "Error: $($result.error)" -ForegroundColor Red }
    Write-Host "Root cause ($($result.confidence)): $($result.root_cause)" -ForegroundColor Cyan
    if ($result.evidence) { Write-Host "Evidence: $($result.evidence)" }
    if ($result.fix) { Write-Host "Fix: $($result.fix)" -ForegroundColor Green }
    Write-Host "($($result.tool_calls) tool calls, $($result.seconds)s, transcript $transcript)"
}
if ($PassThru) { [pscustomobject]$result }
