<#
.SYNOPSIS
    Watch a microduck policy walk in the browser (mjlab's viser viewer).

.DESCRIPTION
    Starts mjlab's ``play`` viewer inside the duck-trainer container and
    opens http://localhost:8090. The viewer's checkpoint picker lists every
    model_<N>.pt in the run directory and re-reads it when opened, so
    pointing it at a job that is still training lets you switch to newer
    checkpoints as they are saved and watch the gait improve.

    A resumed job spreads its checkpoints over one run directory per
    attempt; the picker only shows the directory of the checkpoint it
    started from (the latest one by default).

    The viewer shares the GPU with any running training job; it simulates
    a handful of robots, so the slowdown is small. Stop it with -Stop.

.PARAMETER JobId
    Microduck TRAIN job to watch. Defaults to the most recently created
    one that has a checkpoint.

.PARAMETER Checkpoint
    A specific model_<N>.pt, as a container path (/microduck_rl/logs/...)
    or a host path under saved_models\microduck. Overrides -JobId.

.PARAMETER Task
    mjlab task id. Defaults to the job's task_id.

.PARAMETER NumEnvs
    Robots shown in the viewer. Default 4.

.PARAMETER NoBrowser
    Don't open the viewer page once it is ready.

.PARAMETER Stop
    Stop a running viewer and exit.

.EXAMPLE
    .\scripts\Watch-Duck.ps1
.EXAMPLE
    .\scripts\Watch-Duck.ps1 -JobId 6af1fd71784ae02d91685ee9
.EXAMPLE
    .\scripts\Watch-Duck.ps1 -Stop
#>
param(
    [string]$JobId,
    [string]$Checkpoint,
    [string]$Task,
    [int]$NumEnvs = 4,
    [switch]$NoBrowser,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$composeArgs = @('-f', 'docker-compose.yml', '-f', 'compose/scale.yml', '-f', 'compose/duck.yml')
$url = 'http://localhost:8090'
$viewerLog = '/microduck_rl/logs/jobs/viewer.log'

function Invoke-Duck([string[]]$Command, [switch]$Detached) {
    $execArgs = @('compose') + $composeArgs + @('exec', '-T', '-w', '/microduck_rl')
    if ($Detached) { $execArgs += '-d' }
    & docker @($execArgs + 'duck-trainer' + $Command)
}

# "[/]" keeps the pattern from matching the shell that runs pkill/pgrep.
$viewerPattern = "'[/]opt/venv/bin/play '"
Invoke-Duck @('sh', '-c', "pkill -f $viewerPattern || true") | Out-Null
if ($Stop) {
    Write-Host 'Duck viewer stopped.'
    return
}

if ($Checkpoint) {
    $hostRoot = (Resolve-Path (Join-Path $RepoRoot '..\saved_models\microduck')).Path
    if ($Checkpoint.StartsWith($hostRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $Checkpoint = '/microduck_rl/logs' + $Checkpoint.Substring($hostRoot.Length).Replace('\', '/')
    }
    if (-not $Task) { throw '-Task is required with -Checkpoint.' }
} else {
    $query = @'
import json, os, pymongo
from bson import ObjectId
db = pymongo.MongoClient(os.environ["MONGO_URL"])["robotaxi"]
job_id = os.environ.get("WATCH_JOB_ID")
q = {"robot": "microduck", "job_type": "TRAIN", "duck_checkpoint": {"$ne": None}}
if job_id:
    q["_id"] = ObjectId(job_id) if ObjectId.is_valid(job_id) else job_id
job = db.jobs.find_one(q, sort=[("create_date", -1)])
print(json.dumps(job and {"id": str(job["_id"]), "task": job.get("task_id") or "Mjlab-Velocity-Flat-MicroDuck",
                          "checkpoint": job["duck_checkpoint"], "status": job.get("status")}))
'@
    # Piped through stdin: Windows PowerShell strips the double quotes from
    # native-command arguments, which would mangle the Python source.
    $execArgs = @('compose') + $composeArgs + @('exec', '-T', '-e', "WATCH_JOB_ID=$JobId", 'duck-trainer', 'python', '-')
    $job = ($query | & docker @execArgs | Select-Object -Last 1) | ConvertFrom-Json
    if (-not $job) { throw "No microduck TRAIN job with a checkpoint$(if ($JobId) { " for id $JobId" })." }
    $Checkpoint = $job.checkpoint
    if (-not $Task) { $Task = $job.task }
    Write-Host "Job $($job.id) ($($job.status)): $Checkpoint"
}

Invoke-Duck -Detached @('sh', '-c',
    "play $Task --checkpoint-file '$Checkpoint' --viewer viser --num-envs $NumEnvs > $viewerLog 2>&1")
Write-Host "Starting viewer for $Task (first start compiles GPU kernels, up to a few minutes)..."

$deadline = (Get-Date).AddMinutes(6)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    try {
        # Windows PowerShell tries localhost over IPv6 first and times out on
        # Docker Desktop's port forward; the IPv4 loopback answers at once.
        Invoke-WebRequest -Uri 'http://127.0.0.1:8090' -UseBasicParsing -TimeoutSec 3 | Out-Null
        Write-Host "Viewer ready: $url  (stop with .\scripts\Watch-Duck.ps1 -Stop)"
        if (-not $NoBrowser) { Start-Process $url }
        return
    } catch { }
    $running = Invoke-Duck @('sh', '-c', "pgrep -f $viewerPattern >/dev/null && echo yes || echo no")
    if (($running | Out-String).Trim() -ne 'yes') {
        Invoke-Duck @('tail', '-n', '30', $viewerLog)
        throw 'Viewer exited before it was ready; log above.'
    }
}
throw "Viewer did not come up within 6 minutes; see $viewerLog in the duck-trainer container."
