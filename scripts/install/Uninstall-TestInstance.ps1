<#
.SYNOPSIS
    Remove a test install made with Install.ps1 -ProjectName/-ImagePrefix, so
    the next test starts from nothing.

.DESCRIPTION
    Stops the instance's Unity clients and supervisors, takes its compose
    project down with its volumes, deletes its prefixed images, prunes the
    Docker build cache (so the next build cannot reuse cached layers) and
    deletes the install folder.

    Refuses to run unless <InstallDir>\rl-roboracer\.env sets both
    COMPOSE_PROJECT_NAME and IMAGE_PREFIX, i.e. unless the folder really is
    an isolated test install. A normal install shares image names with any
    other install on the machine, and removing it would remove those too.

.PARAMETER InstallDir
    The -InstallDir the test install was made with.

.PARAMETER KeepBuildCache
    Leave the Docker build cache alone (faster next build, but the next
    test no longer proves a from-scratch build).
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$InstallDir,
    [switch]$KeepBuildCache
)

$ErrorActionPreference = 'Stop'
$repo = Join-Path $InstallDir 'rl-roboracer'
$envFile = Join-Path $repo '.env'

if (-not (Test-Path $InstallDir)) { Write-Host "Nothing to remove: $InstallDir does not exist."; exit 0 }

function Get-EnvValue([string]$Key) {
    if (-not (Test-Path $envFile)) { return $null }
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -Last 1
    if ($line) { ($line -split '=', 2)[1].Trim() }
}

$project = Get-EnvValue 'COMPOSE_PROJECT_NAME'
$prefix  = Get-EnvValue 'IMAGE_PREFIX'
if (-not $project -or -not $prefix) {
    throw "$envFile does not set both COMPOSE_PROJECT_NAME and IMAGE_PREFIX, so this does not look like an isolated test install. Not touching it."
}
if ($project -eq 'rl-roboracer') { throw "COMPOSE_PROJECT_NAME is the default 'rl-roboracer'; refusing to remove the main project." }

Write-Host "Removing test install '$project' (images '$prefix*') at $InstallDir"

$buildsRoot = Join-Path $repo 'unity\Builds'
$old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
try {
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$repo*" -and $_.ProcessId -ne $PID } |
        ForEach-Object { Write-Host "  stopping supervisor pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($buildsRoot, [StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object { Write-Host "  stopping Unity pid $($_.Id)"; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

    if (Test-Path (Join-Path $repo 'docker-compose.yml')) {
        Push-Location $repo
        Write-Host '  docker compose down --volumes'
        docker compose -f docker-compose.yml -f compose/scale.yml down --volumes --remove-orphans 2>&1 | Out-Null
        Pop-Location
    }

    $images = @(docker images --format '{{.Repository}}:{{.Tag}}' | Where-Object { $_ -like "$prefix*" })
    foreach ($img in $images) { Write-Host "  removing image $img"; docker rmi -f $img 2>&1 | Out-Null }

    if (-not $KeepBuildCache) {
        Write-Host '  pruning Docker build cache'
        docker builder prune -af 2>&1 | Select-Object -Last 1 | ForEach-Object { Write-Host "    $_" }
    }
} finally {
    $ErrorActionPreference = $old
}

Write-Host "  deleting $InstallDir"
Remove-Item -LiteralPath $InstallDir -Recurse -Force
Write-Host 'Done.'
