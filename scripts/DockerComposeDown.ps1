# Stop the RL stack and any Unity client window. Run from repo root.
docker compose stop

$clientName = "robotaxi gym level 1"
$processes = Get-Process -Name $clientName -ErrorAction SilentlyContinue
foreach ($p in $processes) { $p.Kill() }
Write-Host "done"
