---
name: rl-roboracer
source: https://github.com/frenb/rl-roboracer
summary: RL training stack for the Roboracer (Docker Desktop on WSL 2, NVIDIA GPU, Unity gyms), on Windows
catalog: scripts/install/catalog
---

# rl-roboracer install

Installs the rl-roboracer training stack on a Windows PC with an NVIDIA GPU,
from this repo's folder: WSL 2, Git and Docker Desktop with the GPU visible
to containers, the data folders beside the repo, the three Docker images,
the starter Unity gyms and expert demonstrations, then the running stack
(ros-server, MongoDB, mongo-express, the trainer, TensorBoard on
http://localhost:6006, fly-brain and the dashboard on http://localhost) with
the starter documents seeded into MongoDB.

Run it at the repo root: `installzero install`. The data folders
(`saved_models`, `mongodb`, `tfrecords`) and the unpacked gyms
(`UnityBinary`) go in the folder that holds the repo, where
`docker-compose.yml` expects them (`../saved_models`, `../mongodb`).

Step 1 installs WSL 2 when it is missing, which needs a restart of Windows:
the install stops there and offers to continue at your next sign-in (or
run `installzero install` again after restarting); steps already done are
skipped. Building the images takes 20 to 60 minutes the
first time.

Docker Desktop is free for personal use, education, non-commercial open
source and small businesses (under 250 employees and under USD 10M
revenue); larger organisations need a paid subscription. Approving step 3
accepts the Docker Subscription Service Agreement:
https://www.docker.com/legal/docker-subscription-service-agreement/

The last three steps start the Unity client, add a shortcut to your Startup
folder that starts it again at each sign-in, and run a first short TRAIN
job (2,000 iterations, up to 45 minutes) to show the stack trains end to end.
To leave that job out: `installzero install --skip firstjob`. This file runs
one Unity client; `scripts/install/Install.ps1 -Clients N` runs up to four.

## Step 1: wsl - WSL 2

Why: Docker Desktop runs its engine in WSL 2.

```requires
os: windows
arch: x64
gpu: nvidia
min_driver: 551.61
min_ram_gb: 16
min_disk_gb: 100
```

```detect powershell
wsl.exe --version | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }
wsl.exe --status | Out-Null
exit $LASTEXITCODE
```

```install powershell
$p = Start-Process wsl.exe -ArgumentList '--install', '--no-distribution' -Verb RunAs -Wait -PassThru
if ($p.ExitCode -ne 0) { Write-Error "wsl --install exited $($p.ExitCode)"; exit 1 }
exit 3010
```

```verify powershell
wsl.exe --version | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }
wsl.exe --status | Out-Null
exit $LASTEXITCODE
```

## Step 2: git - Git for Windows

Why: Docker Compose builds and the repo's scripts use git.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
git --version
exit $LASTEXITCODE
```

```install powershell
winget install --id Git.Git --exact --silent --source winget --accept-package-agreements --accept-source-agreements
exit $LASTEXITCODE
```

```verify powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
git --version
exit $LASTEXITCODE
```

## Step 3: docker - Docker Desktop

Why: runs the stack's containers, with the GPU passed through from WSL 2.

```detect powershell
if (-not (Test-Path (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))) { exit 1 }
```

```install powershell
winget install --id Docker.DockerDesktop --exact --source winget --accept-package-agreements --accept-source-agreements --override 'install --quiet --accept-license --backend=wsl-2'
if (-not (Test-Path (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))) { Write-Error "Docker Desktop did not install (winget exit $LASTEXITCODE)"; exit 1 }
```

```verify powershell
if (-not (Test-Path (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))) { exit 1 }
```

## Step 4: engine - Docker engine running, GPU visible

Why: the images build and the trainer runs on the GPU inside containers.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
docker info --format '{{.ServerVersion}}' 2>$null
exit $LASTEXITCODE
```

```install powershell phase=docker-start
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
docker info 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { exit 0 }
$session = [Diagnostics.Process]::GetCurrentProcess().SessionId
$other = @(Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -ne $session })
if ($other) { Write-Error 'Docker Desktop is running for another signed-in Windows account; it runs for one account at a time. Quit it there, then run installzero install again.'; exit 1 }
Start-Process (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe')
Write-Output 'Starting Docker Desktop (the first start can take a few minutes) ...'
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { exit 0 }
    Start-Sleep -Seconds 10
}
docker info
exit 1
```

```verify powershell phase=docker-gpu
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$out = docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi -L 2>&1 | Out-String
Write-Output $out.Trim()
if ($LASTEXITCODE -ne 0 -or $out -notmatch 'GPU 0') { exit 1 }
```

## Step 5: data - Data folders and .env

Why: model checkpoints, the MongoDB database and demo recordings live
beside the repo; .env holds the stack's settings and a Mongo password.

```detect powershell
$parent = Split-Path '{{INSTALL_DIR}}'
foreach ($d in 'saved_models', 'mongodb', 'tfrecords') { if (-not (Test-Path (Join-Path $parent $d))) { exit 1 } }
if (-not (Test-Path '{{INSTALL_DIR}}\.env')) { exit 1 }
```

```install powershell
$parent = Split-Path '{{INSTALL_DIR}}'
foreach ($d in 'saved_models', 'mongodb', 'tfrecords') { New-Item -ItemType Directory -Force (Join-Path $parent $d) | Out-Null }
$envFile = '{{INSTALL_DIR}}\.env'
if (-not (Test-Path $envFile)) {
    $lines = @(Get-Content '{{INSTALL_DIR}}\.env.example')
    $lines += ''
    $lines += '# ---- Written by installzero install ----'
    # Mongo applies the password only when it creates the database.
    if (@(Get-ChildItem (Join-Path $parent 'mongodb') -Force).Count -gt 0) {
        Write-Output 'The mongodb folder already holds a database: keeping its password (set MONGO_ROOT_PASSWORD in .env if it is not "example").'
    } else {
        $chars = [char[]]'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $lines += 'MONGO_ROOT_PASSWORD=' + (-join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] }))
    }
    [IO.File]::WriteAllLines($envFile, [string[]]$lines)
}
```

```verify powershell
$parent = Split-Path '{{INSTALL_DIR}}'
foreach ($d in 'saved_models', 'mongodb', 'tfrecords') { if (-not (Test-Path (Join-Path $parent $d))) { Write-Output "$d is missing"; exit 1 } }
if (-not (Test-Path '{{INSTALL_DIR}}\.env')) { exit 1 }
```

## Step 6: ros-server - ros-server image

Why: the ROS bridge between the Unity gyms and the trainer.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'ros-server'.image
docker image inspect $img 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }
$check = "set -eo pipefail`nsource /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash`npython3 -c 'from roboracer.msg import CarSceneData, SimCommand, SimStatus, Camera; from ros_tcp_endpoint import TcpServer; from virtual_endpoint import VirtualNode; print(""imports ok"")'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

```install powershell phase=build timeout=90m
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
docker compose build --progress=plain ros-server
exit $LASTEXITCODE
```

```verify powershell phase=verify
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'ros-server'.image
$check = "set -eo pipefail`nsource /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash`npython3 -c 'from roboracer.msg import CarSceneData, SimCommand, SimStatus, Camera; from ros_tcp_endpoint import TcpServer; from virtual_endpoint import VirtualNode; print(""imports ok"")'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

## Step 7: sim-controller - Trainer image

Why: the trainer (TensorFlow, TF-Agents, Reverb) and TensorBoard.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'sim-controller'.image
docker image inspect $img 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }
$check = "set -eo pipefail`nkb=`$(du -s /usr/local/lib/python3.8/dist-packages/tensorflow | cut -f1)`n[ `"`$kb`" -gt 900000 ] || { echo `"tensorflow folder is only `${kb} KB - corrupted layer`"; exit 1; }`npython3 -c 'import tensorflow, tf_agents, reverb, pymongo; from tensorboard_plugin_profile import profile_plugin; print(""tf"", tensorflow.__version__)' 2>&1 | tail -1`nsource /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash`npython3 -c 'from roboracer.msg import CarSceneData, SimCommand; print(""roboracer msgs ok"")'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

```install powershell phase=build timeout=90m
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
docker compose build --progress=plain sim-controller
exit $LASTEXITCODE
```

```verify powershell phase=verify
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'sim-controller'.image
$check = "set -eo pipefail`nkb=`$(du -s /usr/local/lib/python3.8/dist-packages/tensorflow | cut -f1)`n[ `"`$kb`" -gt 900000 ] || { echo `"tensorflow folder is only `${kb} KB - corrupted layer`"; exit 1; }`npython3 -c 'import tensorflow, tf_agents, reverb, pymongo; from tensorboard_plugin_profile import profile_plugin; print(""tf"", tensorflow.__version__)' 2>&1 | tail -1`nsource /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash`npython3 -c 'from roboracer.msg import CarSceneData, SimCommand; print(""roboracer msgs ok"")'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

## Step 8: fly-brain - fly-brain image

Why: the fly connectome simulator (CuPy) the FlyBrain gym talks to.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'fly-brain'.image
docker image inspect $img 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }
$check = "set -eo pipefail`npython -c 'import flybrain, cupy, grpc, google.protobuf; print(""grpc"", grpc.__version__)'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

```install powershell phase=build timeout=90m
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
docker compose build --progress=plain fly-brain
exit $LASTEXITCODE
```

```verify powershell phase=verify
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$img = (docker compose config --format json | ConvertFrom-Json).services.'fly-brain'.image
$check = "set -eo pipefail`npython -c 'import flybrain, cupy, grpc, google.protobuf; print(""grpc"", grpc.__version__)'"
docker run --rm --entrypoint bash $img -c "echo $([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))) | base64 -d | bash"
exit $LASTEXITCODE
```

## Step 9: gyms - Starter Unity gyms

Why: the simulated courses the trainer drives; the default gym (v13) is
also copied to unity\Builds\latest, where the client launcher starts it.

```detect powershell
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
$exe = 'robotaxi gym level 1.exe'
foreach ($f in 'wCourseJetRacer2026.09.20-v13', 'wCourseJetRacer2026.09.28-v49') {
    if (-not (Test-Path (Join-Path $bin "$f\$exe")) -or -not (Test-Path (Join-Path $bin "$f\.roboracer-client.json"))) { exit 1 }
}
$marker = '{{INSTALL_DIR}}\unity\Builds\latest\.roboracer-client.json'
if (-not (Test-Path $marker) -or (Get-Content $marker -Raw | ConvertFrom-Json).gym -ne 'wCourseJetRacer2026.09.20-v13') { exit 1 }
```

```install powershell phase=download
$ProgressPreference = 'SilentlyContinue'
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
$exe = 'robotaxi gym level 1.exe'
$base = 'https://github.com/frenb/rl-roboracer/releases/download/v0.1'
$gyms = @(
    @{ Name = 'wCourseJetRacer2026.09.20-v13'; Folder = 'wCourseJetRacer2026.09.20-v13' },
    @{ Name = 'FlyBrain-wCourseJetRacer2026.09.28-v49'; Folder = 'wCourseJetRacer2026.09.28-v49' }
)
$tmp = Join-Path $env:TEMP 'rl-roboracer-gyms'
New-Item -ItemType Directory -Force $tmp, $bin | Out-Null
foreach ($g in $gyms) {
    $dest = Join-Path $bin $g.Folder
    if ((Test-Path (Join-Path $dest $exe)) -and (Test-Path (Join-Path $dest '.roboracer-client.json'))) { continue }
    $name = "roboracer-gym-$($g.Folder).zip"
    $zip = Join-Path $tmp $name
    Write-Output "Downloading $base/$name"
    Invoke-WebRequest "$base/$name" -OutFile $zip -UseBasicParsing
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    try {
        Invoke-WebRequest "$base/$name.sha256" -OutFile "$zip.sha256" -UseBasicParsing
        $want = ((Get-Content "$zip.sha256" -Raw).Trim() -split '\s+')[0].ToLower()
        if ($actual -ne $want) { Write-Error "checksum mismatch for $name (expected $want, got $actual)"; exit 1 }
    } catch [Net.WebException] { Write-Output "No checksum published for $name; not verified." }
    $unpack = Join-Path $tmp 'unpack'
    if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
    Expand-Archive -LiteralPath $zip -DestinationPath $unpack
    $root = $unpack
    $top = @(Get-ChildItem $unpack)
    if ($top.Count -eq 1 -and $top[0].PSIsContainer) { $root = $top[0].FullName }
    if (-not (Test-Path (Join-Path $root $exe))) { Write-Error "$name has no '$exe' at its top level"; exit 1 }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Move-Item -LiteralPath $root -Destination $dest
    [pscustomobject]@{ gym = $g.Name; source = "$base/$name"; sha256 = $actual; installedAt = (Get-Date).ToString('s') } |
        ConvertTo-Json | Set-Content -Path (Join-Path $dest '.roboracer-client.json') -Encoding UTF8
    Remove-Item $zip, "$zip.sha256" -Force -ErrorAction SilentlyContinue
}
$latest = '{{INSTALL_DIR}}\unity\Builds\latest'
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
New-Item -ItemType Directory -Force $latest | Out-Null
Copy-Item -Path (Join-Path $bin 'wCourseJetRacer2026.09.20-v13\*') -Destination $latest -Recurse -Force
```

```verify powershell
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
foreach ($f in 'wCourseJetRacer2026.09.20-v13', 'wCourseJetRacer2026.09.28-v49') {
    if (-not (Test-Path (Join-Path $bin "$f\robotaxi gym level 1.exe"))) { Write-Output "$f is missing"; exit 1 }
}
if (-not (Test-Path '{{INSTALL_DIR}}\unity\Builds\latest\robotaxi gym level 1.exe')) { exit 1 }
```

## Step 10: demos - Expert demonstrations

Why: the donut course's TRAIN jobs start from these recordings.

```detect powershell
$dir = Join-Path (Split-Path '{{INSTALL_DIR}}') 'tfrecords\job_64168c1b58d4d8ccdb76e721'
if (-not @(Get-ChildItem $dir -Filter '*.tfrecord' -File -ErrorAction SilentlyContinue)) { exit 1 }
```

```install powershell phase=download
$ProgressPreference = 'SilentlyContinue'
$dir = Join-Path (Split-Path '{{INSTALL_DIR}}') 'tfrecords\job_64168c1b58d4d8ccdb76e721'
$base = 'https://github.com/frenb/rl-roboracer/releases/download/v0.1'
$tmp = Join-Path $env:TEMP 'rl-roboracer-demos'
New-Item -ItemType Directory -Force $tmp, $dir | Out-Null
$zip = Join-Path $tmp 'roboracer-demos-donut.zip'
Invoke-WebRequest "$base/roboracer-demos-donut.zip" -OutFile $zip -UseBasicParsing
try {
    Invoke-WebRequest "$base/roboracer-demos-donut.zip.sha256" -OutFile "$zip.sha256" -UseBasicParsing
    $want = ((Get-Content "$zip.sha256" -Raw).Trim() -split '\s+')[0].ToLower()
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $want) { Write-Error "checksum mismatch for roboracer-demos-donut.zip (expected $want, got $actual)"; exit 1 }
} catch [Net.WebException] { Write-Output 'No checksum published for roboracer-demos-donut.zip; not verified.' }
$unpack = Join-Path $tmp 'unpack'
if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
Expand-Archive -LiteralPath $zip -DestinationPath $unpack
$records = @(Get-ChildItem $unpack -Recurse -Filter '*.tfrecord' -File)
if (-not $records) { Write-Error 'roboracer-demos-donut.zip contains no .tfrecord files'; exit 1 }
$records | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $dir -Force }
Remove-Item $tmp -Recurse -Force
```

```verify powershell
$dir = Join-Path (Split-Path '{{INSTALL_DIR}}') 'tfrecords\job_64168c1b58d4d8ccdb76e721'
$n = @(Get-ChildItem $dir -Filter '*.tfrecord' -File -ErrorAction SilentlyContinue).Count
Write-Output "$n recording(s)"
if ($n -eq 0) { exit 1 }
```

## Step 11: start - Start the stack

Why: brings up ros-server (port 10000), MongoDB, mongo-express, the
trainer with TensorBoard (6006), fly-brain and the dashboard (80).

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$want = 'ros-server', 'mongo', 'mongo-express', 'sim-controller', 'dashboard', 'fly-brain'
$running = @(docker compose ps --status running --services 2>$null | ForEach-Object { $_.Trim() })
if (@($want | Where-Object { $running -notcontains $_ }).Count -gt 0) { exit 1 }
foreach ($port in 10000, 6006) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $port) } catch { exit 1 } finally { $c.Close() }
}
```

```install powershell phase=compose-up
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
function Wait-Port([int]$Port, [int]$Minutes) {
    $deadline = (Get-Date).AddMinutes($Minutes)
    while ((Get-Date) -lt $deadline) {
        $c = New-Object Net.Sockets.TcpClient
        try { $c.Connect('127.0.0.1', $Port); return $true } catch { Start-Sleep -Seconds 5 } finally { $c.Close() }
    }
    $false
}
$services = 'ros-server', 'mongo', 'mongo-express', 'sim-controller', 'dashboard', 'fly-brain'
$envFile = '{{INSTALL_DIR}}\.env'
$lines = @(Get-Content $envFile)
if ($lines -notcontains 'NUM_ENVS=1') {
    $lines = @($lines | Where-Object { $_ -notmatch '^\s*NUM_ENVS\s*=' }) + 'NUM_ENVS=1'
    [IO.File]::WriteAllLines($envFile, [string[]]$lines)
}
# Mongo runs as uid 1001; a data folder Docker creates on a Windows drive is root-owned.
if (-not (Get-ChildItem (Join-Path (Split-Path '{{INSTALL_DIR}}') 'mongodb') -Force | Select-Object -First 1)) {
    docker compose run --rm --no-deps --user root --entrypoint chown mongo -R 1001:1001 /bitnami/mongodb
}
docker compose up -d --no-build @services
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 20
$failing = @(docker compose ps -a --format '{{.Service}}|{{.State}}' | Where-Object { $_ -match '\|(restarting|exited|dead)' } | ForEach-Object { ($_ -split '\|')[0] } | Where-Object { $services -contains $_ })
if ($failing) { docker compose logs --no-color --tail 60 @failing; Write-Error "these services keep crashing: $($failing -join ', ')"; exit 1 }
if (-not (Wait-Port 10000 2)) { Write-Error 'ros-server did not open port 10000 within 2 minutes'; exit 1 }
if (-not (Wait-Port 6006 3)) { Write-Error 'TensorBoard did not open port 6006 within 3 minutes'; exit 1 }
Write-Output 'Waiting for fly-brain (its first start downloads the ~260 MB connectome) ...'
$deadline = (Get-Date).AddMinutes(15)
do {
    docker compose exec -T fly-brain python -c "import socket; socket.create_connection(('127.0.0.1', 50061), 2)" 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 10
} while ((Get-Date) -lt $deadline)
if ($LASTEXITCODE -ne 0) { docker compose logs --no-color --tail 60 fly-brain; Write-Error 'fly-brain did not open port 50061 within 15 minutes'; exit 1 }
if (-not (Wait-Port 80 5)) { Write-Output 'The dashboard is still starting (its first start runs npm install).' }
```

```verify powershell phase=container
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$want = 'ros-server', 'mongo', 'mongo-express', 'sim-controller', 'dashboard', 'fly-brain'
$running = @(docker compose ps --status running --services | ForEach-Object { $_.Trim() })
$missing = @($want | Where-Object { $running -notcontains $_ })
if ($missing) { Write-Output "not running: $($missing -join ', ')"; exit 1 }
foreach ($port in 10000, 6006) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $port) } catch { Write-Output "nothing answers on port $port"; exit 1 } finally { $c.Close() }
}
```

## Step 12: seed - Starter documents in MongoDB

Why: the starter gyms, reward design and experiment designs that new TRAIN
jobs pick from (scripts\install\seed\seed.json); existing documents only
gain the fields they are missing.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
$seed = (Get-Content '{{INSTALL_DIR}}\scripts\install\seed\seed.json' -Raw).Replace('{{UNITY_BINARY_DIR}}', $bin.Replace('\', '\\')) | ConvertFrom-Json
$want = 0; $checks = @()
foreach ($coll in $seed.PSObject.Properties.Name) {
    foreach ($doc in $seed.$coll) {
        $want++
        $id = if ($doc._id -is [string]) { "'$($doc._id)'" } else { "ObjectId('$($doc._id.'$oid')')" }
        $fields = @($doc.PSObject.Properties.Name | Where-Object { $_ -ne '_id' } | ForEach-Object { "'$_': {`$exists: true}" })
        $checks += "db.getCollection('$coll').countDocuments({_id: $id, $($fields -join ', ')})"
    }
}
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
$have = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval "print($($checks -join ' + '))" 2>$null |
    Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -Last 1
if (-not $have -or [int]$have -ne $want) { exit 1 }
```

```install powershell phase=container
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
# A new database restarts once after its first-time setup: wait for three pings in a row.
$streak = 0; $deadline = (Get-Date).AddMinutes(4)
while ($streak -lt 3 -and (Get-Date) -lt $deadline) {
    $ok = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval 'print(db.runCommand({ping: 1}).ok)' 2>$null
    if ($LASTEXITCODE -eq 0 -and ($ok -match '^1\s*$')) { $streak++ } else { $streak = 0 }
    Start-Sleep -Seconds 5
}
if ($streak -lt 3) { docker compose logs --no-color --tail 60 mongo; Write-Error 'MongoDB did not answer within 4 minutes'; exit 1 }
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
$tmp = Join-Path $env:TEMP 'rl-seed.json'
[IO.File]::WriteAllText($tmp, (Get-Content '{{INSTALL_DIR}}\scripts\install\seed\seed.json' -Raw).Replace('{{UNITY_BINARY_DIR}}', $bin.Replace('\', '\\')), (New-Object Text.UTF8Encoding $false))
docker compose cp $tmp mongo:/tmp/rl-seed.json
if ($LASTEXITCODE -ne 0) { exit 1 }
docker compose cp '{{INSTALL_DIR}}\scripts\install\seed\seed.js' mongo:/tmp/rl-seed.js
if ($LASTEXITCODE -ne 0) { exit 1 }
$out = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --file /tmp/rl-seed.js 2>&1 | Out-String
Write-Output ($out -split "`n" | Where-Object { $_ -match '^SEEDED ' } | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0 -or $out -notmatch '(?m)^SEEDED ') { Write-Output $out; exit 1 }
Remove-Item $tmp -Force
```

```verify powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$bin = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary'
$seed = (Get-Content '{{INSTALL_DIR}}\scripts\install\seed\seed.json' -Raw).Replace('{{UNITY_BINARY_DIR}}', $bin.Replace('\', '\\')) | ConvertFrom-Json
$want = 0; $checks = @()
foreach ($coll in $seed.PSObject.Properties.Name) {
    foreach ($doc in $seed.$coll) {
        $want++
        $id = if ($doc._id -is [string]) { "'$($doc._id)'" } else { "ObjectId('$($doc._id.'$oid')')" }
        $fields = @($doc.PSObject.Properties.Name | Where-Object { $_ -ne '_id' } | ForEach-Object { "'$_': {`$exists: true}" })
        $checks += "db.getCollection('$coll').countDocuments({_id: $id, $($fields -join ', ')})"
    }
}
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
$have = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval "print($($checks -join ' + '))" |
    Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -Last 1
Write-Output "$have of $want starter documents"
if (-not $have -or [int]$have -ne $want) { exit 1 }
```

## Step 13: clients - Start the Unity client

Why: training drives the gym through a Unity client on this PC, supervised
by scripts\RunClientWrapper.ps1 (which restarts it if it exits). Unity sends
its ROS handshake only at startup, so a client started before the current
ros-server is restarted too.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$builds = '{{INSTALL_DIR}}\unity\Builds'
$clients = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($builds, [StringComparison]::OrdinalIgnoreCase) -and $_.Path -notmatch 'UnityCrashHandler' })
if ($clients.Count -ne 1 -or -not $clients[0].StartTime) { exit 1 }
if (Get-NetTCPConnection -OwningProcess $clients[0].Id -RemotePort 10000 -State Established -ErrorAction SilentlyContinue) { exit 0 }
$since = $clients[0].StartTime.ToUniversalTime().AddSeconds(-5).ToString('yyyy-MM-ddTHH:mm:ssZ')
if (-not (docker compose logs --no-color --since $since ros-server 2>$null | Select-String 'ROS-Unity Handshake received')) { exit 1 }
```

```install powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$builds = '{{INSTALL_DIR}}\unity\Builds'
# The wrapper relaunches a client it sees exit, so it goes first.
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
    Where-Object { $_.CommandLine -like '*{{INSTALL_DIR}}\scripts\RunClientWrapper.ps1*' -or $_.CommandLine -like '*{{INSTALL_DIR}}\scripts\install\Start-ClientAtLogon.ps1*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($builds, [StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
$gym = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary\wCourseJetRacer2026.09.20-v13\robotaxi gym level 1.exe'
Start-Process -FilePath powershell.exe -WindowStyle Minimized -WorkingDirectory '{{INSTALL_DIR}}' -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"{{INSTALL_DIR}}\scripts\install\Start-ClientAtLogon.ps1`" -N 1 -GymSource `"$gym`""
Write-Output 'Unity client starting (minimized); waiting for it to connect to ros-server ...'
```

```verify powershell phase=unity-client
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$builds = '{{INSTALL_DIR}}\unity\Builds'
$deadline = (Get-Date).AddMinutes(3)
do {
    $client = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($builds, [StringComparison]::OrdinalIgnoreCase) -and $_.Path -notmatch 'UnityCrashHandler' } | Select-Object -First 1
    if ($client -and $client.StartTime) {
        $since = $client.StartTime.ToUniversalTime().AddSeconds(-5).ToString('yyyy-MM-ddTHH:mm:ssZ')
        $line = docker compose logs --no-color --since $since ros-server 2>$null | Select-String 'ROS-Unity Handshake received' | Select-Object -Last 1
        if ($line -or (Get-NetTCPConnection -OwningProcess $client.Id -RemotePort 10000 -State Established -ErrorAction SilentlyContinue)) {
            Write-Output "Unity pid $($client.Id) connected to ros-server"
            exit 0
        }
    }
    Start-Sleep -Seconds 10
} while ((Get-Date) -lt $deadline)
$log = '{{INSTALL_DIR}}\unity\Builds\latest\Player.log'
if (Test-Path $log) { Get-Content $log -Tail 120 }
Write-Output "the Unity client did not connect to ros-server within 3 minutes (log: $log)"
exit 1
```

```uninstall powershell
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
    Where-Object { $_.CommandLine -like '*{{INSTALL_DIR}}\scripts\RunClientWrapper.ps1*' -or $_.CommandLine -like '*{{INSTALL_DIR}}\scripts\install\Start-ClientAtLogon.ps1*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith('{{INSTALL_DIR}}\unity\Builds', [StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
```

## Step 14: autostart - Start the Unity client at sign-in

Why: Docker brings the containers back by itself after a restart; the Unity
client is a Windows program and does not come back unless something starts
it. This puts a shortcut in your Startup folder that waits for Docker and
ros-server, then starts the client (the same shortcut Install.ps1 makes).

```detect powershell
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'rl-roboracer Unity client.lnk'
if (-not (Test-Path $lnk)) { exit 1 }
$gym = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary\wCourseJetRacer2026.09.20-v13\robotaxi gym level 1.exe'
$want = "-NoProfile -ExecutionPolicy Bypass -File `"{{INSTALL_DIR}}\scripts\install\Start-ClientAtLogon.ps1`" -N 1 -GymSource `"$gym`""
if ((New-Object -ComObject WScript.Shell).CreateShortcut($lnk).Arguments -ne $want) { exit 1 }
```

```install powershell
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'rl-roboracer Unity client.lnk'
$gym = Join-Path (Split-Path '{{INSTALL_DIR}}') 'UnityBinary\wCourseJetRacer2026.09.20-v13\robotaxi gym level 1.exe'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$s.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"{{INSTALL_DIR}}\scripts\install\Start-ClientAtLogon.ps1`" -N 1 -GymSource `"$gym`""
$s.WorkingDirectory = '{{INSTALL_DIR}}'
$s.WindowStyle = 7
$s.Description = 'Waits for Docker and ros-server, then supervises the rl-roboracer Unity client (gym switching on).'
$s.Save()
```

```verify powershell
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'rl-roboracer Unity client.lnk'
if (-not (Test-Path $lnk)) { Write-Output "missing $lnk"; exit 1 }
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
if ($s.Arguments -notlike '*Start-ClientAtLogon.ps1*') { Write-Output "$lnk does not start Start-ClientAtLogon.ps1"; exit 1 }
Write-Output "Startup shortcut: $lnk"
```

```uninstall powershell
Remove-Item -LiteralPath (Join-Path ([Environment]::GetFolderPath('Startup')) 'rl-roboracer Unity client.lnk') -Force -ErrorAction SilentlyContinue
```

## Step 15: firstjob - First TRAIN job

Why: proves the whole stack trains: queues a 2,000-iteration TRAIN job on
the starter gym (scripts\install\first-job.js), waits for the trainer to
pick it up, for two training iterations to finish, and for TensorBoard
(http://localhost:6006) to show its scalars. The job keeps running after
the install; watch it on the dashboard. Skip with `--skip firstjob`.

```detect powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
$status = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval "const j = db.jobs.findOne({installer_first_job: true}, {status: 1}); print(j ? j.status : 'NONE')" 2>$null | Select-Object -Last 1
if ("$status".Trim() -notin 'IN_PROGRESS', 'DONE') { exit 1 }
```

```install powershell phase=trainer timeout=50m
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
function Get-JobStatus {
    $s = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval "const j = db.jobs.findOne({installer_first_job: true}, {status: 1}); print(j ? j.status : 'NONE')" 2>$null | Select-Object -Last 1
    "$s".Trim()
}
$trainLog = '{{INSTALL_DIR}}\rl_agent\robotaxi.out'
function Stop-Trainer([string]$Message) {
    if (Test-Path $trainLog) { Get-Content $trainLog -Tail 200 }
    Write-Output "$Message (log: $trainLog)"
    exit 1
}
docker compose cp '{{INSTALL_DIR}}\scripts\install\first-job.js' mongo:/tmp/rl-first-job.js
if ($LASTEXITCODE -ne 0) { exit 1 }
$out = docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --file /tmp/rl-first-job.js 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $out -notmatch '(?m)^FIRST_JOB ') { Write-Output $out; exit 1 }
Write-Output ($out -split "`n" | Where-Object { $_ -match '^(designs|FIRST_JOB) ' })
$deadline = (Get-Date).AddMinutes(10)
while ((Get-JobStatus) -notin 'IN_PROGRESS', 'DONE', 'FAILED') {
    if ((Get-Date) -gt $deadline) { Stop-Trainer "the trainer did not start the job within 10 minutes (status $(Get-JobStatus))" }
    Start-Sleep -Seconds 10
}
Write-Output 'Training started; waiting for the first two iterations ...'
$deadline = (Get-Date).AddMinutes(30)
do {
    if ((Get-JobStatus) -eq 'FAILED') { Stop-Trainer 'the first training job failed' }
    $n = if (Test-Path $trainLog) { @(Select-String -Path $trainLog -Pattern 'TRAIN end:\s+iter=').Count } else { 0 }
    if ($n -ge 2) { break }
    if ((Get-Date) -gt $deadline) { Stop-Trainer 'no training steps after 30 minutes' }
    Start-Sleep -Seconds 20
} while ($true)
```

```verify powershell phase=trainer
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$pw = (Get-Content '{{INSTALL_DIR}}\.env' | Where-Object { $_ -match '^\s*MONGO_ROOT_PASSWORD\s*=' } | Select-Object -Last 1)
$pw = if ($pw) { ($pw -split '=', 2)[1].Trim() } else { 'example' }
$status = "$(docker compose exec -T mongo mongosh --quiet -u root -p $pw --authenticationDatabase admin robotaxi --eval "const j = db.jobs.findOne({installer_first_job: true}, {status: 1}); print(j ? j.status : 'NONE')" 2>$null | Select-Object -Last 1)".Trim()
if ($status -notin 'IN_PROGRESS', 'DONE') { Write-Output "first job status: $status"; exit 1 }
$deadline = (Get-Date).AddMinutes(4)
do {
    $tags = 0
    try {
        $runs = Invoke-RestMethod -Uri 'http://127.0.0.1:6006/data/plugin/scalars/tags' -TimeoutSec 10
        foreach ($run in $runs.PSObject.Properties) { $tags += @($run.Value.PSObject.Properties).Count }
    } catch { }
    if ($tags -gt 0) { Write-Output "first job $status; $tags TensorBoard scalar tags"; exit 0 }
    Start-Sleep -Seconds 15
} while ((Get-Date) -lt $deadline)
$log = '{{INSTALL_DIR}}\rl_agent\robotaxi.out'
if (Test-Path $log) { Get-Content $log -Tail 120 }
Write-Output 'training is running but TensorBoard shows no scalars (http://127.0.0.1:6006)'
exit 1
```
