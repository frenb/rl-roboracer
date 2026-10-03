# One-script Windows install, and how to test it honestly

The goal: on a Windows PC with an NVIDIA GPU, one PowerShell script takes
the machine from nothing to a first TRAIN job producing TensorBoard steps.
The user's only manual work is approving one admin prompt, one reboot, and
(if needed) installing the NVIDIA driver.

The harder half of this plan is the testing. An installer that has only ever
run on the machine it was written on proves nothing: Git, WSL, Docker
Desktop, the images, the build cache and the Unity build are all already
there, so every "install" step quietly succeeds by doing nothing. Part 3 is
about making that impossible.

Sibling docs: the manual setup this replaces is **Setup** in
[`README.md`](../README.md).

---

## Part 1 — Preparation (before writing the script)

Each of these is a problem a new user would hit today, installer or not.

### Step 1 — Rebuild the Unity client

The build in `unity\Builds\latest\` is from 2026-05-09. Its
`Assembly-CSharp.dll` still uses `niryo_moveit/...` message names, so it
cannot talk to the ROS server since the `roboracer` rename. Testing in the
Editor hid this because the Editor compiles current source.

**Do this.** Build `w-course-jetracer` into a dated folder, run
`scripts\PromoteLatestBuild.ps1`, then check the build (not the Editor)
connects with `.\scripts\Start-Stack.ps1 -N 1`.

### Step 2 — Prove `ros-server` builds from scratch

This machine builds it from a local `docker_ros-server:working` image a new
user will not have. They build `docker/ros_server/Dockerfile`, untested since
the cleanup. It still installs `ros-noetic-moveit`, an arm-era leftover worth
dropping (smaller image, fewer downloads to fail).

**Do this.** `docker build --no-cache` that Dockerfile, plus `fly_brain` and
`madscientist`. Run the same file-integrity and import checks used on
`sim_controller` on 2026-10-03.

### Step 3 — Pin versions

Unpinned packages make every build a different build. 2026-10-03's
profiler-plugin breakage came from exactly this.

**Do this.** Pin the pip installs in `docker/sim_controller/Dockerfile`
(reference list: `..\sim_controller-2026-10-03-pip-freeze.txt`), the
`mongo-express` tag, and base images.

### Step 4 — Prove a first run on an empty database

**Do this.** With an empty `mongodb` folder, confirm the trainer seeds its
default reward and experiment designs, picks up a hand-inserted TRAIN job,
trains with one Unity client, saves a checkpoint and shows in TensorBoard.
Record how long N iterations take; that sets the first job's size (target:
under 30 minutes).

### Step 5 — Decide what the first start runs

`fly-brain` needs a separate connectome download and `madscientist` idles;
neither is needed for a first model.

**Do this.** Put both under a Compose profile (e.g. `profiles: [extras]`) so
a plain `docker compose up -d` skips them.

### Step 6 — Publish the Unity client

**Do this.** Zip the Step 1 build (without `Player.log`), publish it as a
GitHub Release (`v0.1`) with a `.sha256` file next to it. The installer
verifies that checksum after downloading.

### Step 7 — Settle access and requirements

- Is the repo public? A private repo means the installer must handle
  credentials.
- Minimums the script will check: Windows 11 or Windows 10 22H2, NVIDIA GPU
  with driver R495+, 16 GB RAM (32 recommended), 150 GB free disk,
  virtualization enabled in firmware.

---

## Part 2 — The script

### Layout

```text
scripts\install\
  Install.ps1          entry point: elevation, phase runner, resume, report
  Preflight.psm1       hardware/OS/driver/network checks
  Prereqs.psm1         WSL, Git, Docker Desktop
  Workspace.psm1       clone, sibling folders, .env
  Stack.psm1           image builds, Unity client, start, first job
  Uninstall-TestInstance.ps1   resets a Part 3 Tier A test instance
```

### Phases

Each phase first checks whether its work is already done, so the script can
be re-run and can resume after the reboot. State lives in
`%ProgramData%\rl-roboracer\install-state.json`; a full log goes to
`install-<timestamp>.log` next to it.

| # | Phase | Done when |
|---|---|---|
| 0 | Preflight | All checks pass, or the script stops with one clear message per failure |
| 1 | WSL | `wsl --status` reports WSL 2 working |
| 2 | Git | `git --version` runs |
| 3 | Docker Desktop | `docker info` answers and `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi` sees the GPU |
| 4 | Workspace | Repo cloned, `saved_models`/`mongodb`/`tfrecords` exist, `.env` written |
| 5 | Images | All images built and pass import checks |
| 6 | Unity client | Release zip downloaded, checksum matches, extracted to `unity\Builds\latest\` |
| 7 | Start | Stack up, one Unity client connected to `ros-server` |
| 8 | First job | TRAIN job IN_PROGRESS and TensorBoard step count rising |

Details that matter:

- **Elevation.** The script relaunches itself as admin once at the start,
  so the user sees one UAC prompt.
- **Phase 1.** `wsl --install --no-distribution`. If Windows says a reboot
  is needed, register a `RunOnce` entry that relaunches the script, tell the
  user, and reboot on their confirmation.
- **Phase 2.** `winget install --id Git.Git --exact --silent
  --accept-package-agreements --accept-source-agreements`, then reload
  `PATH` from the registry so the new `git` is visible.
- **Phase 3.** Show the Docker Desktop licence terms and ask for a yes
  (free for personal, education and small business; paid for larger
  organisations), then `winget install --id Docker.DockerDesktop --exact
  --override "install --quiet --accept-license --backend=wsl-2"`. Start
  Docker Desktop and poll `docker info` for up to 5 minutes. If the account
  was just added to `docker-users`, stop and ask the user to sign out and
  back in, then resume.
- **Phase 4.** Default location `%USERPROFILE%\rl-roboracer\rl-roboracer`,
  overridable with `-InstallDir`. Never overwrite an existing `.env`;
  generate a random 32-character Mongo password for a new one.
- **Phase 5.** `docker compose build`, retried up to 6 times. Retries are
  needed: Docker Desktop's network intermittently truncates downloads (2 of 5
  attempts on 2026-10-03), and cached steps make each retry resume rather
  than restart. Then run the import checks inside each image, because a
  build can "succeed" with corrupted layers (also seen on 2026-10-03).
- **Phase 8.** Insert the TRAIN job directly into Mongo with the six fields
  a job needs (`job_type`, `model_type`, `robot_type`, `num_iterations`,
  `status: NOT_STARTED`, `create_date`). Wait for IN_PROGRESS, then for
  TensorBoard steps to increase. Print the dashboard and TensorBoard links.

### The install report — what keeps results honest

Every phase records **what it found and what it did**, not just pass/fail:

```json
{ "phase": "docker", "outcome": "installed",
  "evidence": "winget exit 0; docker info ok after 74 s; nvidia-smi saw RTX 4090" }
```

`outcome` is one of `installed`, `already-present`, `skipped`, `failed`.
The report is written to `install-report.json` and printed as a summary.

Two switches build on it:

- **`-ExpectClean`** fails the run immediately if any prerequisite phase
  finds its software already present. On a clean test machine this turns a
  silent false positive into a red failure.
- **`-DryRun`** runs only the detection half of every phase and prints the
  report. Useful for checking what the script *thinks* is on a machine.

Detection must test that software **works**, not that it **exists**: `docker
info` rather than "Docker Desktop.exe is on disk", the GPU container test
rather than "an NVIDIA card is listed". A half-uninstalled Docker otherwise
reads as present.

Other switches: `-InstallDir`, `-SkipPrereqs`, `-NoFirstJob`,
`-FirstJobIterations`, `-UnityReleaseTag`.

---

## Part 3 — Testing without false positives

Windows 11 Home has no Hyper-V and no Windows Sandbox, and VMware or
VirtualBox cannot run WSL 2 inside a guest while the host's own WSL/Docker
holds the hypervisor. So there is no cheap disposable VM on this laptop.
Testing comes in three tiers, from fast and partial to slow and complete.

### Tier A — Isolated instance on this Windows install *(every change)*

Tests Phases 4–8 for real. Phases 0–3 can only report `already-present`
here, which the report shows plainly.

The false positives to remove, and how:

| Already on this machine | Neutralised by |
|---|---|
| The working repo and data | Install into a separate folder, e.g. `C:\rl-test\`, with its own sibling data folders |
| Images and build cache | Tag test images with a separate prefix, build with `--no-cache --pull`, prune build cache before the run |
| A running stack | Stop the main stack first. Fixed host ports (80, 6006, 10000, 27017) and the fixed `container_name` mean two stacks cannot run side by side |
| `unity\Builds\latest\` | The test clone starts with none; the client must come from the release download |
| Mongo data | The test instance's empty `mongodb` folder |

`Uninstall-TestInstance.ps1` removes the test folder, its containers,
volumes, images and build cache, restoring the starting point. Run it before
every Tier A run.

The image-prefix line needs one change in Part 1: make the compose `image:`
names overridable, e.g. `${IMAGE_PREFIX:-}sim_controller:latest`.

### Tier B — A second, clean Windows on this laptop *(before each release)*

The only setup that tests every phase, GPU included, from nothing.

1. Back up the laptop and save the BitLocker recovery key (Settings →
   Privacy & security → Device encryption). Changing partitions can trigger
   a recovery prompt.
2. Shrink `C:` by about 300 GB in Disk Management and create an empty
   partition.
3. Install Windows 11 Home onto it from a Media Creation Tool USB. It
   activates with the laptop's existing digital licence. The laptop now
   dual-boots.
4. In the new Windows: finish setup with a local account, run Windows
   Update until clean (it normally installs an NVIDIA driver, like a real
   user's machine), and install nothing else.
5. **Capture a baseline image.** Boot the USB, press Shift+F10 for a command
   prompt, and capture the test partition to a file on the main partition:
   `dism /Capture-Image /ImageFile:<main>:\rl-test\clean.wim
   /CaptureDir:<test>:\ /Name:clean`. Drive letters differ in this
   environment; check with `diskpart` → `list volume`.

A test run: boot the test Windows, download and run the installer exactly
as a user would, with `-ExpectClean`, and collect `install-report.json`.
Then restore the baseline from the USB: format the test partition and run
`dism /Apply-Image /ImageFile:<main>:\rl-test\clean.wim /Index:1
/ApplyDir:<test>:\`. About 15 minutes, against an hour to reinstall Windows.

This tier also covers what Tier A cannot: the reboot and resume, the
`docker-users` sign-out, licence acceptance, `PATH` refresh, and a user
profile path containing a space (create the local account as e.g.
`Test User`).

### Tier C — Hyper-V VM *(optional, needs Windows 11 Pro)*

Upgrading this laptop to Pro (about USD 99) adds Hyper-V and Windows
Sandbox. A Hyper-V VM with nested virtualization runs WSL 2 and Docker
Desktop, and checkpoints reset it in seconds. GPU use from WSL inside that
VM is not realistic, so this tier needs a `-NoGpu` mode (a compose override
dropping the NVIDIA reservation; TensorFlow falls back to CPU) and tests
everything except the GPU. Worth it only if Tier B resets become a
bottleneck.

### Test scenarios

| Scenario | Tier | Expected |
|---|---|---|
| Clean machine, full run | B | Phases 1–3 `installed`; first job trains |
| Re-run after success | A, B | Every phase `already-present`; nothing changes; under 2 minutes |
| Interrupted during image build (close the window) | A | Re-run resumes at Phase 5 |
| Reboot and resume after WSL | B | Continues at Phase 2 without user input beyond sign-in |
| WSL present, Docker absent | B (partial restore, then `wsl --install` by hand) | WSL `already-present`, Docker `installed` |
| Docker installed but not running | A | Script starts it and waits |
| No NVIDIA GPU or old driver | B (driver removed) | Stops in Preflight with a link to NVIDIA's download page |
| Profile path with a space | B | All phases pass |
| Docker download truncation | A (happens naturally) | Retry recovers; report shows the attempt count |

---

## Order of work

1. Part 1, Steps 1–7. Steps 1 and 4 also fix today's setup.
2. Prepare Tier B (partition, second Windows, baseline image) while the
   script is being written. It is mostly waiting on installers.
3. Write Phases 4–8 and the report first; test in Tier A.
4. Write Phases 0–3; test in Tier B.
5. Run the scenario table in Tier B; fix; repeat until two consecutive clean
   runs pass.
6. Replace README **Setup** with: download and run `Install.ps1`, with the
   manual steps kept as a fallback section.
