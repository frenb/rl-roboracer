# One-script macOS install, and how to test it honestly

The macOS counterpart of [`install-script-plan.md`](install-script-plan.md)
(Windows). Same goal: one script takes a Mac from nothing to a first TRAIN
job producing TensorBoard steps. Read the Windows plan first; this doc only
repeats what changes.

**Set expectations first.** A Mac can run the whole platform, but not the
way the Windows PC does:

- **No GPU for training.** Docker on macOS cannot reach the Mac's GPU, and
  Macs have no NVIDIA GPU anyway, so TensorFlow trains on the CPU. Fine for
  a first small model and for trying the platform. Not for serious runs.
- **Apple Silicon runs the containers in x86 emulation.** `sim-controller`
  and `ros-server` both install `tensorflow==2.7.0rc1` and
  `tf-agents[reverb]==0.11.0rc0`. Reverb only publishes Linux x86-64
  packages and TF 2.7 has no official ARM64 Linux build, so on M-series
  Macs those images run as `linux/amd64` through Rosetta. Intel Macs run
  them natively.
- **The Unity client needs a separate macOS build.**
- **The host scripts are Windows-only.** `Start-Stack.ps1`,
  `RunClientWrapper.ps1` and `RunNClients.ps1` use robocopy, Windows
  Terminal and Windows process handling; the Mac needs shell equivalents.

Because of the second point, this plan starts with a feasibility check. If
it fails, macOS support is Intel-only or not offered, and nothing else here
is worth doing.

---

## Part 0 — Feasibility check (needs a Mac, before anything else)

### Check 1 — Does the trainer image run under emulation?

TensorFlow's prebuilt packages use AVX CPU instructions. Rosetta did not
translate AVX until macOS 15 (Sequoia), and whether Docker's Rosetta-based
Linux emulation inherits that needs confirming, not assuming. Without it
TensorFlow dies on import with "Illegal instruction".

**Do this.** On an Apple Silicon Mac with macOS 15+ and Docker Desktop's
"Use Rosetta for x86_64/amd64 emulation" on:

```bash
docker run --rm --platform linux/amd64 sim_controller:latest \
  python3 -c "import tensorflow as tf, reverb, tf_agents; print(tf.__version__)"
```

using an image pushed from the Windows PC (see Part 1, Step 3). Repeat on
macOS 14 to learn whether it is a hard minimum.

**If it fails:** native ARM64 would mean upgrading TensorFlow and replacing
Reverb in the trainer, a large refactor outside this plan. Ship macOS as
Intel-only, or not at all.

### Check 2 — How slow is it?

**Do this.** With the stack up (GPU reservation removed, Part 1 Step 2) and
one Unity client, run a TRAIN job for 2,000 iterations and record time per
1,000 steps. Compare with the Windows PC's rate.

This decides the first job's size and the wording in the README. If a
useful first run takes many hours, say so plainly.

### Check 3 — Does a macOS Unity build work?

**Do this.** In the Unity Hub on the Windows PC, add **Mac Build Support
(Mono)** to Editor 2020.3.11f1. Mono builds for macOS can be made from
Windows. Build `w-course-jetracer` for macOS, copy it to the Mac, and check:

- it launches on Apple Silicon (Unity 2020.3 can build Intel, Apple Silicon
  or Universal players; Universal is the target);
- the WebRTC plugin (`com.unity.webrtc` 2.4.0-exp.4) loads;
- it connects to `ros-server` with the same `--ros-port` / `--unity-port`
  arguments the Windows launcher passes;
- two instances can run at once (`open -n`), the Mac equivalent of the
  per-instance copies `RunNClients.ps1` makes.

---

## Part 1 — Preparation

All seven preparation steps in the Windows plan apply here too (fresh Unity
build, `ros-server` from-scratch build, pinned versions, empty-database
test, Compose profile for `fly-brain`/`madscientist`, release, repo access).
These are the Mac additions.

### Step 1 — macOS Unity client release

**Do this.** Zip the Check 3 build with `ditto -c -k --keepParent` (keeps
the `.app` bundle's permissions) and attach it to the same GitHub Release as
the Windows zip, with its `.sha256`.

Unsigned apps: a zip downloaded by a browser gets macOS's quarantine flag
and Gatekeeper blocks it ("app is damaged"). Downloads made with `curl`, as
the installer does, are not flagged. The installer also removes the flag
(`xattr -dr com.apple.quarantine`) in case the user downloaded it by hand.
Signing and notarizing with an Apple Developer account (USD 99/year) would
remove the problem for everyone; optional.

### Step 2 — Compose override for the Mac

**Do this.** Add `compose/mac.yml`:

- drop the NVIDIA reservations from `sim-controller` (and `fly-brain`) with
  Compose's `!reset` tag;
- set `platform: linux/amd64` on `sim-controller` and `ros-server` (and the
  scale overlay's `ros-server-1..3`).

Nothing else should differ from the Windows stack.

### Step 3 — Publish prebuilt images

Building the images on an Apple Silicon Mac means compiling the ROS
workspace and installing hundreds of pip packages under emulation: slow,
and exposed to the same download truncation seen on Windows.

**Do this.** Push the tested `linux/amd64` images to GitHub Container
Registry (`ghcr.io/frenb/...`), tagged per release. The Mac installer pulls
instead of building. This also benefits the Windows installer; building
from source stays available as an option.

### Step 4 — Mac host scripts

**Do this.** Port the launchers to `scripts/mac/` as `zsh` scripts
(`start-stack.sh`, `stop-stack.sh`, `run-client.sh`). They start the stack
with `-f compose/mac.yml`, launch N clients with `open -n -a <app> --args
--ros-port <10000+i> --unity-port <5005+i>`, and restart a client that
exits, as `RunClientWrapper.ps1` does. Use `zsh` rather than `bash`: macOS
ships bash 3.2, which lacks features a script this size will want.

---

## Part 2 — The script

`scripts/install/install-macos.sh`, run with:

```bash
curl -fsSL https://raw.githubusercontent.com/frenb/rl-roboracer/main/scripts/install/install-macos.sh | zsh
```

Same structure as the Windows script: phases that check before acting,
state in `~/Library/Application Support/rl-roboracer/install-state.json`, a
log beside it, and the same `install-report.json` with `-ExpectClean` and
`-DryRun` (as `--expect-clean`, `--dry-run`).

| # | Phase | Done when |
|---|---|---|
| 0 | Preflight | macOS version, chip, RAM, disk and network checks pass |
| 1 | Rosetta | `arch -x86_64 /usr/bin/true` succeeds (Apple Silicon only) |
| 2 | Command Line Tools | `xcode-select -p` returns a path and `git --version` runs |
| 3 | Homebrew | `brew --version` runs |
| 4 | Docker Desktop | `docker info` answers and an `--platform linux/amd64` container runs |
| 5 | Workspace | Repo cloned, sibling folders exist, `.env` written |
| 6 | Images | Images pulled (or built) and pass import checks |
| 7 | Unity client | Release zip downloaded, checksum matches, unpacked, quarantine removed |
| 8 | Start | Stack up, one Unity client connected |
| 9 | First job | TRAIN job IN_PROGRESS and TensorBoard steps rising |

Differences from Windows that matter:

- **No reboot.** Nothing in this list needs one. The equivalent hurdles are
  password prompts.
- **One admin password, entered up front.** The script runs `sudo -v` at the
  start and keeps the credential alive while it runs. The account must be an
  administrator; the script checks and stops with a clear message if not.
- **Phase 0.** Minimum macOS from Check 1 (likely 15 on Apple Silicon).
  16 GB RAM minimum. 150 GB free disk.
- **Phase 1.** `softwareupdate --install-rosetta --agree-to-license`.
- **Phase 2.** Command Line Tools provide `git`. The Homebrew installer
  installs them without the usual pop-up, so Phases 2 and 3 usually happen
  together; Phase 2 still verifies.
- **Phase 3.** The official installer with `NONINTERACTIVE=1`, then add
  `brew shellenv` to `~/.zprofile`. Homebrew lives in `/opt/homebrew` on
  Apple Silicon and `/usr/local` on Intel; detect both. An Intel-era
  Homebrew in `/usr/local` on an Apple Silicon Mac is a real-world mess the
  script should detect and warn about, not "fix".
- **Phase 4.** Show the Docker licence terms and ask for a yes, then
  `brew install --cask docker-desktop` (cask name per current Homebrew; the
  older name was `docker`). Then
  `sudo /Applications/Docker.app/Contents/MacOS/install --accept-license
  --user=$USER`, which pre-approves the privileged helper so the first
  launch shows no dialogs. Set the Docker VM's memory to at least 8 GB and
  turn on Rosetta emulation in Docker's settings file, `open -a Docker`, and
  poll `docker info` for up to 5 minutes.
- **Phase 6.** `docker compose -f docker-compose.yml -f compose/mac.yml
  pull`, with retries. Run the same import checks as on Windows.
- **Phase 7.** Download with `curl`, verify with `shasum -a 256`, unpack
  with `ditto -x -k` into `unity/Builds/latest/`.
- **Phase 8.** If the macOS firewall is on, the Unity client's first launch
  asks to accept incoming connections. Tell the user before it appears.
- **Phase 9.** Same job insert as Windows, with the iteration count from
  Check 2.

Docker Desktop alternatives (OrbStack, Colima) avoid Docker's licence terms
and are often faster on Macs. Worth a later look; start with Docker Desktop
so both platforms use the same engine.

---

## Part 3 — Testing without false positives

Same principle as Windows: every prerequisite phase must report
`installed` on a clean machine, enforced with `--expect-clean`.

The Mac-specific constraint is the reverse of Windows Home's. Making a
clean macOS virtual machine is easy and fast on Apple Silicon, but **Docker
Desktop cannot run inside a macOS VM**: the guest cannot start VMs of its
own. So VMs test the installer's logic but not the stack.

### Tier A — Isolated instance on your Mac *(every change)*

As on Windows: separate install folder and data folders, test images under
a separate prefix, build cache pruned, main stack stopped, no
`unity/Builds/latest/`. Tests Phases 5–9 for real; Phases 0–4 report
`already-present`.

### Tier B — macOS VM *(every change to Phases 0–4)*

Free tools (Tart, UTM) create a macOS VM from Apple's restore image and
clone it in seconds, so every run starts clean.

Covers: preflight, Rosetta, Command Line Tools, Homebrew, the admin
password flow, Docker Desktop's install and licence step, the Gatekeeper
handling of the Unity zip. Expected and accepted: Phase 4 stops at
`docker info`. The report marks this `vm-limited` rather than `failed`, and
the test checks for exactly that.

Run it as a standard (non-admin) user too, to see the admin check stop
cleanly.

### Tier C — A real clean Mac *(before each release)*

The only full test. Options, cheapest first:

1. **A second macOS on your Mac.** In Disk Utility, add an APFS volume to
   the internal disk; install macOS onto it from Recovery; switch with
   System Settings → Startup Disk. It shares free space with your main
   macOS, needs no partitioning, and leaves your main volume untouched. To
   reset, erase that volume and reinstall macOS (about 45 minutes).
2. **A dedicated test Mac.** "Erase All Content and Settings" (Apple
   Silicon, or Intel with a T2 chip, macOS 12+) returns it to factory state
   in minutes. A used base M1 Mac mini is the cheapest way to get this.
3. **AWS EC2 Mac.** These are real Mac minis rented by the hour, so
   Docker's virtualization works (confirm in Part 0). Launching a fresh
   instance from Apple's macOS image gives a clean machine each time. Apple's
   licence forces a 24-hour minimum allocation, so a test day costs roughly
   USD 15–25 (check current pricing).

### Test scenarios

| Scenario | Tier | Expected |
|---|---|---|
| Clean Apple Silicon Mac, current macOS | C | Phases 1–4 `installed`; first job trains |
| Clean Intel Mac | C (or skip if Intel is dropped) | Phase 1 `skipped`; images run natively |
| macOS below the Check 1 minimum | B | Stops in Preflight, says which version is needed |
| Re-run after success | A, C | Every phase `already-present`; nothing changes |
| Docker Desktop installed from the .dmg, not Homebrew | B, C | Detected as present; not reinstalled |
| Intel Homebrew in `/usr/local` on Apple Silicon | B | Warning; installs native Homebrew alongside |
| Standard (non-admin) user | B | Stops with a clear message before changing anything |
| Unity zip downloaded by browser | B | Quarantine removed; app opens |
| Docker VM memory below 8 GB | A | Script raises it, or explains how |
| Interrupted during image pull | A | Re-run resumes at Phase 6 |

---

## Order of work

1. Get access to an Apple Silicon Mac on macOS 15+ (yours, a borrowed one,
   or an EC2 Mac for a day). Nothing below can be checked from Windows.
2. Part 0. Stop here if Check 1 fails, and decide about Intel-only.
3. Windows plan Part 1, then Mac Part 1, Steps 1–4. Step 3 (prebuilt
   images) is worth doing first; it shortens both installers.
4. Write Phases 5–9 and test in Tier A.
5. Write Phases 0–4 and test in Tier B.
6. Run the scenario table in Tier C until two consecutive clean runs pass.
7. Add a macOS section to README **Setup**.
