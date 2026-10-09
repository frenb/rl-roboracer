# Clean install test in the `rltest` account

A clean test runs the current working tree's InstallZero installer and doctor in a second
Windows account (`rltest`). That account has its own profile and its own Docker
Desktop data, so nothing from your main install (images, build cache,
database, `.env`) can leak into the result.

Three scripts do the work:

| Script | Run as | What it does |
| --- | --- | --- |
| `scripts/install/Prepare-RlTest.ps1` | you | Stages the current code, installer, doctor and model files in `C:\Users\Public` |
| `Reset-TestAccount.ps1` (staged copy) | `rltest` | Removes everything the installer put in `rltest` |
| `Install.ps1` (staged copy) | `rltest` | The install under test; `-Doctor` picks the model that diagnoses failures |

Only one account may run Docker Desktop at a time. Its engine pipe is shared
by all accounts, so a second running Docker Desktop makes `docker` commands
reach the wrong account's engine.

## 0. Create the test account (once)

Settings > Accounts > Other users > Add account > "I don't have this person's
sign-in information" > "Add a user without a Microsoft account". Name it
`rltest`, give it a password, then under Other users > `rltest` > Change
account type > Administrator.

## 1. Stage the current version (your account)

```powershell
cd C:\Users\benja\Documents\agents\robots\LATEST\rl-roboracer
powershell -ExecutionPolicy Bypass -File scripts\install\Prepare-RlTest.ps1
```

This writes `C:\Users\Public\rl-roboracer-test\rl-roboracer.bundle`, a git
bundle whose branch `rltest` is the working tree as it is now, including
uncommitted and untracked files but not `.gitignore`d ones such as `.env`. It
does not commit to or change your branch. It also copies the installer, the
reset script, a copy of the doctor and this file (as `RUN-TEST.md`), and stages
llama.cpp plus the local models in `C:\Users\Public\doctor-assets`.

Options:

- `-LocalModels gemma,gpt-oss` (default, about 16 GB). Use `all`
  (adds `qwen` and `granite`) or `none` if you will only test hosted models.
- The gym zips and `roboracer-demos-donut.zip` must already be in
  `C:\Users\Public\rl-roboracer-test`. The script warns if one is missing.

Run it again after every code change you want to test.

Then hand the machine over:

1. Quit Docker Desktop (tray whale icon > Quit Docker Desktop).
2. Close any Unity client windows.
3. Sign out (Start > your name > Sign out). Switching user is not enough.

## 2. Fully uninstall the previous test (signed in as `rltest`)

Open PowerShell (not as administrator):

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\Reset-TestAccount.ps1 -ResetDocker
```

Add `-WhatIf` first to see what it would remove. In `rltest` it removes:

- the Unity clients and their supervisor windows
- the compose project and its volumes, then every container, image, volume
  and build-cache entry in `rltest`'s Docker engine
- with `-ResetDocker`, `rltest`'s Docker Desktop data and settings (its
  `docker-desktop` WSL distro, `%APPDATA%\Docker`, `%LOCALAPPDATA%\Docker`,
  `~\.docker`). Docker Desktop then starts as on first use: licence, sign-in
  and survey screens again. Without `-ResetDocker` those settings are kept.
- the Startup shortcut and any pending RunOnce resume entry
- `%USERPROFILE%\rl-roboracer`: code, `.env`, database, models, gyms
- `%LOCALAPPDATA%\rl-roboracer-install`: installer state, logs, report and
  the saved doctor choice
- `%LOCALAPPDATA%\rl-roboracer`: doctor transcripts

First it copies the reports, logs and doctor results to
`C:\Users\Public\rl-roboracer-test\results\<timestamp>`, so you can read them
from your account. It refuses to run in any account other than `rltest`, and
refuses to touch Docker while another account's Docker Desktop is running.

It does not remove what the whole machine shares with your account: WSL,
Git, the Docker Desktop program, and `rltest`'s membership of `docker-users`.
So the installer's `wsl`, `git` and `docker` phases report `already-present`.
To also re-test the "sign out and back in" step, remove the group membership
from an administrator PowerShell:

```powershell
net localgroup docker-users rltest /delete
```

To test the WSL, Git and Docker Desktop installs themselves, use a fresh VM
or Windows Sandbox; uninstalling them here would break your own setup.

**Heavier alternative:** delete the account (Settings > Accounts > Other users
> `rltest` > Remove), delete `C:\Users\rltest` if it is left behind, and
create it again (step 0). That also removes its Docker data (40-60 GB).

## 3. Run the install and pick the doctor's model (signed in as `rltest`)

Wait a minute after the first sign-in for Windows to finish setting up the
account, then in PowerShell (not as administrator):

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\Public\rl-roboracer-test\Install.ps1
```

No other options are needed. The installer finds the staged code bundle,
gyms and demos next to it, and the local models in
`C:\Users\Public\doctor-assets`. Its first lines show what it found (`Code:`
and `Gyms and demos:`).

Without `-Doctor`, the installer asks before it starts:

```text
If the install fails, the InstallZero doctor can investigate and report the cause and the fix.
   1  none       none (catalog diagnoses only)
   2  local      local model picked for this PC
   3  gpt-oss    gpt-oss-20b (local; ~14 GB GPU memory or 24 GB RAM)
   4  gemma      Gemma 4 E4B (local; small, runs anywhere)
   5  qwen       Qwen3.5 9B (local)
   6  granite    Granite 4.1 8B (local)
   7  anthropic  Claude Opus 5.5 (hosted, Anthropic)
   8  openai     GPT 5.6 sol (hosted, OpenAI)
   9  xai        Grok 4.7 (hosted, xAI)
  10  google     Gemini 3.1 Pro (hosted, Google)
Doctor model [1]:
```

To choose on the command line instead, add `-Doctor <choice>`:

| `-Doctor` | Model |
| --- | --- |
| `none` | no doctor; the failure catalog only |
| `local` | the bundled model that fits this PC's GPU and RAM |
| `gpt-oss`, `gemma`, `qwen`, `granite` | that bundled local model (it must be staged, see `-LocalModels`) |
| `anthropic`, `openai`, `xai`, `google` | that provider's default model, listed in the menu above |
| `xai:grok-4.7`, `openai:gpt-5.6-sol`, ... | a specific hosted model id |
| `something.gguf` | a specific model file in `-DoctorAssets` |

Examples:

```powershell
... -Doctor xai
... -Doctor anthropic:claude-opus-5-5
... -Doctor gemma
```

**Hosted models need an API key**, and `rltest` cannot see your `.env`. Either
set it in the PowerShell window before running the installer, for example
`$env:XAI_API_KEY = '...'`, or let the installer ask for it (typing is
hidden). The variable names are `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`XAI_API_KEY` and `GOOGLE_GEMINI_API_KEY`. The installer checks the key against the provider's model list
and keeps it in memory for this run only; it is never written to disk.
Choosing a hosted model sends the doctor's logs, file contents and command
output from `rltest` to that provider, with secret values masked.

**The choice is remembered** in
`%LOCALAPPDATA%\rl-roboracer-install\doctor-choice.txt`, so a resumed run
(after a restart or sign-out) does not ask again. A hosted key is asked for
again on resume unless it is set in the environment. To change the choice, pass
`-Doctor` again, or run the reset (step 2).

**When the doctor runs:** only when the install stops with a failure (exit
code 1). Its root cause, fix and confidence are printed under the failure,
added to `install-report.json` as a `diagnoses` entry with id `doctor`, and
saved as `doctor-result-<time>.json` next to the report. Its full transcript is
in `%LOCALAPPDATA%\rl-roboracer\doctor`. The doctor only reads; it changes
nothing. A pause to restart Windows (exit code 2) or to sign out (exit code 3)
is not a failure. Do what it says and run the same command again.

Expected along the way on a clean account:

- `wsl`, `git` and `docker` report `already-present`; everything after that
  reports `installed`.
- Docker Desktop opens for this account the first time: accept the licence,
  skip sign-in and the survey.
- The image build takes 30-60 minutes; this account has no build cache.
- It ends once the first TRAIN job is producing training steps and
  TensorBoard (http://localhost:6006) shows scalars.

## 4. Exercise the doctor when the install passes

A clean install should not fail, so the doctor never runs. To see a model
diagnose real failures on the installed stack, use the fault scenarios. Each
one breaks something on purpose, runs the doctor, grades its answer and
restores:

```powershell
cd $env:USERPROFILE\rl-roboracer\rl-roboracer\scripts\install\doctor\eval
$env:RL_DOCTOR_ASSETS = 'C:\Users\Public\doctor-assets'
.\Invoke-DoctorEval.ps1 -Action List
.\Invoke-DoctorEval.ps1 -Action Run -Models xai:grok-4.7
.\Invoke-DoctorEval.ps1 -Action Run -Models gemma-4-E4B-it-Q4_K_M.gguf -Scenario <name>
```

To ask the doctor about something directly:

```powershell
cd $env:USERPROFILE\rl-roboracer\rl-roboracer\scripts\install\doctor
.\Doctor.ps1 -Problem 'The dashboard at http://localhost shows no gyms' -Remote xai:grok-4.7
.\Doctor.ps1 -Problem '...' -Model gemma-4-E4B-it-Q4_K_M.gguf
```

## 5. Back to your account

In `rltest`: quit Docker Desktop, close the Unity window, sign out. Run step 2
first if you want the results copied to `C:\Users\Public\rl-roboracer-test\results`.

In your account: start Docker Desktop, then `docker compose -p rl-roboracer start`
in the repo to bring your own stack back.
