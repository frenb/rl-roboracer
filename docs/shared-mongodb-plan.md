# Sharing the project database with a second developer

Goal: one MongoDB that both you and your collaborator use — they see results
now, and run their own training jobs soon after.

Nothing here has been done yet. Read Part 0 first. It contains one finding
that means **you must not point two trainers at one database until Part 5 is
done**, and one measurement that keeps the whole thing inside GCP's free
tier.

Sibling docs:

- The local stack this database serves: [`../README.md`](../README.md).
- Why the Mongo image is pinned to `bitnamilegacy`: "Migrating off the legacy
  Bitnami Mongo image" in [`../README.md`](../README.md).

---

## Part 0 — What the survey found

Measured on the running system, 2026-09-20.

### What you're actually sharing is 3.7 MiB

| Collection | Docs | Data |
|---|---|---|
| `logs` | 339,153,048 | 219,654.7 MiB |
| `models` | 7,680 | 2.2 MiB |
| `research_notes` | 189 | 0.5 MiB |
| `jobs` | 1,164 | 0.4 MiB |
| `leaderboard_scores` | 911 | 0.3 MiB |
| `proposals` | 30 | 0.2 MiB |
| everything else (6 collections) | 81 | 0.1 MiB |

`logs` is 99.998% of the database and is not part of collaboration: the
trainer writes it, and the only reader is the dashboard's per-job trajectory
endpoint, which takes at most 5,000 documents per job. Leaving it behind
makes everything else free and instant. Part 6 covers adding a bounded slice
later if the trajectory view turns out to matter.

Everything your collaborator needs is about 10,000 documents.

### The local mongod hosts other projects too

Alongside `robotaxi` it carries `eval` (15.2 MiB), `golden_questions_db`,
`groovy`, `partygirl`, `sim`, `travel-wiki` and `control_center` — about
18 MiB of unrelated work. None of it should reach the shared VM, and none of
it does: Step 2 dumps `--db robotaxi`, and Step 10 grants your collaborator
`readWrite` on `robotaxi` alone. Worth knowing before anyone reaches for a
whole-instance `mongodump` or an `--eval` loop over `listDatabases`.

### Change streams are load-bearing

`dashboard/src/server.ts` calls `superviseChangeStream(...)` on `jobs`,
`models`, `leaderboard_scores`, `env_specs`, `reward_designs`,
`experiment_designs` and `gyms` to drive live dashboard updates. Change
streams need a replica set, so the shared instance must run as one even with
a single member — same as your local setup, which is replica set
`replicaset`, one member advertised as `mongo:27017`, MongoDB 6.0.13.

This is the main reason the plan reuses your existing container image rather
than `apt install mongodb`: the replica-set configuration is already correct
in `docker-compose.yml`.

### Your tooling and account

`gcloud` and `kubectl` are installed. Project `conversational-ai-403204`,
region `us-central1`, zone `us-central1-c` — which is one of the three
regions GCP's always-free e2-micro is available in. **Your gcloud
credentials are expired** (`invalid_grant`); Step 4 starts with re-auth.
`mongodump` and `mongosh` are not on the host, but both ship inside the Mongo
container, which is how Part 1 avoids installing anything.

---

## Part 0b — Two trainers cannot share this queue as it stands

This is the finding that orders the whole plan.

`get_jobs()` in `rl_agent/robotaxi.py` is a plain query with no atomic claim,
and it returns jobs that are **already running**:

```python
is_not_started = {"status": "NOT_STARTED"}
is_in_progress = {"status": "IN_PROGRESS"}
jobs = db.jobs.find({"$or":[is_not_started, is_in_progress]}).sort("create_date", 1)
```

Including `IN_PROGRESS` is deliberate and correct for one trainer: it is the
crash-recovery path, so a job that was running when the trainer was killed
gets picked up again and resumed from its Learner checkpoint.

With two trainers on one database it means the opposite of what you want.
Your collaborator's trainer polls, sees *your* running job sitting at
`IN_PROGRESS`, concludes it crashed, and starts running it — resuming from
your checkpoint, writing to the same job document, saving over the same model
records and leaderboard rows. This is not a narrow race that you might get
away with; it happens on their very first poll.

So the rollout is staged:

- **Phase 1 (Parts 1-4).** Shared database. Your collaborator reads: the
  dashboard, leaderboard, job history, model records. Your trainer stays the
  only one running. Nothing about the queue changes, and the risk is low.
- **Phase 2 (Part 5).** Give jobs an owner before a second trainer ever
  starts. Also sort out the artifacts, which the database does not contain.

---

## Part 1 — Export from the local database

Local machine only. Nothing here is destructive.

### Step 1 — Quiet the writers, or accept a known drift

**Do this.** Wait for the running `fly_donut` job to finish, or accept that
a job completing mid-dump may be captured half-updated.

**Why it matters.** The trainer inserts ~12 documents/second during a run.
For `logs` that is irrelevant since you are not exporting it, but `jobs` and
`models` are written at job boundaries.

**You are done when.** No job is running, or you have decided you don't care.

### Step 2 — Dump everything except `logs`

**Do this.** Stream an archive out of the container to a host file:

```powershell
docker compose exec -T mongo mongodump `
  --username root --password example --authenticationDatabase admin `
  --db robotaxi --excludeCollection logs `
  --archive --gzip > meta.archive.gz
```

**Why it matters.** This is the entire shared dataset. It will be a few
megabytes and take seconds.

**You are done when.** `meta.archive.gz` is nonzero and this reports no
errors:

```powershell
docker compose exec -T mongo mongorestore --archive --gzip --dryRun -v < meta.archive.gz
```

---

## Part 2 — Stand up MongoDB on a GCE VM

### Step 3 — Re-authenticate

**Do this.**

```powershell
gcloud auth login
gcloud config set project conversational-ai-403204
```

**You are done when.** `gcloud compute instances list` runs without an auth
error.

### Step 4 — Create the VM

**Do this.**

```powershell
gcloud compute instances create mongo-shared `
  --zone us-central1-c `
  --machine-type e2-micro `
  --image-family debian-12 --image-project debian-cloud `
  --boot-disk-size 30GB --boot-disk-type pd-standard `
  --no-address
```

**Why it matters.** `e2-micro` + 30 GB `pd-standard` in `us-central1` is
exactly GCP's always-free allowance, so this costs approximately nothing for
a 4 MiB database. `--no-address` gives it no public IP at all, which is what
makes Step 7's access model safe — there is no Mongo port on the internet to
find.

The tradeoff: e2-micro has 1 GB of RAM and burstable CPU. Ample here, but if
you later add the `logs` slice from Part 6, move to `e2-small` (2 GB,
roughly $13/month) first.

**You are done when.** `gcloud compute instances list` shows `mongo-shared`
`RUNNING` with no external IP.

### Step 5 — Allow IAP to reach it, and install Docker

**Do this.**

```powershell
gcloud compute firewall-rules create allow-iap-ssh `
  --allow tcp:22 --source-ranges 35.235.240.0/20

gcloud compute ssh mongo-shared --zone us-central1-c --tunnel-through-iap
```

Then on the VM:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # log out and back in
```

**Why it matters.** `35.235.240.0/20` is Google's IAP forwarding range; that
one rule lets you SSH to a VM with no public IP, authenticated by your Google
identity rather than by a key you have to distribute. Port 22 from that range
is the *only* thing open.

**You are done when.** `docker run --rm hello-world` works on the VM.

### Step 6 — Run Mongo with your existing configuration

**Do this.** On the VM, create `~/docker-compose.yml` mirroring the `mongo`
service you already run, with a real password and the port bound to
localhost only:

```yaml
services:
  mongo:
    image: bitnamilegacy/mongodb:6.0
    restart: always
    environment:
      MONGODB_ROOT_PASSWORD: ${MONGO_ROOT_PW}
      MONGODB_REPLICA_SET_MODE: primary
      MONGODB_ADVERTISED_HOSTNAME: localhost
      MONGODB_REPLICA_SET_KEY: ${MONGO_RS_KEY}
    ports:
      - "127.0.0.1:27017:27017"
    volumes:
      - mongo_data:/bitnami/mongodb
volumes:
  mongo_data:
```

```bash
printf 'MONGO_ROOT_PW=%s\nMONGO_RS_KEY=%s\n' \
  "$(openssl rand -base64 24)" "$(openssl rand -hex 16)" > ~/.env
chmod 600 ~/.env
docker compose up -d
```

**Why it matters.** Same image, same replica-set environment variables as
your desktop, so change streams behave identically and there is nothing new
to learn. Three deliberate differences: a generated root password instead of
`example`, `127.0.0.1:` on the port binding so Docker cannot publish it to
the VM's network interface, and `MONGODB_ADVERTISED_HOSTNAME: localhost`
because every client reaches this through a tunnel that terminates on
localhost.

**You are done when.** `docker compose exec mongo mongosh -u root -p "$MONGO_ROOT_PW" --eval "rs.status().myState"` prints `1` (primary).

---

## Part 3 — Import and verify

### Step 7 — Open a tunnel from your desktop

**Do this.** In a terminal you leave running:

```powershell
gcloud compute ssh mongo-shared --zone us-central1-c --tunnel-through-iap `
  -- -N -L 27018:localhost:27017
```

**Why it matters.** Local port 27018 avoids colliding with your own Mongo on
27017. Nothing is exposed: the tunnel lives only as long as this command, and
it is authorised by your Google account.

**You are done when.** The command sits there without returning.

### Step 8 — Restore

**Do this.**

```powershell
docker compose exec -T mongo mongorestore `
  --uri "mongodb://root:<password>@host.docker.internal:27018/?authSource=admin&directConnection=true" `
  --archive --gzip < meta.archive.gz
```

**Why it matters.** `directConnection=true` is not optional. Without it the
driver reads the replica-set config, tries to reconnect to whatever hostname
it advertises, and hangs until it times out. This is the most common way this
kind of migration fails.

**You are done when.** `mongorestore` reports zero failures.

### Step 9 — Verify by content, not just by count

**Do this.** Run this against both databases and compare:

```javascript
db.runCommand({dbHash: 1, collections: ['jobs','models','gyms',
  'experiment_designs','reward_designs','leaderboard_scores','env_specs']})
```

**Why it matters.** Counts catch a truncated restore; `dbHash` catches silent
corruption and type coercion — which matters here because `logs.job_id` is
stored as an `ObjectId` while the dashboard passes hex strings around, so
this codebase already has type-mismatch history.

**You are done when.** Every collection's hash matches.

---

## Part 4 — Give your collaborator access

### Step 10 — A user for them, not your root account

**Do this.** On the VM:

```javascript
db.getSiblingDB('admin').createUser({
  user: 'dev2',
  pwd: '<generated>',
  roles: [{role: 'readWrite', db: 'robotaxi'}]
})
```

**Why it matters.** `readWrite` lets them use the dashboard fully — including
creating jobs — without holding an account that can drop databases or add
users. Revoking them later is one `dropUser`.

**You are done when.** They can connect with that user and not with root.

### Step 11 — Grant them the tunnel, not the network

**Do this.**

```powershell
gcloud projects add-iam-policy-binding conversational-ai-403204 `
  --member "user:<their-email>" --role roles/iap.tunnelResourceAccessor
gcloud projects add-iam-policy-binding conversational-ai-403204 `
  --member "user:<their-email>" --role roles/compute.osLogin
```

They then run the same Step 7 command and point their stack at the tunnel.

**Why it matters.** Access is their Google identity, so there is no shared
secret, no IP allowlist to chase when a home connection changes address, and
removing them is instant and total. It also means the database is never
reachable from the open internet, which matters more than usual given this
project's credentials were `root`/`example` until today.

**You are done when.** They can open the tunnel and read the leaderboard.

### Step 12 — Point a stack at the shared database

**Do this.** Change one line in `.env`:

```
MONGO_URL=mongodb://dev2:<password>@host.docker.internal:27018/?authSource=admin&directConnection=true
```

Then `docker compose up -d`. Check first that every service reading
`MONGO_URL` can resolve `host.docker.internal` — `sim-controller` already
has the `extra_hosts: host.docker.internal:host-gateway` mapping, and the
dashboard service may need it added.

**Why it matters.** `.env` line 21 is the single source of the connection
string for both the trainer and the dashboard, so switching over and
switching back are both one line.

**You are done when.** Their dashboard lists your jobs and the tables update
on their own — that last part is what exercises the change streams.

---

## Part 5 — Before a second trainer ever runs

Do not skip to this. Phase 1 above is safe; this is the part that needs code.

### Step 13 — Give jobs an owner

**Do this.** Add a `worker_id` to the queue, set per machine by an env var:

1. Each developer sets `WORKER_ID` (`benja-desktop`, `dev2-laptop`) in `.env`.
2. Replace the `find()` in `get_jobs()` with an atomic claim, so a job is
   taken by exactly one trainer:

   ```python
   db.jobs.find_one_and_update(
       {"status": "NOT_STARTED",
        "$or": [{"worker_id": None}, {"worker_id": WORKER_ID}]},
       {"$set": {"status": "IN_PROGRESS", "worker_id": WORKER_ID,
                 "claimed_at": now}},
       sort=[("create_date", 1)])
   ```

3. Scope crash recovery to your own jobs: only reclaim an `IN_PROGRESS` job
   when `worker_id == WORKER_ID`.
4. Add a worker column to the Jobs tab, and a worker selector on the new-job
   dialog so a job can be aimed at whoever has the right gym.

**Why it matters.** Step 3 is the crux. It keeps the crash-recovery
behaviour you rely on while making it impossible for one trainer to adopt
another's running job — which, per Part 0b, is otherwise guaranteed rather
than merely possible.

**You are done when.** Two trainers poll the same database and each only ever
logs pickups for its own `worker_id`.

### Step 14 — Deal with the artifacts the database doesn't hold

**Do this.** Decide how gym binaries and saved models are shared.

The `gyms` documents store absolute Windows paths — the v13 gym is
`C:\Users\benja\Documents\agents\robots\LATEST\UnityBinary\wCourseJetRacer2026.09.20-v13\robotaxi gym level 1.exe`
— and model records point at `/saved_models/robotaxi/SacAgent/...`. Neither
resolves on another machine, and `_signal_gym_switch` sends that exact path
to the Unity supervisor for hot-swap.

The smallest fix is to store gym paths relative to a per-machine root
(`UNITY_BINARY_ROOT`) and resolve at use. Checkpoints are the bigger problem:
they are not in Mongo at all, so sharing model records without the files
gives your collaborator rows they cannot evaluate. A GCS bucket synced to
`/saved_models` is the usual answer.

**Why it matters.** A shared database gets them your metadata. It does not
get them a working system, and this is the gap people discover after the
database move rather than before.

**You are done when.** Your collaborator can run an EVAL job on a model
*you* trained.

---

## Part 6 — Optional: the trajectory view

The dashboard's per-job trajectory endpoint reads `logs`, so on the shared
database it will be empty. If it matters, copy the newest 5,000
`position_history` documents per job — which is exactly the endpoint's own
limit, so the result is indistinguishable from local — into a staging
collection, dump it, and restore it renamed to `logs`:

```javascript
db.logs_slice.drop();
db.jobs.find({}, {_id: 1}).forEach(function (j) {
  db.logs.find({job_id: j._id, position_history: {$ne: null}})
    .sort({_id: -1}).limit(5000)
    .forEach(function (d) { db.logs_slice.insertOne(d); });
});
```

```powershell
docker compose exec -T mongo mongorestore --uri "..." `
  --nsFrom 'robotaxi.logs_slice' --nsTo 'robotaxi.logs' --archive --gzip < logs_slice.archive.gz
```

Expect a few GiB. Confirm the `weakness_job_recent` index on
`{job_id: 1, _id: -1}` exists afterwards or the endpoint full-scans and hits
its 20-second `maxTimeMS` ceiling. Move to `e2-small` first.

---

## Rollback

Parts 1-4 do not modify the local database; it keeps running on
`mongodb://root:example@mongo:27017/` the whole time. To go back, revert the
`.env` line and `docker compose up -d`. To remove the cloud side entirely:

```powershell
gcloud compute instances delete mongo-shared --zone us-central1-c
gcloud compute firewall-rules delete allow-iap-ssh
```

---

## Known traps, collected

1. **Two trainers on one database will collide by design**, not by race. See
   Part 0b. Part 5 before a second trainer, always.
2. **A standalone `mongod` breaks the dashboard silently** — change streams
   need a replica set. The UI just stops updating.
3. **Omitting `directConnection=true`** makes any tunnelled client hang on a
   hostname it cannot resolve.
4. **`root`/`example` must not leave the laptop.** The VM gets a generated
   password; the collaborator gets their own `readWrite` user.
5. **Gym paths and checkpoints are machine-local** and are not fixed by
   moving the database.
6. **Your local mongod holds six unrelated projects.** Keep every dump and
   every grant scoped to `robotaxi`.
7. **`logs` has no TTL index.** If you ever migrate it, it resumes growing at
   ~800 MB per training day. A TTL index there would help the local database
   too.
