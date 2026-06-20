import * as express from 'express';
import * as bodyParser from 'body-parser';
import * as path from 'path';
import * as fs from 'fs';
import * as mongoDB from 'mongodb';
const ObjectID = require('mongodb').ObjectID;
const cors = require('cors');
const { spawn } = require('child_process');
const crypto = require('crypto');
import { log, LogLevel } from './log';
import * as morgan from 'morgan';

const WebSocket = require('ws');

// ----------------------------------------------------------------
// WebSocket server for real-time data updates (MongoDB change streams)
// Clients connect to port 8081, subscribe to collections, and receive
// incremental updates instead of polling.
// ----------------------------------------------------------------
const dataWss = new WebSocket.Server({ port: 8082 });

// Track connected clients and their subscriptions
// Map<WebSocket, Set<collectionName>>
const dataClients: Map<any, Set<string>> = new Map();

// Broadcast a change event to all clients subscribed to a collection
function broadcastChange(collection: string, change: any) {
  const message = JSON.stringify({
    type: 'change',
    collection,
    operationType: change.operationType,
    documentKey: change.documentKey,
    // For updates, include the changed fields
    updateDescription: change.updateDescription,
    // For inserts/replaces, include the full document
    fullDocument: change.fullDocument,
  });

  dataClients.forEach((subscriptions, client) => {
    if (subscriptions.has(collection) && client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

dataWss.on('connection', (ws) => {
  console.log('[DataWS] Client connected');
  dataClients.set(ws, new Set());

  ws.on('message', (data: string) => {
    try {
      const msg = JSON.parse(data);
      if (msg.type === 'subscribe' && Array.isArray(msg.collections)) {
        const subs = dataClients.get(ws);
        msg.collections.forEach((c: string) => subs?.add(c));
        console.log('[DataWS] Client subscribed to:', msg.collections);
        ws.send(JSON.stringify({ type: 'subscribed', collections: msg.collections }));
      }
    } catch (e) {
      console.error('[DataWS] Invalid message:', e);
    }
  });

  ws.on('close', () => {
    console.log('[DataWS] Client disconnected');
    dataClients.delete(ws);
  });

  ws.on('error', (err) => {
    console.error('[DataWS] Error:', err);
    dataClients.delete(ws);
  });
});

console.log('[DataWS] WebSocket server for data updates listening on port 8082');

export const createServer = (config): express.Application => {
  const app: express.Application = express();
  const databaseName = process.env.DATABASE_NAME || "robotaxi";
  var jobsChanged = true;
  var modelsChanged = true;
  var leaderboardScoresChanged = true;
  // The env_specs collection holds one document per robot_type
  // describing the live env's observation / action spec. Populated by
  // robotaxi.py's publish_env_spec() at every TRAIN / EVAL job start.
  // Used by the Models tab's "Compat" column to flag rows whose
  // training-time spec no longer matches the running env.
  var envSpecsChanged = true;
  // The reward_designs collection holds user-authored reward functions
  // (see rl_agent/reward_designs.py). One document per design with a
  // human-friendly name + Python source code + version. Selected
  // per-job via the New-Job form's "Reward design" dropdown.
  var rewardDesignsChanged = true;
  // Sibling of rewardDesignsChanged: the experiment_designs collection
  // holds user-authored training-loop configs (SAC hyperparameters,
  // BC pretrain steps, replay capacity, network sizes). Selected per-
  // job via the New-Job form's "Experiment design" dropdown alongside
  // the existing Reward design dropdown.
  var experimentDesignsChanged = true;
  // The gyms collection records registered Unity scene / gym
  // configurations (name + full file path). Selected per-job via the
  // New-Job and Eval dialogs' "Gym" dropdowns.
  var gymsChanged = true;
  var MongoClient = mongoDB.MongoClient;
  var url = process.env.MONGODB_URL || "mongodb://root:example@mongo:27017";
  var dbo;
  MongoClient.connect(url, function(err, db) {
    if (err) throw err;
    
    dbo = db.db(databaseName);
    // create a collection object for the collection based on collection_name
    // collectionNames.forEach(function(collectionName) {
    //   const collection = dbo.collection(collectionName);
    //   const changeStream = collection.watch();
  
    //   // listen for changes in the collection
    //   changeStream.on('change', function(change) {
    //     console.log(`Change detected in ${collectionName}:`, change);
    //     // handle the change event here
    //   });
    // });
    const jobs = dbo.collection("jobs");
    const models = dbo.collection("models");
    const leaderboardScores = dbo.collection("leaderboard_scores");
    const envSpecs = dbo.collection("env_specs");
    
    // Watch with fullDocument option so inserts/replaces include the full doc
    const jobsChangeStream = jobs.watch([], { fullDocument: 'updateLookup' });
    jobsChangeStream.on('change', (change) => {
      console.log('Change detected (jobs):', change.operationType, change.documentKey?._id);
      jobsChanged=true;
      broadcastChange('jobs', change);
    });

    const modelsChangeStream = models.watch([], { fullDocument: 'updateLookup' });
    modelsChangeStream.on('change', (change) => {
      console.log('Change detected (models):', change.operationType, change.documentKey?._id);
      modelsChanged=true;
      broadcastChange('models', change);
    });

    const leaderboardScoresChangeStream = leaderboardScores.watch([], { fullDocument: 'updateLookup' });
    leaderboardScoresChangeStream.on('change', (change) => {
      console.log('Change detected (leaderboard_scores):', change.operationType, change.documentKey?._id);
      leaderboardScoresChanged=true;
      broadcastChange('leaderboard_scores', change);
    });

    const envSpecsChangeStream = envSpecs.watch([], { fullDocument: 'updateLookup' });
    envSpecsChangeStream.on('change', (change) => {
      console.log('Change detected (env_specs):', change.operationType, change.documentKey?._id);
      envSpecsChanged=true;
      broadcastChange('env_specs', change);
    });

    const rewardDesigns = dbo.collection("reward_designs");
    const rewardDesignsChangeStream = rewardDesigns.watch([], { fullDocument: 'updateLookup' });
    rewardDesignsChangeStream.on('change', (change) => {
      console.log('Change detected (reward_designs):', change.operationType, change.documentKey?._id);
      rewardDesignsChanged=true;
      broadcastChange('reward_designs', change);
    });

    const experimentDesigns = dbo.collection("experiment_designs");
    const experimentDesignsChangeStream = experimentDesigns.watch([], { fullDocument: 'updateLookup' });
    experimentDesignsChangeStream.on('change', (change) => {
      console.log('Change detected (experiment_designs):', change.operationType, change.documentKey?._id);
      experimentDesignsChanged=true;
      broadcastChange('experiment_designs', change);
    });

    const gyms = dbo.collection("gyms");
    const gymsChangeStream = gyms.watch([], { fullDocument: 'updateLookup' });
    gymsChangeStream.on('change', (change) => {
      console.log('Change detected (gyms):', change.operationType, change.documentKey?._id);
      gymsChanged=true;
      broadcastChange('gyms', change);
    });

    // Index for the Weakness Map (/weakness_map). The logs collection is
    // dominated by per-step "did not fail" rows (no position_history), so
    // an unindexed {job_id, position_history} query degrades to a full
    // collection scan - which for a non-existent job_id never short-
    // circuits and hangs the request. This compound index lets Mongo
    // walk one job's rows newest-first directly. Idempotent / background.
    dbo.collection("logs").createIndex(
      { job_id: 1, _id: -1 },
      { name: "weakness_job_recent", background: true }
    ).then(
      (n) => console.log('logs index ensured:', n),
      (e) => console.warn('logs index create failed:', e && e.message)
    );

  });

  if (config.logging != "none") {
    app.use(morgan(config.logging));
  }

  app.use(cors());
  app.options('*', cors());

  app.use(bodyParser.urlencoded({ extended: true }));
  app.use(bodyParser.json());
  app.use(express.static(path.join(__dirname, '/..')));
  app.get('/leaderboard', (req, res) => {
    const lb: string = path.join(__dirname, '/../leaderboard.html');
    res.sendFile(lb);
  });
  var needsUpdate = function (req, changed) {
    var force = (req.query.force == 'true');
    console.log(`force: ${force} req.query.force: ${req.query.force} jobsChanged: ${changed}`);
    return (changed || force);
  }


  app.get('/leaderboard_scores', (req,res) => {
    if(needsUpdate(req, leaderboardScoresChanged))
    {
      leaderboardScoresChanged=false;
      dbo.collection("leaderboard_scores").find({}).toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} leaderboard scores retrieved`);
        res.json(result);
      });
      return;
    }
    
    console.log(`No leaderboard scores retrieved`);
    res.status(200).send('NO_CHANGES');
  
  });
  app.get('/logs', (req, res) => {
    const lb: string = path.join(__dirname, '/../logs.html');
    res.sendFile(lb);
    // const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);
    // dockerLogs.stdout.on('data', (data) => {
    //   res.write(`${data}\n`);
    // });
    // dockerLogs.stderr.on('data', (data) => {
    //   console.error(`stderr: ${data}`);
    // });
    // dockerLogs.on('close', (code) => {
    //   console.log(`child process exited with code ${code}`);
    //   res.end();
    // });

  })
  app.post('/add_job', (req,res) => {
    console.log("add_job: " + JSON.stringify(req.body));
    dbo.collection("jobs").insertOne(req.body,function(err, result) {
      if (err) throw err;
      //console.log(result);
      res.json(result)
    });
  });
  app.get('/get_jobs', (req,res) => {
    if (needsUpdate(req, jobsChanged)) {
      jobsChanged=false;
      dbo.collection("jobs").find({}).toArray(function(err, result) {
          if (err) throw err;
          //console.log(result);
          console.log(`${result.length} jobs retrieved`)
          res.json(result)
      });
      return;
    }
    
    console.log(`No jobs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  // ----------------------------------------------------------------
  // Weakness heatmap (Phase 1 diagnostic)
  //
  // Aggregates per-step car world positions (logs.position_history)
  // for a single job into a 2D spatial grid and returns the per-cell
  // counters the client blends into a "weakness" score:
  //   * crash rate   = terminal cells of "has failed" episodes / visits
  //   * slowness      = inverse mean per-step travel distance (speed)
  //   * low return    = inverse mean terminal episode score
  //
  // Read-only: consumes data already written by log_reward() in
  // rl_agent/environments/courses/utils/logging.py. position_history
  // is stored as a Python list-repr joined by commas, e.g.
  //   "[1.2, 3.4, 5.6],[1.3, 3.5, 5.7],..."
  // Only "has succeeded" / "has failed - reward" docs carry a
  // position_history (the per-step "did not fail" rows do not), so we
  // filter on its presence to skip the high-volume per-step rows.
  //
  // The ground plane is auto-detected as the two of (x,y,z) with the
  // largest spatial spread (Unity's up-axis is ~constant, so it falls
  // out as the discarded third axis). Sorted newest-first + capped by
  // `limit` so the map reflects RECENT policy behaviour, which matters
  // because the policy drifts over a training run.
  // ----------------------------------------------------------------
  app.get('/weakness_map', (req, res) => {
    const jobId = String(req.query.job_id || '');
    const bins = Math.max(8, Math.min(200, parseInt(String(req.query.bins || '48'), 10) || 48));
    const limit = Math.max(1, Math.min(5000, parseInt(String(req.query.limit || '500'), 10) || 500));
    if (!jobId) { res.status(400).json({ ok: false, error: 'job_id required' }); return; }

    // logs.job_id is written by the trainer as an ObjectId (not the hex
    // string the dashboard passes around), so match on the ObjectId form
    // and keep a string branch for any legacy/string-typed rows. The
    // equality on job_id lets the {job_id:1,_id:-1} index seek directly
    // instead of falling back to a full-collection _id scan.
    const jobOr: any[] = [{ job_id: jobId }];
    try { jobOr.push({ job_id: new ObjectID(jobId) }); } catch (e) { /* not a valid ObjectId */ }
    const jobFilter = { $and: [{ $or: jobOr }, { position_history: { $ne: null } }] };

    dbo.collection('logs')
      .find(jobFilter)
      .project({ type: 1, score: 1, position_history: 1 })
      .sort({ _id: -1 })
      .limit(limit)
      .maxTimeMS(20000)
      .toArray(function (err, docs) {
        if (err) { res.status(500).json({ ok: false, error: String(err) }); return; }
        docs = docs || [];

        // Parse each doc's position_history string into [x,y,z] points.
        const trajs: Array<{ pts: number[][], type: string, score: number }> = [];
        for (const d of docs) {
          const s: string = d.position_history;
          if (!s) continue;
          const toks = s.match(/\[[^\]]*\]/g) || [];
          const pts: number[][] = [];
          for (const tok of toks) {
            const parts = tok.slice(1, -1).split(',');
            if (parts.length < 3) continue;
            const x = parseFloat(parts[0]), y = parseFloat(parts[1]), z = parseFloat(parts[2]);
            if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
            pts.push([x, y, z]);
          }
          if (pts.length) trajs.push({ pts, type: String(d.type || ''), score: Number(d.score) });
        }

        if (!trajs.length) {
          res.json({ ok: true, job_id: jobId, episodes: 0, cells: [] });
          return;
        }

        // Auto-pick the two highest-spread axes as the ground plane.
        const mins = [Infinity, Infinity, Infinity];
        const maxs = [-Infinity, -Infinity, -Infinity];
        for (const t of trajs) for (const p of t.pts) for (let k = 0; k < 3; k++) {
          if (p[k] < mins[k]) mins[k] = p[k];
          if (p[k] > maxs[k]) maxs[k] = p[k];
        }
        const ranges = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
        const order = [0, 1, 2].sort((a, b) => ranges[b] - ranges[a]);
        const ax = order[0], ay = order[1];
        let xmin = mins[ax], xmax = maxs[ax], ymin = mins[ay], ymax = maxs[ay];
        if (!(xmax > xmin)) xmax = xmin + 1;
        if (!(ymax > ymin)) ymax = ymin + 1;

        const nx = bins, ny = bins;
        const cw = (xmax - xmin) / nx, ch = (ymax - ymin) / ny;
        const binOf = (x: number, y: number) => {
          let ix = Math.floor((x - xmin) / cw); if (ix < 0) ix = 0; if (ix >= nx) ix = nx - 1;
          let iy = Math.floor((y - ymin) / ch); if (iy < 0) iy = 0; if (iy >= ny) iy = ny - 1;
          return iy * nx + ix;
        };

        const N = nx * ny;
        const visits = new Float64Array(N), speedSum = new Float64Array(N), speedCnt = new Float64Array(N);
        const crash = new Float64Array(N), goal = new Float64Array(N);
        const scoreSum = new Float64Array(N), scoreCnt = new Float64Array(N);
        // Dense "pre-crash risk" field: counts step-visits that fall within
        // the last HAZARD_WINDOW steps before a crash. Unlike `crash` (which
        // only marks the exact terminal cell), this lights up the whole
        // approach a trajectory takes into its demise, so the signal is
        // dense across visited cells and spatially meaningful ("cars that
        // pass through here tend to die soon after").
        const hazard = new Float64Array(N);
        const HAZARD_WINDOW = 30;

        for (const t of trajs) {
          const P = t.pts;
          for (let i = 0; i < P.length; i++) {
            const b = binOf(P[i][ax], P[i][ay]);
            visits[b] += 1;
            if (i > 0) {
              const dx = P[i][ax] - P[i - 1][ax], dy = P[i][ay] - P[i - 1][ay];
              speedSum[b] += Math.sqrt(dx * dx + dy * dy);
              speedCnt[b] += 1;
            }
          }
          const last = P[P.length - 1];
          const lb = binOf(last[ax], last[ay]);
          const isFail = t.type.indexOf('fail') >= 0;
          if (isFail) {
            crash[lb] += 1;
            const start = Math.max(0, P.length - HAZARD_WINDOW);
            for (let i = start; i < P.length; i++) {
              hazard[binOf(P[i][ax], P[i][ay])] += 1;
            }
          } else if (t.type.indexOf('succeed') >= 0) {
            goal[lb] += 1;
          }
          if (isFinite(t.score)) { scoreSum[lb] += t.score; scoreCnt[lb] += 1; }
        }

        // Sparse emission: only cells that were actually visited.
        const cells: any[] = [];
        for (let b = 0; b < N; b++) {
          if (visits[b] <= 0) continue;
          cells.push({
            ix: b % nx, iy: Math.floor(b / nx),
            v: visits[b], ss: speedSum[b], sc: speedCnt[b],
            cr: crash[b], gl: goal[b], scs: scoreSum[b], scc: scoreCnt[b],
            hz: hazard[b],
          });
        }

        const axisNames = ['x', 'y', 'z'];
        res.json({
          ok: true, job_id: jobId, episodes: trajs.length,
          axes: [ax, ay], axisNames: [axisNames[ax], axisNames[ay]],
          extent: { xmin, xmax, ymin, ymax }, nx, ny, cells,
        });
      });
  });

  app.get('/jobs', (req, res) => {
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
    const lb: string = path.join(__dirname, '/../jobs.html');
    res.sendFile(lb);
  });

  app.post('/update_job_status', (req, res) => {
    console.log("update_job_status: " + JSON.stringify(req.body));
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    const newvalues = { "$set": { "status": job["status"] } };
    const options = { upsert: false };
    dbo.collection("jobs").updateOne(myquery,newvalues, options, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  app.post('/delete_job', (req,res) => {
    var job = req.body;
    const myquery = { "_id": ObjectID(job["_id"]) };
    console.log(myquery);
    dbo.collection("jobs").deleteOne(myquery, function(err, result) {
      if (err) throw err;
      console.log(result);
      res.json(result)
    });
  });

  // Full-cleanup delete of a model. Removes, in order:
  //   1. the saved-model files on disk at the model's `location`
  //      (e.g. /saved_models/robotaxi/SacAgent/8) — bounded to the
  //      /saved_models tree for safety,
  //   2. the leaderboard_scores rows that reference it (matched by
  //      `path` == location), and
  //   3. the model document itself.
  //
  // `location` is passed by the client (read off the model row) so we
  // don't need a pre-read; if absent we still delete the DB doc and skip
  // the file step. The DB doc delete is last so a file/score failure
  // doesn't orphan the record without the user knowing.
  app.post('/delete_model', (req, res) => {
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }

    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      idFilter = { "_id": String(body._id) }; // legacy string _id
    }

    const location = typeof body.location === 'string' ? body.location.trim() : '';
    const summary: any = { files_deleted: false, scores_deleted: 0 };

    // 1. Delete on-disk files. Safety: only paths strictly under
    //    /saved_models/ (never the root itself) are eligible, so a
    //    malformed/empty location can't wipe the whole tree.
    const SAVED_ROOT = '/saved_models';
    if (location.startsWith(SAVED_ROOT + '/')) {
      const resolved = path.resolve(location);
      if (resolved.startsWith(SAVED_ROOT + path.sep) && resolved !== SAVED_ROOT) {
        try {
          if (fs.existsSync(resolved)) {
            fs.rmSync(resolved, { recursive: true, force: true });
            summary.files_deleted = true;
            console.log('delete_model: removed files at', resolved);
          }
        } catch (e) {
          console.error('delete_model: file removal failed for', resolved, e);
          summary.file_error = String((e as Error).message || e);
        }
      }
    }

    // 2. Delete leaderboard_scores rows referencing this model by path.
    const removeScores = (cb: () => void) => {
      if (!location) { cb(); return; }
      dbo.collection("leaderboard_scores").deleteMany(
        { path: location },
        function(err, result) {
          if (err) {
            console.error('delete_model: leaderboard_scores cleanup failed:', err);
          } else if (result) {
            summary.scores_deleted = result.deletedCount || 0;
          }
          cb();
        });
    };

    // 3. Delete the model document.
    removeScores(() => {
      dbo.collection("models").deleteOne(idFilter, function(err, result) {
        if (err) {
          console.error('delete_model failed:', err);
          res.status(500).json({ error: String(err.message || err), ...summary });
          return;
        }
        console.log('delete_model: removed model', body._id, summary);
        res.json({ ...result, ...summary });
      });
    });
  });

  // Global job-queue pause switch. A singleton doc in queue_control that
  // the trainer's run_jobs_loop polls (_is_queue_paused). When paused, the
  // trainer stops picking up NEW jobs but leaves all job statuses alone, so
  // the operator can halt the queue, test something, then resume where it
  // left off. Distinct from per-job Pause (which checkpoints + PAUSES a
  // single running job).
  app.get('/get_queue_state', (req, res) => {
    dbo.collection("queue_control").findOne({ _id: "singleton" }, function(err, doc) {
      if (err) {
        console.error('get_queue_state failed:', err);
        res.status(500).json({ error: String(err) });
        return;
      }
      res.json({ paused: !!(doc && (doc as any).paused) });
    });
  });

  app.post('/set_queue_state', (req, res) => {
    const paused = !!(req.body && req.body.paused);
    dbo.collection("queue_control").updateOne(
      { _id: "singleton" },
      { "$set": { paused: paused, updated_at: new Date() } },
      { upsert: true },
      function(err) {
        if (err) {
          console.error('set_queue_state failed:', err);
          res.status(500).json({ error: String(err) });
          return;
        }
        console.log('queue paused =', paused);

        // Also pause/resume the currently-running job(s), reusing the
        // per-job lifecycle machinery the trainer already understands:
        //   * Pausing: flip IN_PROGRESS jobs to PAUSE_REQUESTED. The
        //     trainer's training loop polls this each iteration, saves a
        //     Learner checkpoint, and sets PAUSED. We tag them with
        //     queue_auto_paused so resume knows which jobs WE paused (vs
        //     jobs the operator paused by hand, which should stay paused).
        //   * Resuming: flip those tagged jobs back to NOT_STARTED so the
        //     trainer re-picks them (FIFO) and resumes from the checkpoint
        //     via its resume-detection path; clear the tag.
        const jobsCol = dbo.collection("jobs");
        const after = (e2: any, summary: any) => {
          if (e2) {
            console.error('set_queue_state job update failed:', e2);
            res.status(500).json({ error: String(e2) });
            return;
          }
          res.json({ ok: true, paused, jobs: summary });
        };
        if (paused) {
          jobsCol.updateMany(
            { status: "IN_PROGRESS" },
            { "$set": { status: "PAUSE_REQUESTED", queue_auto_paused: true } },
            (e2, r) => after(e2, { paused: r ? r.modifiedCount : 0 }));
        } else {
          jobsCol.updateMany(
            { queue_auto_paused: true },
            { "$set": { status: "NOT_STARTED" }, "$unset": { queue_auto_paused: "" } },
            (e2, r) => after(e2, { resumed: r ? r.modifiedCount : 0 }));
        }
      });
  });

  app.get('/models', (req, res) => {
    const lb: string = path.join(__dirname, '/../models.html');
    res.sendFile(lb);
  });

  // Side-by-side model comparison view. Reachable both as a standalone
  // page and as a GoldenLayout tab; the heavy lifting (joining models
  // with leaderboard_scores, computing stats) is done client-side from
  // the existing /get_models and /leaderboard_scores endpoints, so no
  // dedicated data endpoint is needed here.
  app.get('/analysis', (req, res) => {
    const lb: string = path.join(__dirname, '/../analysis.html');
    res.sendFile(lb);
  });

  // Reward Design tab. Same pattern as /jobs / /models / /analysis -
  // the static HTML is also served by express.static so this clean
  // route is convenience; the iframe in goldenlayout requests the
  // .html file directly to side-step any rebuild-lag on this route.
  app.get('/reward_designs', (req, res) => {
    const f: string = path.join(__dirname, '/../reward_designs.html');
    res.sendFile(f);
  });

  // Experiment Design tab. Sibling of /reward_designs above. The
  // iframe in goldenlayout requests the .html file directly so the
  // tab works the moment the static file lands on disk, without
  // needing the dashboard container to be rebuilt to pick up this
  // route.
  app.get('/experiment_designs', (req, res) => {
    const f: string = path.join(__dirname, '/../experiment_designs.html');
    res.sendFile(f);
  });

  app.get('/gyms', (req, res) => {
    const f: string = path.join(__dirname, '/../gyms.html');
    res.sendFile(f);
  });

  // ---------------------------------------------------------------- *
  // Desired-gym state for the Unity supervisor hot-swap feature.
  //
  // When do_job in robotaxi.py picks up a job that carries a gym,
  // it POSTs to /set_desired_gym with the gym's file_path. Each
  // RunClientWrapper.ps1 supervisor polls /get_desired_gym?index=N
  // and restarts the Unity process with the new binary if the path
  // has changed.
  //
  // State is in-memory (keyed by actor index or '*' for all actors).
  // It does not survive a dashboard container restart, which is
  // intentional: after a restart the supervisors poll once more and
  // the running job's do_job will re-signal if it hasn't finished yet.
  // ---------------------------------------------------------------- *
  const desiredGymState: Map<string, any> = new Map();

  app.get('/get_desired_gym', (req, res) => {
    const idx = req.query.index !== undefined ? String(req.query.index) : '*';
    // Return the per-index entry if present, otherwise the wildcard entry.
    const state = desiredGymState.get(idx) || desiredGymState.get('*') || null;
    res.json(state || {});
  });

  app.post('/set_desired_gym', (req, res) => {
    const body = req.body || {};
    if (!body.file_path) {
      res.status(400).json({ error: 'file_path is required' });
      return;
    }
    // If index is omitted or '*', broadcast to all actors.
    const idx = body.index !== undefined ? String(body.index) : '*';
    const state = {
      gym_id:    String(body.gym_id   || ''),
      gym_name:  String(body.gym_name || ''),
      file_path: String(body.file_path),
      set_at:    new Date().toISOString(),
    };
    desiredGymState.set(idx, state);
    console.log(`[desired-gym] index=${idx} -> ${state.file_path}`);
    res.json({ ok: true, index: idx, state });
  });

  // MadScientist Lab tab. Phase 0 just serves the static stub so the
  // GoldenLayout tab in the left column (next to the Tensorboard tabs)
  // has somewhere to point. Phase 1 will add data endpoints under
  // /madscientist/activity, /madscientist/proposals, etc. that this
  // page polls on the standard 5s interval.
  app.get('/madscientist', (req, res) => {
    const f: string = path.join(__dirname, '/../madscientist.html');
    // The HTML embeds an inline <script>, so a stale cached copy of
    // this file ships stale JS too. Disable the HTTP cache for this
    // route so iterating on the dashboard during dev doesn't require
    // a hard reload every time. Hot path is cheap (one file read per
    // tab open).
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.sendFile(f);
  });

  // ---- MadScientist API ----------------------------------------------
  //
  // Three endpoints feed the Mad Scientist Lab dashboard tab. Each is
  // intentionally narrow:
  //
  //   GET /madscientist/proposals?status=<csv>&limit=<n>
  //       Returns proposals filtered by status. status defaults to
  //       "pending_user,implementing,pr_open,training,done,failed"
  //       (everything an operator might want to see). Heavy fields
  //       (rubric markdown, full audit_events) are stripped server-
  //       side to keep the payload small for the 5s poll cadence.
  //
  //   GET /madscientist/activity?limit=<n>
  //       Returns recent audit_events flattened across all proposals,
  //       sorted by timestamp desc. Renders as the activity feed.
  //
  //   POST /madscientist/decide
  //       Body: { proposal_id, action: "approve"|"reject"|"defer", note }
  //       Updates the proposal's status + decision fields. Phase 1A
  //       does NOT yet trigger the orchestrator on "approve" - that
  //       lands in Phase 1C. For now an approved proposal just sits
  //       at status=approved waiting for the human to act on it.
  //
  // The status enum + collection name come straight from the Python
  // side (rl_agent/madscientist/constants.py); we duplicate them
  // here because the dashboard server doesn't import Python modules.
  // Drift risk is low because the values are extremely stable.

  const MS_PROPOSALS_COLL = 'proposals';
  const MS_VALID_DECISIONS = new Set([
    'approve', 'reject', 'defer', 'approve_with_revisions',
  ]);
  const MS_OBJID_RE = /^[a-fA-F0-9]{24}$/;

  app.get('/madscientist/proposals', (req, res) => {
    const rawStatus = (req.query.status as string | undefined) || '';
    let statusFilter: string[];
    if (rawStatus.trim()) {
      statusFilter = rawStatus.split(',').map((s) => s.trim()).filter((s) => s.length);
    } else {
      // Default: everything except still-being-judged (those don't
      // need UI attention yet) and the explicitly-rejected ones
      // (they live in the activity feed instead).
      statusFilter = [
        'pending_user', 'deferred', 'approved', 'implementing',
        'pr_open', 'training', 'done', 'failed',
      ];
    }
    const limit = Math.max(1, Math.min(200,
      parseInt((req.query.limit as string) || '50', 10) || 50));

    // Projection: omit heavy / sensitive / not-yet-used fields. Keeps
    // the polled payload small.
    const projection = {
      // never send: rubric markdown, full audit_event details (we
      // serve those separately from /madscientist/activity),
      // implementation_log streams (those are tailed via a
      // dedicated endpoint in Phase 1C).
      audit_events: 0,
      implementation_log: 0,
    };

    dbo.collection(MS_PROPOSALS_COLL)
      .find({ status: { $in: statusFilter } }, { projection })
      .sort({ updated_at: -1, created_at: -1 })
      .limit(limit)
      .toArray((err, docs) => {
        if (err) {
          console.error('GET /madscientist/proposals:', err);
          res.status(500).json({ error: String(err) });
          return;
        }
        res.json(docs || []);
      });
  });

  // GET /madscientist/proposals/:id
  //   Returns the full proposal document including audit_events and
  //   the full judge_review (no projection). Used by the Activity
  //   Feed's "view details" modal so the operator can review the
  //   proposal + judge review + audit timeline after approval (the
  //   carousel only shows pending proposals).
  app.get('/madscientist/proposals/:id', (req, res) => {
    const id = String(req.params.id || '').trim();
    if (!MS_OBJID_RE.test(id)) {
      res.status(400).json({ error: 'proposal id must be a 24-char hex ObjectId' });
      return;
    }
    dbo.collection(MS_PROPOSALS_COLL).findOne(
      { _id: ObjectID(id) },
      {},
      (err, doc) => {
        if (err) {
          console.error('GET /madscientist/proposals/:id:', err);
          res.status(500).json({ error: String(err) });
          return;
        }
        if (!doc) {
          res.status(404).json({ error: 'proposal not found' });
          return;
        }
        res.json(doc);
      });
  });

  app.get('/madscientist/activity', (req, res) => {
    const limit = Math.max(1, Math.min(200,
      parseInt((req.query.limit as string) || '30', 10) || 30));
    // Page offset (number of most-recent events to skip). Enables the
    // dashboard's Prev/Next paging through older activity.
    const offset = Math.max(0,
      parseInt((req.query.offset as string) || '0', 10) || 0);

    // Aggregate the most-recent N audit_events across all proposals,
    // sorted by timestamp desc. We use a Mongo aggregation so the
    // unwind + sort + limit happens server-side instead of streaming
    // every proposal across the wire.
    //
    // For proposals in `training` state, we $lookup the related TRAIN
    // jobs and project their status array as `job_statuses`. The
    // dashboard Activity Feed uses this to render an aggregated job-
    // status badge (e.g. "IN_PROGRESS 2/4 · 1 FAILED") instead of the
    // bare `training` proposal status. Lookup happens BEFORE $unwind
    // so we do one lookup per proposal, not one per event.
    dbo.collection(MS_PROPOSALS_COLL).aggregate([
      { $match: { audit_events: { $exists: true, $ne: [] } } },
      // $lookup with a let + $expr because proposals._id is ObjectId
      // and jobs.proposal_id is stored as the 24-char hex STRING (the
      // madscientist orchestrator stamps it via str(proposal['_id'])
      // - see rl_agent/madscientist/orchestrator.py). Plain
      // localField/foreignField $lookup does strict equality and
      // would return [] for every row because ObjectId != str.
      { $lookup: {
          from: 'jobs',
          let: { proposalIdStr: { $toString: '$_id' } },
          pipeline: [
            { $match: { $expr: { $eq: ['$proposal_id', '$$proposalIdStr'] } } },
            { $project: { _id: 0, status: 1 } },
          ],
          as: '_jobs',
        } },
      { $unwind: '$audit_events' },
      { $project: {
          _id: 0,
          proposal_id: '$_id',
          proposal_title: '$title',
          proposal_status: '$status',
          at: '$audit_events.at',
          by_agent: '$audit_events.by_agent',
          event: '$audit_events.event',
          detail: '$audit_events.detail',
          job_statuses: '$_jobs.status',
        } },
      { $sort: { at: -1 } },
      // $facet returns one page of events AND the total event count in a
      // single round-trip, so the dashboard pager can show "X–Y of Z"
      // and disable Next at the end without a second query.
      { $facet: {
          rows:  [ { $skip: offset }, { $limit: limit } ],
          total: [ { $count: 'count' } ],
      } },
    ]).toArray((err, result) => {
      if (err) {
        console.error('GET /madscientist/activity:', err);
        res.status(500).json({ error: String(err) });
        return;
      }
      const facet = (result && result[0]) || {};
      const rows = facet.rows || [];
      const total = (facet.total && facet.total[0] && facet.total[0].count) || 0;
      res.json({ rows, total, offset, limit });
    });
  });

  // POST /madscientist/proposals/:id/edit
  //   Body: { patch: { <field>: <value>, ... } }
  //   Mutates whitelisted fields on a proposal that's still in
  //   `pending_user` or `deferred` state. Once the operator approves /
  //   rejects / defers (or the orchestrator picks it up), the proposal
  //   is locked - mutating it after the orchestrator has queued TRAIN
  //   jobs would silently break the experiment design.
  //
  //   The whitelist below is intentionally minimal - everything an
  //   operator would realistically want to nudge before approving (the
  //   hypothesis text, the success threshold, seeds/iters/wall-time).
  //   Editing experiment_arms is NOT supported here; reject and let
  //   the researcher regenerate is the right flow for that.
  //
  //   Every edit appends an audit_events entry with a field-level diff
  //   so the operator's nudges are reconstructible later.
  const MS_EDITABLE_TOP_LEVEL = new Set([
    'title', 'hypothesis', 'motivation', 'code_changes_summary',
    'n_seeds_per_arm', 'num_iterations_per_seed',
    'expected_wall_time_hours',
  ]);
  const MS_EDITABLE_PRIMARY = new Set([
    'metric', 'arm_a', 'arm_b', 'comparator',
    'threshold', 'threshold_kind',
  ]);
  const MS_VALID_COMPARATORS = new Set(['>=', '<=', '>', '<']);
  const MS_VALID_THRESHOLD_KINDS = new Set(['relative', 'absolute']);
  const MS_EDITABLE_STATUSES = new Set(['pending_user', 'deferred']);

  app.post('/madscientist/proposals/:id/edit', (req, res) => {
    const proposalId = String(req.params.id || '').trim();
    if (!MS_OBJID_RE.test(proposalId)) {
      res.status(400).json({ error: 'proposal id must be a 24-char hex ObjectId' });
      return;
    }
    const body = req.body || {};
    const rawPatch = (body.patch && typeof body.patch === 'object') ? body.patch : null;
    if (!rawPatch) {
      res.status(400).json({ error: 'body.patch object required' });
      return;
    }

    // Validate + assemble the $set + diff trail. We fetch the
    // existing doc first so the diff can record before-values; that
    // makes the audit log readable ("hypothesis: 'A' -> 'B'") rather
    // than just "field X was edited".
    dbo.collection(MS_PROPOSALS_COLL).findOne(
      { _id: ObjectID(proposalId) },
      {},
      (findErr, existing) => {
        if (findErr) {
          console.error('POST /madscientist/proposals/:id/edit findOne:', findErr);
          res.status(500).json({ error: String(findErr) });
          return;
        }
        if (!existing) {
          res.status(404).json({ error: 'proposal not found' });
          return;
        }
        if (!MS_EDITABLE_STATUSES.has(existing.status)) {
          res.status(409).json({
            error: `proposal status=${existing.status} is no longer editable. ` +
              `Only pending_user / deferred proposals can be edited.`,
          });
          return;
        }

        const set: Record<string, any> = {};
        const diff: Record<string, { from: any; to: any }> = {};

        // Top-level fields
        for (const k of Object.keys(rawPatch)) {
          if (!MS_EDITABLE_TOP_LEVEL.has(k)) continue;
          const v = rawPatch[k];
          let normalized: any = v;
          if (k === 'n_seeds_per_arm' || k === 'num_iterations_per_seed') {
            const n = Number.parseInt(String(v), 10);
            if (!Number.isFinite(n) || n < 1 || n > 100000) {
              res.status(400).json({
                error: `${k} must be an integer in [1, 100000]; got ${JSON.stringify(v)}`,
              });
              return;
            }
            normalized = n;
          } else if (k === 'expected_wall_time_hours') {
            // null allowed (means "unknown"); otherwise positive float
            if (v === null || v === undefined || v === '') {
              normalized = null;
            } else {
              const f = Number.parseFloat(String(v));
              if (!Number.isFinite(f) || f < 0 || f > 10000) {
                res.status(400).json({
                  error: `expected_wall_time_hours must be a non-negative float < 10000; ` +
                    `got ${JSON.stringify(v)}`,
                });
                return;
              }
              normalized = f;
            }
          } else {
            if (typeof v !== 'string') {
              res.status(400).json({
                error: `${k} must be a string; got ${typeof v}`,
              });
              return;
            }
            normalized = v.slice(0, 16000);
          }
          if (existing[k] !== normalized) {
            set[k] = normalized;
            diff[k] = { from: existing[k] ?? null, to: normalized };
          }
        }

        // success_criteria.primary (string)
        if (typeof rawPatch['success_criteria.primary'] === 'string') {
          const v = rawPatch['success_criteria.primary'].slice(0, 8000);
          const existingPrimary = (existing.success_criteria && existing.success_criteria.primary) || '';
          if (existingPrimary !== v) {
            set['success_criteria.primary'] = v;
            diff['success_criteria.primary'] = { from: existingPrimary, to: v };
          }
        }

        // success_criteria.primary_parsed.{metric,arm_a,arm_b,comparator,threshold,threshold_kind}
        for (const k of Array.from(MS_EDITABLE_PRIMARY)) {
          const dottedKey = `success_criteria.primary_parsed.${k}`;
          if (!(dottedKey in rawPatch)) continue;
          let v = rawPatch[dottedKey];
          if (k === 'comparator') {
            if (!MS_VALID_COMPARATORS.has(String(v))) {
              res.status(400).json({
                error: `comparator must be one of ${Array.from(MS_VALID_COMPARATORS).join(', ')}; ` +
                  `got ${JSON.stringify(v)}`,
              });
              return;
            }
          } else if (k === 'threshold_kind') {
            if (!MS_VALID_THRESHOLD_KINDS.has(String(v))) {
              res.status(400).json({
                error: `threshold_kind must be one of ${Array.from(MS_VALID_THRESHOLD_KINDS).join(', ')}; ` +
                  `got ${JSON.stringify(v)}`,
              });
              return;
            }
          } else if (k === 'threshold') {
            const f = Number.parseFloat(String(v));
            if (!Number.isFinite(f)) {
              res.status(400).json({
                error: `threshold must be a finite number; got ${JSON.stringify(v)}`,
              });
              return;
            }
            v = f;
          } else {
            // metric / arm_a / arm_b: strings, length-capped
            if (typeof v !== 'string' || !v.trim()) {
              res.status(400).json({
                error: `${k} must be a non-empty string; got ${JSON.stringify(v)}`,
              });
              return;
            }
            v = v.trim().slice(0, 200);
          }
          const existingParsed = (existing.success_criteria
            && existing.success_criteria.primary_parsed) || {};
          if (existingParsed[k] !== v) {
            set[dottedKey] = v;
            diff[dottedKey] = { from: existingParsed[k] ?? null, to: v };
          }
        }

        if (Object.keys(set).length === 0) {
          res.json({
            ok: true,
            proposal_id: proposalId,
            updated_fields: [],
            note: 'patch contained no changes',
          });
          return;
        }

        const now = new Date();
        set.updated_at = now;
        const auditEvent = {
          at: now,
          by_agent: 'user',
          event: 'edited',
          detail: {
            source: 'dashboard',
            field_diff: diff,
          },
        };

        dbo.collection(MS_PROPOSALS_COLL).updateOne(
          { _id: ObjectID(proposalId), status: { $in: Array.from(MS_EDITABLE_STATUSES) } },
          { $set: set, $push: { audit_events: auditEvent } },
          {},
          (updErr, result) => {
            if (updErr) {
              console.error('POST /madscientist/proposals/:id/edit updateOne:', updErr);
              res.status(500).json({ error: String(updErr) });
              return;
            }
            if (result.matchedCount === 0) {
              // Lost the race: status changed between findOne and updateOne.
              res.status(409).json({
                error: 'proposal status changed concurrently; refresh and retry',
              });
              return;
            }
            res.json({
              ok: true,
              proposal_id: proposalId,
              updated_fields: Object.keys(set).filter((k) => k !== 'updated_at'),
            });
          });
      });
  });

  // GET /madscientist/act?token=<pid.action.exp.hmac>
  //   The magic-link target for the Approve / Reject / Defer buttons in
  //   the proposal notification email (see
  //   rl_agent/madscientist/email_bridge.py). Verifies the HMAC-SHA256
  //   token (signed with MADSCIENTIST_TOKEN_SECRET, must match the
  //   Python signer), applies the decision via the same logic as
  //   /madscientist/decide, and returns a small HTML confirmation page
  //   (since it's opened in a browser from an email client).
  //
  //   "Single use" is enforced by gating the update on the proposal
  //   still being in pending_user/deferred - a stale link can't
  //   override a decision already taken (e.g. can't reject a proposal
  //   that's already approved + training).
  const MS_TOKEN_SECRET = (process.env.MADSCIENTIST_TOKEN_SECRET || '').trim();
  const MS_ACT_STATUSES = new Set(['pending_user', 'deferred']);

  function msActPage(opts: { ok: boolean; heading: string; body: string }): string {
    const accent = opts.ok ? '#4f46e5' : '#dc2626';
    return `<!DOCTYPE html><html><head><meta charset="utf-8">` +
      `<meta name="viewport" content="width=device-width,initial-scale=1">` +
      `<title>MadScientist decision</title></head>` +
      `<body style="margin:0;background:#0f172a;font-family:Arial,sans-serif">` +
      `<div style="max-width:520px;margin:40px auto;background:#ffffff;` +
      `border-radius:14px;overflow:hidden;border:1px solid #e2e8f0">` +
      `<div style="padding:18px 22px;background:#0f172a;color:#818cf8;` +
      `font-size:11px;letter-spacing:0.08em;text-transform:uppercase;` +
      `font-weight:700">MadScientist</div>` +
      `<div style="padding:24px">` +
      `<h2 style="margin:0 0 10px 0;color:${accent};font-size:20px">${opts.heading}</h2>` +
      `<div style="color:#334155;font-size:14px;line-height:1.6">${opts.body}</div>` +
      `<div style="margin-top:22px"><a href="/madscientist" ` +
      `style="display:inline-block;padding:10px 18px;background:#4f46e5;` +
      `color:#fff;text-decoration:none;border-radius:8px;font-weight:600;` +
      `font-size:14px">Open the dashboard</a></div>` +
      `</div></div></body></html>`;
  }

  app.get('/madscientist/act', (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    const token = String(req.query.token || '');

    if (!MS_TOKEN_SECRET) {
      res.status(503).send(msActPage({
        ok: false,
        heading: 'One-click decisions are disabled',
        body: 'MADSCIENTIST_TOKEN_SECRET is not configured on the server, ' +
          'so signed decision links can\'t be verified. Use the dashboard ' +
          'to decide.',
      }));
      return;
    }

    const parts = token.split('.');
    if (parts.length !== 4) {
      res.status(400).send(msActPage({
        ok: false, heading: 'Invalid link',
        body: 'This decision link is malformed.',
      }));
      return;
    }
    const [pid, action, expStr, sig] = parts;
    const signed = `${pid}.${action}.${expStr}`;
    const expected = crypto.createHmac('sha256', MS_TOKEN_SECRET)
      .update(signed).digest('hex');
    // Constant-time compare; lengths must match for timingSafeEqual.
    const sigBuf = Buffer.from(sig);
    const expBuf = Buffer.from(expected);
    const sigOk = sigBuf.length === expBuf.length &&
      crypto.timingSafeEqual(sigBuf, expBuf);
    if (!sigOk) {
      res.status(403).send(msActPage({
        ok: false, heading: 'Invalid or tampered link',
        body: 'The signature on this decision link did not verify.',
      }));
      return;
    }
    const exp = parseInt(expStr, 10);
    if (!Number.isFinite(exp) || Math.floor(Date.now() / 1000) > exp) {
      res.status(410).send(msActPage({
        ok: false, heading: 'Link expired',
        body: 'This decision link has expired. Use the dashboard to decide.',
      }));
      return;
    }
    if (!MS_ACT_STATUSES.has('pending_user') || !MS_VALID_DECISIONS.has(action)) {
      res.status(400).send(msActPage({
        ok: false, heading: 'Invalid action',
        body: `Action ${action} is not recognized.`,
      }));
      return;
    }
    if (!MS_OBJID_RE.test(pid)) {
      res.status(400).send(msActPage({
        ok: false, heading: 'Invalid link',
        body: 'The proposal id in this link is malformed.',
      }));
      return;
    }

    const now = new Date();
    let nextStatus: string;
    if (action === 'reject') nextStatus = 'rejected';
    else if (action === 'defer') nextStatus = 'deferred';
    else nextStatus = 'approved';

    const decision = {
      at: now, by: 'user', action, note: '',
      revision_applied: false, source: 'email',
    };
    const auditEvent = {
      at: now, by_agent: 'user', event: 'decided',
      detail: { action, next_status: nextStatus, source: 'email' },
    };

    // Gate the update on the proposal still awaiting a decision so a
    // re-clicked / stale link can't override an already-made decision.
    dbo.collection(MS_PROPOSALS_COLL).findOneAndUpdate(
      { _id: ObjectID(pid), status: { $in: Array.from(MS_ACT_STATUSES) } },
      { $set: { status: nextStatus, decision, updated_at: now },
        $push: { audit_events: auditEvent } },
      { returnDocument: 'before' },
      (err, result) => {
        if (err) {
          console.error('GET /madscientist/act:', err);
          res.status(500).send(msActPage({
            ok: false, heading: 'Server error',
            body: 'Something went wrong applying your decision. Try the dashboard.',
          }));
          return;
        }
        const matched = result && result.value;
        if (!matched) {
          // Either the proposal doesn't exist or it's no longer
          // awaiting a decision (already decided / in flight).
          dbo.collection(MS_PROPOSALS_COLL).findOne(
            { _id: ObjectID(pid) }, { projection: { status: 1, title: 1 } },
            (e2, cur) => {
              if (e2 || !cur) {
                res.status(404).send(msActPage({
                  ok: false, heading: 'Proposal not found',
                  body: 'This proposal no longer exists.',
                }));
                return;
              }
              res.status(409).send(msActPage({
                ok: false, heading: 'Already decided',
                body: `This proposal is already <b>${cur.status}</b>, so your ` +
                  `<b>${action}</b> click had no effect. Open the dashboard ` +
                  `to see its current state.`,
              }));
            });
          return;
        }
        const title = matched.title || '(untitled)';
        const verb = action === 'approve' ? 'approved'
          : action === 'reject' ? 'rejected' : 'deferred';
        res.send(msActPage({
          ok: true,
          heading: `Proposal ${verb}`,
          body: `<b>${String(title).replace(/[&<>]/g, '')}</b> is now ` +
            `<b>${nextStatus}</b>.` +
            (action === 'approve'
              ? ' The orchestrator will queue its training jobs shortly.'
              : ''),
        }));
      });
  });

  app.post('/madscientist/decide', (req, res) => {
    const body = req.body || {};
    const proposalId = String(body.proposal_id || '').trim();
    const action = String(body.action || '').trim();
    const note = String(body.note || '').slice(0, 4000);

    if (!MS_OBJID_RE.test(proposalId)) {
      res.status(400).json({ error: 'proposal_id must be a 24-char hex ObjectId' });
      return;
    }
    if (!MS_VALID_DECISIONS.has(action)) {
      res.status(400).json({
        error: `action must be one of ${Array.from(MS_VALID_DECISIONS).join(', ')}`,
      });
      return;
    }

    const now = new Date();
    // Map the action to the next status:
    //   approve / approve_with_revisions -> "approved" (Phase 1C
    //     orchestrator will pick this up). For now it just sits.
    //   reject -> "rejected" (terminal, no further worker action).
    //   defer -> "deferred" (soft state; user can revisit later).
    let nextStatus: string;
    if (action === 'reject') {
      nextStatus = 'rejected';
    } else if (action === 'defer') {
      nextStatus = 'deferred';
    } else {
      nextStatus = 'approved';
    }

    const decision = {
      at: now,
      by: 'user',
      action,
      note,
      revision_applied: action === 'approve_with_revisions',
      source: 'dashboard',
    };
    const auditEvent = {
      at: now,
      by_agent: 'user',
      event: 'decided',
      detail: {
        action,
        next_status: nextStatus,
        note_chars: note.length,
        source: 'dashboard',
      },
    };

    dbo.collection(MS_PROPOSALS_COLL).updateOne(
      { _id: ObjectID(proposalId) },
      {
        $set: { status: nextStatus, decision, updated_at: now },
        $push: { audit_events: auditEvent },
      },
      {},
      (err, result) => {
        if (err) {
          console.error('POST /madscientist/decide:', err);
          res.status(500).json({ error: String(err) });
          return;
        }
        if (result.matchedCount === 0) {
          res.status(404).json({ error: 'proposal not found' });
          return;
        }
        res.json({
          ok: true,
          proposal_id: proposalId,
          new_status: nextStatus,
          action,
        });
      });
  });
  app.get('/get_models', (req,res) => {
    if(needsUpdate(req, modelsChanged))
    {
      modelsChanged=false;
      // Projection excludes ``reward_design_code`` from the list-view
      // payload. That field stores the verbatim Python source of the
      // model's training-time reward design - small per model (often
      // ~1 KB) but with thousands of historical model docs it
      // dominates the /get_models response size (multi-MB JSON parsed
      // on every dashboard load). The Models tab's "View source"
      // modal fetches the source on click via /get_model_source/:id
      // below, so no UX is lost - the source just isn't pre-loaded.
      //
      // observation_spec / action_spec are kept because the Compat
      // column on the Models tab reads them directly from each row
      // to compare against the live env spec; they're tiny (a few
      // tens of bytes each) so they're not a payload concern.
      dbo.collection("models")
         .find({}, { projection: { reward_design_code: 0 } })
         .toArray(function(err, result) {
        if (err) throw err;
        //console.log(result);
        console.log(`${result.length} models retrieved`)
        res.json(result)
      });
      return;
    }
    console.log(`No models retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  // Per-model source fetch. Returns just the ``reward_design_code``
  // field for one model so the Models tab's source-view modal can
  // populate on click without paying the cost of preloading every
  // model's source through /get_models. Tiny payload (one document,
  // one field) so no projection/index tuning needed.
  //
  // 404 if the id doesn't parse as an ObjectId or no such document
  // exists. 200 with {code: null} when the model exists but has no
  // reward design recorded (legacy / RandomPyPolicy / etc.).
  app.get('/get_model_source/:id', (req, res) => {
    const idStr = String(req.params.id || '');
    let oid;
    try {
      oid = ObjectID(idStr);
    } catch (_e) {
      res.status(400).json({ error: 'invalid id' });
      return;
    }
    dbo.collection("models").findOne(
      { _id: oid },
      { projection: { reward_design_code: 1, reward_design_name: 1, reward_design_version: 1, reward_design_id: 1 } },
      function(err, doc) {
        if (err) {
          console.error('get_model_source failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        if (!doc) {
          res.status(404).json({ error: 'not found' });
          return;
        }
        res.json({
          _id: idStr,
          reward_design_id: doc.reward_design_id || null,
          reward_design_name: doc.reward_design_name || null,
          reward_design_version: doc.reward_design_version || null,
          reward_design_code: doc.reward_design_code || null,
        });
      });
  });

  // env_specs: one document per robot_type, written by
  // publish_env_spec() in rl_agent/robotaxi.py whenever a TRAIN or
  // EVAL job starts. The Models tab fetches this alongside
  // /get_models and /leaderboard_scores so the "Compat" column can
  // compare each model's stored observation/action spec against the
  // live env's. Same NO_CHANGES short-circuit as the other read-only
  // endpoints to keep the dashboard polling cheap.
  app.get('/get_env_specs', (req,res) => {
    if(needsUpdate(req, envSpecsChanged))
    {
      envSpecsChanged=false;
      dbo.collection("env_specs").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} env_specs retrieved`)
        res.json(result)
      });
      return;
    }
    console.log(`No env_specs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  // ---- reward_designs CRUD ----------------------------------------
  //
  // User-authored reward function modules (see rl_agent/reward_designs.py
  // for the contract). The dashboard's future Reward Design tab and the
  // New-Job form's reward-design dropdown both consume these.
  //
  // /get_reward_designs       : list current designs (with NO_CHANGES short-
  //                             circuit, same as other read endpoints)
  // /add_reward_design        : insert a new design
  // /update_reward_design     : update an existing design (bumps version)
  // /archive_reward_design    : soft-delete (archived: true) so historical
  //                             models keep resolving their reward design
  // /lint_reward_design       : compile-check the code via the sim-controller
  //                             container without saving anything

  app.get('/get_reward_designs', (req, res) => {
    if (needsUpdate(req, rewardDesignsChanged)) {
      rewardDesignsChanged = false;
      dbo.collection("reward_designs").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} reward_designs retrieved`);
        res.json(result);
      });
      return;
    }
    console.log(`No reward_designs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  app.post('/add_reward_design', (req, res) => {
    // Body shape: { name, description, code, author?, archived? (default false) }
    // We stamp created_at / updated_at / version=1 here so the client
    // never has to set them; subsequent edits go through
    // /update_reward_design which bumps version.
    const body = req.body || {};
    if (!body.name || !body.code) {
      res.status(400).json({ error: "name and code are required" });
      return;
    }
    const now = new Date();
    const doc = {
      name: String(body.name),
      description: String(body.description || ''),
      code: String(body.code),
      author: String(body.author || ''),
      archived: !!body.archived,
      version: 1,
      created_at: now,
      updated_at: now,
    };
    dbo.collection("reward_designs").insertOne(doc, function(err, result) {
      if (err) {
        console.error('add_reward_design failed:', err);
        res.status(500).json({ error: String(err.message || err) });
        return;
      }
      console.log('reward_design inserted', result.insertedId);
      res.json(result);
    });
  });

  app.post('/update_reward_design', (req, res) => {
    // Body shape: { _id, name?, description?, code?, archived? }
    // Edits bump version and updated_at. The previous version's code
    // is NOT preserved here (that would require a separate
    // reward_design_versions collection); historical models record
    // their training-time code on the model document itself, so a
    // user's accidental edit doesn't break reproducibility of
    // already-trained models.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      // Could be the string id used by the canonical seed (see
      // rl_agent/reward_designs.PASSTHROUGH_DESIGN_ID); accept either
      // shape so the dashboard can edit a seeded design too.
      idFilter = { "_id": String(body._id) };
    }
    const update: any = {
      "$set": { updated_at: new Date() },
      "$inc": { version: 1 },
    };
    if (body.name        !== undefined) update["$set"].name        = String(body.name);
    if (body.description !== undefined) update["$set"].description = String(body.description);
    if (body.code        !== undefined) update["$set"].code        = String(body.code);
    if (body.archived    !== undefined) update["$set"].archived    = !!body.archived;
    dbo.collection("reward_designs").updateOne(
      idFilter, update, { upsert: false },
      function(err, result) {
        if (err) {
          console.error('update_reward_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('reward_design updated', body._id, result);
        res.json(result);
      });
  });

  app.post('/archive_reward_design', (req, res) => {
    // Soft-delete. We do NOT physically remove the document because
    // historical model records reference reward_design_id; deleting
    // the design would orphan that reference. Archived designs are
    // hidden from the New-Job dropdown but still resolve when an
    // in-flight job looks them up.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      idFilter = { "_id": String(body._id) };
    }
    dbo.collection("reward_designs").updateOne(
      idFilter,
      { "$set": { archived: true, updated_at: new Date() } },
      { upsert: false },
      function(err, result) {
        if (err) {
          console.error('archive_reward_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('reward_design archived', body._id);
        res.json(result);
      });
  });

  // ---- experiment_designs CRUD + schema -------------------------- *
  //
  // The structured-config sibling of reward_designs. Each document is
  // a named bundle of training-loop hyperparameters (SAC learning
  // rates, BC pretrain steps, replay capacity, network sizes, ...).
  // Selected per-job via the New-Job form's "Experiment design"
  // dropdown alongside the existing Reward design dropdown. The
  // trainer (rl_agent/robotaxi.py::do_job) overlays the doc's fields
  // onto main()'s kwargs via experiment_designs.apply_to_main_kwargs.
  //
  // Endpoint set parallels /get_reward_designs etc. + adds a schema
  // discovery endpoint at /get_experiment_design_schema so the
  // dashboard form (and the future research-planning agent) can
  // introspect what fields exist without reading trainer source.
  //
  // IMPORTANT: the EXPERIMENT_DESIGN_SCHEMA constant below MUST be
  // kept in sync with rl_agent/experiment_designs.py::SCHEMA. The
  // Python module is the source of truth for what the trainer
  // honours; this JS mirror exists because the dashboard container
  // doesn't run Python and we don't want a docker-exec dependency.
  // When you add a field there, add it here too.
  const EXPERIMENT_DESIGN_SCHEMA: any[] = [
    { kind: 'section', label: 'Reinforcement learning loop' },
    { kind: 'field', name: 'num_iterations',              type: 'int',   default: 50000,  min: 1,      max: 10000000, doc: 'Total SAC training iterations after BC pretrain. Each iter = one collect step + one gradient update.', paper_ref: null, kwarg: 'num_iterations_val' },
    { kind: 'field', name: 'initial_collect_steps',       type: 'int',   default: 500,    min: 0,      max: 100000,   doc: 'Pre-RL random-policy collection steps to seed the replay buffer with diverse experience.',         paper_ref: null, kwarg: 'initial_collect_steps_val' },
    { kind: 'field', name: 'collect_steps_per_iteration', type: 'int',   default: 1,      min: 1,      max: 100,      doc: 'Env steps collected per training iteration (between gradient updates).',                            paper_ref: null, kwarg: 'collect_steps_per_iteration_val' },
    { kind: 'field', name: 'eval_interval',               type: 'int',   default: 5000,   min: 1,      max: 100000,   doc: 'How often (in train_steps) to pause training and run an in-loop eval cycle.',                       paper_ref: null, kwarg: 'eval_interval_val' },
    { kind: 'field', name: 'num_eval_episodes',           type: 'int',   default: 10,     min: 1,      max: 500,      doc: 'Eval episodes per in-loop eval cycle.',                                                              paper_ref: null, kwarg: 'num_eval_episodes_val' },
    { kind: 'field', name: 'log_interval',                type: 'int',   default: 5000,   min: 100,    max: 100000,   doc: 'TensorBoard scalar write cadence (in train_steps).',                                                 paper_ref: null, kwarg: 'log_interval_val' },
    { kind: 'field', name: 'policy_save_interval',        type: 'int',   default: 50,     min: 1,      max: 10000,    doc: 'How often (in train_steps) the PolicySavedModelTrigger writes a checkpoint.',                       paper_ref: null, kwarg: 'policy_save_interval_val' },
    { kind: 'section', label: 'Behavior cloning pretrain' },
    { kind: 'field', name: 'bc_pretrain_steps',           type: 'int',   default: 5000,   min: 0,      max: 1000000,  doc: 'BC gradient steps run on the actor before SAC starts. Set 0 to skip and run pure SAC.',             paper_ref: null, kwarg: 'bc_pretrain_steps_val' },
    { kind: 'section', label: 'Replay buffer' },
    { kind: 'field', name: 'replay_buffer_capacity',      type: 'int',   default: 75000,  min: 1000,   max: 10000000, doc: 'Max samples held in the online Reverb table (RL collection). Over capacity, FIFO eviction.',         paper_ref: null,       kwarg: 'replay_buffer_capacity_val' },
    { kind: 'field', name: 'batch_size',                  type: 'int',   default: 256,    min: 1,      max: 16384,    doc: 'SAC gradient-update batch size, also used by the BC pretrain phase.',                                paper_ref: null,       kwarg: 'batch_size_val' },
    { kind: 'field', name: 'demo_prefill_count',          type: 'int',   default: 50000,  min: 0,      max: 10000000, doc: 'Expert-demonstration steps to prefill the buffer with at job start. 0 = no demo prefill (pure SAC from random init).', paper_ref: '1707.08817', kwarg: 'demo_prefill_count_val' },
    { kind: 'field', name: 'demo_min_keep',               type: 'int',   default: 0,      min: 0,      max: 10000000, doc: 'Demo samples PROTECTED from FIFO eviction. 0 = single-table mode (demos pre-fill the online buffer and get FIFO-displaced by RL data over time, current default). >0 = two-table mode where this many demo samples live in a separate Reverb table that never gets new writes, so they stay forever.', paper_ref: '1704.03732', kwarg: 'demo_min_keep_val' },
    { kind: 'field', name: 'demo_sample_ratio',           type: 'float', default: 0.0,    min: 0.0,    max: 1.0,      doc: 'Two-table mode only (demo_min_keep > 0): fraction of each training batch drawn from the demo table vs the online table. 0.0 = pure online sampling (demos kept but never sampled). 1.0 = pure demo sampling. Typical 0.1-0.3 for DDPGfD-style demo over-sampling.', paper_ref: '1707.08817', kwarg: 'demo_sample_ratio_val' },
    { kind: 'section', label: 'SAC optimizer' },
    { kind: 'field', name: 'actor_learning_rate',         type: 'float', default: 3e-5,   min: 1e-7,   max: 1.0,      doc: 'Adam learning rate for the actor network.',                                                          paper_ref: null, kwarg: 'actor_learning_rate_val' },
    { kind: 'field', name: 'critic_learning_rate',        type: 'float', default: 3e-5,   min: 1e-7,   max: 1.0,      doc: 'Adam learning rate for the critic (twin Q-network).',                                                paper_ref: null, kwarg: 'critic_learning_rate_val' },
    { kind: 'field', name: 'alpha_learning_rate',         type: 'float', default: 3e-5,   min: 1e-7,   max: 1.0,      doc: 'Adam learning rate for the temperature parameter (entropy coefficient).',                            paper_ref: null, kwarg: 'alpha_learning_rate_val' },
    { kind: 'field', name: 'target_update_tau',           type: 'float', default: 0.005,  min: 0.0,    max: 1.0,      doc: 'Polyak averaging factor for the target critic. Typical SAC value 0.005.',                            paper_ref: null, kwarg: 'target_update_tau_val' },
    { kind: 'field', name: 'target_update_period',        type: 'int',   default: 1,      min: 1,      max: 10000,    doc: 'Update the target critic every N train_steps (Polyak averaging cadence).',                          paper_ref: null, kwarg: 'target_update_period_val' },
    { kind: 'field', name: 'gamma',                       type: 'float', default: 0.99,   min: 0.0,    max: 1.0,      doc: 'Discount factor for future rewards in the Bellman target.',                                          paper_ref: null, kwarg: 'gamma_val' },
    { kind: 'field', name: 'reward_scale_factor',         type: 'float', default: 1.0,    min: 0.0,    max: 1000.0,   doc: 'Multiplier applied to environment rewards before they enter the Q-target. SAC is sensitive to this.', paper_ref: null, kwarg: 'reward_scale_factor_val' },
    { kind: 'section', label: 'Network architecture' },
    { kind: 'field', name: 'actor_fc_layers_x',           type: 'int',   default: 512,    min: 1,      max: 8192,     doc: 'First-layer width of the actor MLP.',                                                                paper_ref: null, kwarg: 'actor_fc_layer_params_x' },
    { kind: 'field', name: 'actor_fc_layers_y',           type: 'int',   default: 512,    min: 1,      max: 8192,     doc: 'Second-layer width of the actor MLP.',                                                               paper_ref: null, kwarg: 'actor_fc_layer_params_y' },
    { kind: 'field', name: 'critic_fc_layers_x',          type: 'int',   default: 512,    min: 1,      max: 8192,     doc: 'First-layer width of the critic joint MLP (after obs+action concatenation).',                       paper_ref: null, kwarg: 'critic_joint_fc_layer_params_x' },
    { kind: 'field', name: 'critic_fc_layers_y',          type: 'int',   default: 512,    min: 1,      max: 8192,     doc: 'Second-layer width of the critic joint MLP.',                                                        paper_ref: null, kwarg: 'critic_joint_fc_layer_params_y' },
  ];

  app.get('/get_experiment_design_schema', (req, res) => {
    // Self-describing schema for the experiment_designs collection.
    // Returns the field list in form-render order. Consumers (the
    // New-Job form, the future Experiment Design tab UI, the
    // research-planning agent) use this to build inputs / validate
    // submissions without reading trainer source.
    res.json({ fields: EXPERIMENT_DESIGN_SCHEMA });
  });

  app.get('/get_experiment_designs', (req, res) => {
    if (needsUpdate(req, experimentDesignsChanged)) {
      experimentDesignsChanged = false;
      dbo.collection("experiment_designs").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} experiment_designs retrieved`);
        res.json(result);
      });
      return;
    }
    console.log(`No experiment_designs retrieved`);
    res.status(200).send('NO_CHANGES');
  });

  app.post('/add_experiment_design', (req, res) => {
    // Body shape: { name, description, ...field_overrides, author?, archived? }
    // Field overrides are spread directly onto the doc so any key
    // present in the schema becomes a top-level field on the
    // experiment_designs document. We do NOT validate field types
    // here - the trainer's apply_to_main_kwargs() does soft
    // coercion + clamping with logging, which is the right place
    // for that since it has access to the SCHEMA min/max.
    const body = req.body || {};
    if (!body.name) {
      res.status(400).json({ error: "name is required" });
      return;
    }
    const now = new Date();
    const doc: any = {
      name: String(body.name),
      description: String(body.description || ''),
      author: String(body.author || ''),
      archived: !!body.archived,
      version: 1,
      created_at: now,
      updated_at: now,
    };
    // Allow callers to omit the field-override block entirely (= all
    // nulls = trainer defaults), or pass them as a flat top-level
    // spread. We accept both shapes for ergonomics.
    const fieldOverrides = body.fields && typeof body.fields === 'object'
      ? body.fields
      : body;
    for (const entry of EXPERIMENT_DESIGN_SCHEMA) {
      if (entry.kind !== 'field') continue;
      const name = entry.name;
      if (fieldOverrides[name] !== undefined) doc[name] = fieldOverrides[name];
    }
    dbo.collection("experiment_designs").insertOne(doc, function(err, result) {
      if (err) {
        console.error('add_experiment_design failed:', err);
        res.status(500).json({ error: String(err.message || err) });
        return;
      }
      console.log('experiment_design inserted', result.insertedId);
      res.json(result);
    });
  });

  app.post('/update_experiment_design', (req, res) => {
    // Body shape: { _id, name?, description?, archived?, ...field_overrides }
    // Edits bump version + updated_at. Same accept-either-shape on
    // field overrides as /add_experiment_design.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      idFilter = { "_id": String(body._id) };  // canonical "Default" uses string _id
    }
    const update: any = {
      "$set": { updated_at: new Date() },
      "$inc": { version: 1 },
    };
    if (body.name        !== undefined) update["$set"].name        = String(body.name);
    if (body.description !== undefined) update["$set"].description = String(body.description);
    if (body.archived    !== undefined) update["$set"].archived    = !!body.archived;
    const fieldOverrides = body.fields && typeof body.fields === 'object'
      ? body.fields
      : body;
    for (const entry of EXPERIMENT_DESIGN_SCHEMA) {
      if (entry.kind !== 'field') continue;
      const name = entry.name;
      if (fieldOverrides[name] !== undefined) update["$set"][name] = fieldOverrides[name];
    }
    dbo.collection("experiment_designs").updateOne(
      idFilter, update, { upsert: false },
      function(err, result) {
        if (err) {
          console.error('update_experiment_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('experiment_design updated', body._id, result);
        res.json(result);
      });
  });

  app.post('/archive_experiment_design', (req, res) => {
    // Soft-delete. Same rationale as /archive_reward_design: archived
    // designs stay in the collection for backward-compat with model
    // documents that reference them; they just don't appear in the
    // New-job dropdown by default.
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try {
      idFilter = { "_id": ObjectID(body._id) };
    } catch (e) {
      idFilter = { "_id": String(body._id) };
    }
    dbo.collection("experiment_designs").updateOne(
      idFilter,
      { "$set": { archived: true, updated_at: new Date() } },
      { upsert: false },
      function(err, result) {
        if (err) {
          console.error('archive_experiment_design failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('experiment_design archived', body._id);
        res.json(result);
      });
  });

  // ---------------------------------------------------------------- *
  // Gyms CRUD
  //
  // A "gym" is a registered Unity scene / environment configuration:
  //   name      : human-friendly label shown in job dropdowns
  //   file_path : absolute path to the scene/config file on the host
  //   description: optional notes
  //
  // /get_gyms      : list all (non-archived) gyms
  // /add_gym       : register a new gym
  // /update_gym    : rename / change path / description
  // /archive_gym   : soft-delete
  // ---------------------------------------------------------------- *

  app.get('/get_gyms', (req, res) => {
    if (needsUpdate(req, gymsChanged)) {
      gymsChanged = false;
      dbo.collection("gyms").find({}).toArray(function(err, result) {
        if (err) throw err;
        console.log(`${result.length} gyms retrieved`);
        res.json(result);
      });
      return;
    }
    console.log('No gyms retrieved');
    res.status(200).send('NO_CHANGES');
  });

  // Strip a single pair of surrounding quotes from a pasted path.
  // Windows Explorer's "Copy as path" wraps the value in double quotes;
  // storing those breaks Test-Path / robocopy on the supervisor side.
  function stripQuotes(s: string): string {
    const t = String(s == null ? '' : s).trim();
    if (t.length >= 2 && t[0] === t[t.length - 1] && (t[0] === '"' || t[0] === "'")) {
      return t.slice(1, -1).trim();
    }
    return t;
  }

  app.post('/add_gym', (req, res) => {
    const body = req.body || {};
    if (!body.name || !body.file_path) {
      res.status(400).json({ error: "name and file_path are required" });
      return;
    }
    const now = new Date();
    const doc = {
      name:        String(body.name),
      file_path:   stripQuotes(body.file_path),
      description: String(body.description || ''),
      archived:    !!body.archived,
      created_at:  now,
      updated_at:  now,
    };
    dbo.collection("gyms").insertOne(doc, function(err, result) {
      if (err) {
        console.error('add_gym failed:', err);
        res.status(500).json({ error: String(err.message || err) });
        return;
      }
      console.log('gym inserted', result.insertedId);
      res.json(result);
    });
  });

  app.post('/update_gym', (req, res) => {
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try { idFilter = { "_id": ObjectID(body._id) }; }
    catch (e) { idFilter = { "_id": String(body._id) }; }
    const setFields: any = { updated_at: new Date() };
    if (body.name        !== undefined) setFields.name        = String(body.name);
    if (body.file_path   !== undefined) setFields.file_path   = stripQuotes(body.file_path);
    if (body.description !== undefined) setFields.description = String(body.description);
    if (body.archived    !== undefined) setFields.archived    = !!body.archived;
    dbo.collection("gyms").updateOne(
      idFilter, { "$set": setFields }, { upsert: false },
      function(err, result) {
        if (err) {
          console.error('update_gym failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        res.json(result);
      });
  });

  app.post('/archive_gym', (req, res) => {
    const body = req.body || {};
    if (!body._id) {
      res.status(400).json({ error: "_id is required" });
      return;
    }
    let idFilter;
    try { idFilter = { "_id": ObjectID(body._id) }; }
    catch (e) { idFilter = { "_id": String(body._id) }; }
    dbo.collection("gyms").updateOne(
      idFilter,
      { "$set": { archived: true, updated_at: new Date() } },
      { upsert: false },
      function(err, result) {
        if (err) {
          console.error('archive_gym failed:', err);
          res.status(500).json({ error: String(err.message || err) });
          return;
        }
        console.log('gym archived', body._id);
        res.json(result);
      });
  });

  app.post('/lint_reward_design', (req, res) => {
    // Cheap structural lint of a candidate reward-design code string.
    //
    // The authoritative compile-check is what `load_reward_design()` does
    // inside the sim-controller (rl_agent/reward_designs.py): exec'd in
    // a restricted namespace, with the right argument-set enforced. We
    // can't reach Python from inside this Node container without
    // adding docker-CLI plus a socket mount, so we approximate just
    // enough to give the editor useful feedback before save:
    //
    //   - reject obviously empty input,
    //   - require at least one of the three reward function names,
    //   - flag use of obviously-disallowed builtins so users see them
    //     here instead of waiting until job start.
    //
    // A pass here still doesn't guarantee the design loads cleanly;
    // any syntactic error caught only by Python's compile() will surface
    // at the next job pickup as a RewardDesignError on that job (see
    // do_job's TRAIN branch in rl_agent/robotaxi.py).
    const body = req.body || {};
    if (typeof body.code !== 'string') {
      res.status(400).json({ ok: false, error: "code (string) is required" });
      return;
    }
    const code: string = body.code;
    if (!code.trim()) {
      res.json({ ok: false, error: "code is empty", funcs: [] });
      return;
    }
    const allowedFuncs = ['reward_standard', 'reward_success', 'reward_failure'];
    const found: string[] = [];
    for (const name of allowedFuncs) {
      // top-level `def reward_X(...)` declaration. We match any
      // amount of leading whitespace except a leading non-blank
      // character so we still catch a deliberately-indented def
      // (which would mean the function isn't top-level and the loader
      // would skip it anyway, but the user can see they're warned).
      const re = new RegExp('(^|\\n)\\s*def\\s+' + name + '\\s*\\(', 'm');
      if (re.test(code)) found.push(name);
    }
    if (!found.length) {
      res.json({
        ok: false,
        error: "no reward functions defined. Expected one or more of: " +
               allowedFuncs.join(', '),
        funcs: [],
      });
      return;
    }
    // Look for disallowed module access. This is a *hint*, not a
    // security boundary - the real sandbox is the restricted exec
    // builtins in load_reward_design(). We just want to fail fast
    // here when someone tries `import os` because the trainer would
    // also reject it.
    const disallowedImports = [
      /(^|\n)\s*import\s+(os|sys|subprocess|socket|requests|urllib)\b/,
      /(^|\n)\s*from\s+(os|sys|subprocess|socket|requests|urllib)\b/,
    ];
    for (const re of disallowedImports) {
      const m = code.match(re);
      if (m) {
        res.json({
          ok: false,
          error: "import of restricted module: " + m[2] +
                 " (the trainer's sandbox blocks os / sys / subprocess / sockets / network).",
          funcs: found,
        });
        return;
      }
    }
    res.json({ ok: true, error: null, funcs: found });
  });

  // ---- TensorBoard comparison bucket ------------------------------
  //
  // /tmp/tb_compare/ is the on-demand bucket scanned by the
  // TensorBoard server (see sim-controller's --logdir_spec compare:
  // ... in docker-compose.yml). The dashboard manages its contents
  // through this endpoint:
  //
  //   POST /set_tb_compare_jobs  { jobIds: [...] }
  //     1. ensure /tmp/tb_compare exists (mkdir -p)
  //     2. remove every existing symlink (NOT real directories - the
  //        bucket should only contain symlinks, but be defensive)
  //     3. for each jobId, symlink /tmp/jobsdata/<jobId> ->
  //        /tmp/tb_compare/<jobId> if the source exists
  //     4. return { linked: [...], missing: [...] }
  //
  // The dashboard's Models -> Compare flow calls this BEFORE
  // re-pointing the TB iframe at a filter URL, so TB sees exactly
  // the user's selection on next reload_interval (~5s).
  //
  // /tmp is bind-mounted from the host into BOTH the dashboard and
  // sim-controller containers via docker-compose.yml's `source:
  // ../tmp; target: /tmp` blocks, so a symlink we create here is
  // visible to TB inside sim-controller.
  app.post('/set_tb_compare_jobs', (req, res) => {
    const COMPARE_ROOT = '/tmp/tb_compare';
    const ARCHIVED_ROOT = '/tmp/jobsdata';

    const body = req.body || {};

    // Two accepted request shapes:
    //   Legacy: { jobIds: ["<oid>", ...] }
    //     -> symlink at /tmp/tb_compare/<oid>; TB run name becomes
    //        "compare/<oid>/<sub>" - opaque, hard to map back to
    //        the Analysis-tab slot the user sees on the right.
    //   Labeled: { jobs: [{ jobId: "<oid>", label: "base" }, ...] }
    //     -> symlink at /tmp/tb_compare/<label>_<short_jobId>; TB
    //        run becomes "compare/base_6a09ddd9/<sub>" - the operator
    //        can read it off the sidebar and immediately know which
    //        Analysis-tab column the run belongs to.
    //
    // The labeled shape is what the goldenlayout shell sends after
    // sorting selected models by create_date. The legacy shape is
    // still supported as a fallback (e.g., manual curl, older
    // dashboard builds).
    type JobEntry = { jobId: string; label: string | null };
    const safeIdRe = /^[a-zA-Z0-9_-]{1,64}$/;
    // Labels are even more permissive than ids but still bounded;
    // we use them to name a filesystem dir so we strip anything that
    // could break the path (no slashes, dots, etc.).
    const safeLabelRe = /^[a-zA-Z0-9_-]{1,32}$/;
    let entries: JobEntry[] = [];
    if (Array.isArray(body.jobs)) {
      entries = (body.jobs as any[])
        .map((j: any) => ({
          jobId: String((j && j.jobId) || '').trim(),
          label: (j && j.label && safeLabelRe.test(String(j.label)))
            ? String(j.label) : null,
        }))
        .filter((e) => e.jobId.length > 0 && safeIdRe.test(e.jobId));
    } else {
      const rawIds = Array.isArray(body.jobIds) ? body.jobIds : [];
      entries = rawIds
        .map((x: any) => String(x || '').trim())
        .filter((s: string) => s.length > 0 && safeIdRe.test(s))
        .map((jobId: string) => ({ jobId, label: null }));
    }
    const jobIds: string[] = entries.map((e) => e.jobId);

    try {
      fs.mkdirSync(COMPARE_ROOT, { recursive: true });
    } catch (e: any) {
      // mkdir failure is fatal - we can't manage the bucket.
      res.status(500).json({
        ok: false,
        error: `mkdir ${COMPARE_ROOT} failed: ${String(e && e.message || e)}`,
        linked: [], missing: [],
      });
      return;
    }

    // Sweep the bucket. We ONLY remove symlinks (and broken-symlink
    // dangling entries). If somehow a real directory ended up in
    // here, leave it - we shouldn't be deleting real data the
    // dashboard didn't put here.
    try {
      const entries = fs.readdirSync(COMPARE_ROOT);
      for (const name of entries) {
        const p = path.join(COMPARE_ROOT, name);
        let st: any;
        try { st = fs.lstatSync(p); }
        catch (_e) { continue; }
        if (st.isSymbolicLink()) {
          try { fs.unlinkSync(p); }
          catch (e) {
            console.warn(`set_tb_compare_jobs: failed to unlink ${p}: ${e}`);
          }
        }
      }
    } catch (e) {
      // readdir failure isn't fatal; we'll just stack new symlinks
      // alongside whatever was there.
      console.warn(`set_tb_compare_jobs: readdir ${COMPARE_ROOT} failed: ${e}`);
    }

    const linked: string[] = [];
    const missing: string[] = [];
    // jobId -> on-disk symlink basename so the caller can mirror the
    // mapping into its TB regex filter.
    const linkNames: { [jobId: string]: string } = {};
    for (const entry of entries) {
      const id = entry.jobId;
      // move_all_jobs_data() in rl_agent/robotaxi.py archives
      // /tmp/active/<id>/{eval,train,learner,metrics}/ AS
      // /tmp/jobsdata/<id>/<id>/{eval,train,learner,metrics}/ -
      // doubly nested - because the function groups everything for
      // a job under its own subdir of /tmp/jobsdata/<id>/.
      //
      // We want TB run names to look like "compare/<label>_<short>/eval"
      // (parallel to "current/<id>/eval" for the live job), not
      // "compare/<id>/<id>/eval". So if the inner doubled-up dir
      // exists, symlink to THAT; otherwise fall back to the outer
      // dir (covers any legacy single-nested archives + a future
      // refactor of move_all_jobs_data that flattens the layout).
      const outerSrc = path.join(ARCHIVED_ROOT, id);
      const innerSrc = path.join(outerSrc, id);
      let src: string;
      if (fs.existsSync(innerSrc) && fs.statSync(innerSrc).isDirectory()) {
        src = innerSrc;
      } else if (fs.existsSync(outerSrc) && fs.statSync(outerSrc).isDirectory()) {
        src = outerSrc;
      } else {
        // Could be a model whose job was never archived (e.g., the
        // currently-training job - its data is in /tmp/active, not
        // /tmp/jobsdata, so the regex filter on the TB iframe URL
        // will hit it under the "current" experiment instead).
        missing.push(id);
        continue;
      }
      // Compose the symlink basename. With a label: e.g.
      // "base_0f407569" (label + last 8 chars of jobId for visual
      // disambiguation when two labels match - shouldn't happen in
      // normal use, but the short-id suffix keeps the symlink
      // stably distinct).
      //
      // Last 8 chars (not first 8) so this matches the rest of the
      // dashboard: formatters.shortId in dashboard/components.js
      // also uses slice(-8) for table cells like the Models tab's
      // ID column. That way an operator who notes "model
      // ...0f407569" on the Models tab can immediately recognize
      // "base_0f407569" in the TB sidebar without mental
      // remapping.
      //
      // Without a label: the full jobId (legacy back-compat).
      const dstName = entry.label
        ? `${entry.label}_${id.slice(-8)}`
        : id;
      const dst = path.join(COMPARE_ROOT, dstName);
      try {
        // Type 'dir' makes Windows happy if this ever runs on a
        // Windows container; on Linux it's ignored. fs.symlinkSync
        // is sync because the whole endpoint is sub-100ms and an
        // async chain isn't worth the complexity.
        fs.symlinkSync(src, dst, 'dir');
        linked.push(id);
        linkNames[id] = dstName;
      } catch (e: any) {
        console.error(`set_tb_compare_jobs: symlink ${src} -> ${dst} failed: ${e}`);
        missing.push(id);
      }
    }

    res.json({ ok: true, linked, missing, linkNames });
  });

  app.get('/', (req, res) => {
    const indexPagePath: string = path.join(__dirname, '/../index.html');
    fs.access(indexPagePath, (err) => {
      if (err) {
        log(LogLevel.warn, `Can't find file ' ${indexPagePath}`);
        res.status(404).send(`Can't find file ${indexPagePath}`);
      } else {
        res.sendFile(indexPagePath);
      }
    });
  });
  return app;
};
// create websocket server for log tailing (used by logs.html)
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  const dockerLogs = spawn('tail', ['-f', '-n', '100', '/python_ws/src/robotaxi.out']);

  dockerLogs.stdout.on('data', (data) => {
    ws.send(data.toString());
  });

  dockerLogs.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  dockerLogs.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });

  ws.on('close', () => {
    dockerLogs.kill();
  });
});

